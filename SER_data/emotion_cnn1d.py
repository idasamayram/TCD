import os
import glob
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    SR = 22050
    DURATION = 2.5
    OFFSET = 0.6  # Per your Kaggle reference
    TARGET_T = 108  # Expected time frames for 2.5s audio
    BATCH_SIZE = 64
    EPOCHS = 30
    LR = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = 'emotion_best_model.ckpt'


# ==========================================
# 2. FEATURE EXTRACTION (TIME-PRESERVED)
# ==========================================
def extract_features(data, sample_rate):
    """
    Extracts 5 features but DOES NOT use np.mean().
    We must preserve the time axis for the 1D CNN and future explainability.
    """
    stft = np.abs(librosa.stft(data))

    # 1. ZCR (Shape: 1 x T)
    zcr = librosa.feature.zero_crossing_rate(y=data)
    # 2. Chroma STFT (Shape: 12 x T)
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    # 3. MFCC (Shape: 40 x T)
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
    # 4. RMS (Shape: 1 x T)
    rms = librosa.feature.rms(y=data)
    # 5. Mel Spectrogram (Shape: 128 x T)
    mel = librosa.feature.melspectrogram(y=data, sr=sample_rate)

    # Stack vertically along the feature dimension -> Total: 182 x T
    features = np.vstack((zcr, chroma, mfcc, rms, mel))

    # Pad or truncate to ensure uniform time dimension for CNN batching
    if features.shape[1] > Config.TARGET_T:
        features = features[:, :Config.TARGET_T]
    else:
        features = np.pad(features, ((0, 0), (0, Config.TARGET_T - features.shape[1])), mode='constant')

    return features


# --- Augmentation Functions ---
def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise


def stretch_pitch(data, sample_rate, rate=0.8, n_steps=2):
    """Combines stretch and pitch into one augmentation."""
    stretched = librosa.effects.time_stretch(data, rate=rate)
    pitched = librosa.effects.pitch_shift(stretched, sr=sample_rate, n_steps=n_steps)
    return pitched


# ==========================================
# 3. DATA PARSING & UPFRONT EXTRACTION
# ==========================================
def parse_datasets():
    """Parses directories (Assuming CREMA-D, RAVDESS, SAVEE, TESS are in root)."""
    data = []
    # CREMA-D
    crema_map = {'NEU': 'neutral', 'HAP': 'happy', 'SAD': 'sad', 'ANG': 'angry', 'FEA': 'fear', 'DIS': 'disgust'}
    for path in glob.glob('CREMA-D/AudioWAV/*.wav'):
        code = os.path.basename(path).split('_')[2]
        if code in crema_map: data.append([path, crema_map[code]])
    # RAVDESS
    rav_map = {'01': 'neutral', '02': 'neutral', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fear',
               '07': 'disgust', '08': 'surprise'}
    for path in glob.glob('RAVDESS/Actor_*/*.wav'):
        code = os.path.basename(path).split('-')[2]
        if code in rav_map: data.append([path, rav_map[code]])
    # SAVEE
    sav_map = {'a': 'angry', 'd': 'disgust', 'f': 'fear', 'h': 'happy', 'n': 'neutral', 'sa': 'sad', 'su': 'surprise'}
    for path in glob.glob('SAVEE/ALL/*.wav'):
        code = ''.join([c for c in os.path.basename(path).split('_')[1] if not c.isdigit()]).replace('.wav', '')
        if code in sav_map: data.append([path, sav_map[code]])
    # TESS
    tess_map = {'angry': 'angry', 'disgust': 'disgust', 'fear': 'fear', 'happy': 'happy', 'neutral': 'neutral',
                'ps': 'surprise', 'sad': 'sad'}
    for path in glob.glob('TESS/*/*/*.wav'):
        code = os.path.basename(path).replace('.wav', '').split('_')[-1].lower()
        if code in tess_map: data.append([path, tess_map[code]])

    return pd.DataFrame(data, columns=['Path', 'Emotion'])


def prepare_data_arrays(df, is_train=False):
    """
    Extracts features for all files in the dataframe.
    ONLY applies augmentation if is_train=True to prevent data leakage.
    """
    X, Y = [], []

    print(f"Extracting features for {len(df)} files... (Augmentation: {is_train})")
    for idx, row in df.iterrows():
        try:
            data, sr = librosa.load(row['Path'], duration=Config.DURATION, offset=Config.OFFSET)

            # Base feature extraction
            X.append(extract_features(data, sr))
            Y.append(row['Label'])

            # Only augment the training set!
            if is_train:
                # 1. Noise
                noise_data = add_noise(data)
                X.append(extract_features(noise_data, sr))
                Y.append(row['Label'])

                # 2. Stretch & Pitch
                sp_data = stretch_pitch(data, sr)
                X.append(extract_features(sp_data, sr))
                Y.append(row['Label'])

        except Exception as e:
            continue  # Skip corrupted files quietly

    return np.array(X), np.array(Y)


# ==========================================
# 4. STATISTICS & VISUALIZATION
# ==========================================
def plot_feature_statistics(X_train):
    print("\n--- Feature Statistics ---")
    print(f"Extracted Training Matrix Shape: {X_train.shape} -> (Batch, Features, Time)")
    print(f"Global Mean: {np.mean(X_train):.4f}")
    print(f"Global Std: {np.std(X_train):.4f}")

    # Plot the 182-feature map of the very first sample
    sample_feature_map = X_train[0]

    plt.figure(figsize=(12, 6))
    sns.heatmap(sample_feature_map, cmap='viridis', cbar=True)
    plt.title('2D Feature Representation (ZCR, Chroma, MFCC, RMS, Mel)\nTime Axis Preserved for Explainability')
    plt.xlabel('Time Frames')
    plt.ylabel('Stacked Feature Channels (182 total)')
    plt.tight_layout()
    plt.savefig('feature_map_check.png')
    plt.close()
    print("Saved feature map visualization to 'feature_map_check.png'\n")


# ==========================================
# 5. DATASET & MODEL
# ==========================================
class PreExtractedDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class AudioCNN1D(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN1D, self).__init__()

        # in_channels is now 182 (1 + 12 + 40 + 1 + 128)
        self.conv1 = nn.Conv1d(in_channels=182, out_channels=128, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.drop2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.drop3 = nn.Dropout(0.3)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.drop4 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = self.drop2(self.pool2(F.relu(self.conv2(x))))
        x = self.drop3(self.pool3(F.relu(self.conv3(x))))

        x = self.global_avg_pool(x).squeeze(-1)

        x = F.relu(self.fc1(x))
        x = self.drop4(x)
        x = self.fc2(x)
        return x


# ==========================================
# 6. PIPELINE EXECUTION
# ==========================================
def run_pipeline():
    df = parse_datasets()

    encoder = LabelEncoder()
    df['Label'] = encoder.fit_transform(df['Emotion'])
    num_classes = len(encoder.classes_)

    # 1. SPLIT FIRST (To prevent data leakage)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Label'])

    # 2. EXTRACT AND AUGMENT UPFRONT
    print("\n--- Processing Training Set ---")
    X_train, Y_train = prepare_data_arrays(train_df, is_train=True)

    print("\n--- Processing Testing Set ---")
    X_test, Y_test = prepare_data_arrays(test_df, is_train=False)

    plot_feature_statistics(X_train)

    # 3. DATALOADERS
    train_loader = DataLoader(PreExtractedDataset(X_train, Y_train), batch_size=Config.BATCH_SIZE, shuffle=True,
                              pin_memory=True)
    test_loader = DataLoader(PreExtractedDataset(X_test, Y_test), batch_size=Config.BATCH_SIZE, shuffle=False,
                             pin_memory=True)

    # 4. TRAINING
    model = AudioCNN1D(num_classes).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0.0

    print("--- Starting Training ---")
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_acc = correct_train / total_train

        # Validation
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val
        print(
            f"Epoch {epoch + 1:02d} | Train Loss: {train_loss / total_train:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss / total_val:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f" [*] Saved new best model -> {Config.MODEL_PATH}")


if __name__ == '__main__':
    run_pipeline()