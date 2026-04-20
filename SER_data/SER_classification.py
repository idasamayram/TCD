
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ==========================================================
# DATASET PATHS (EDIT THESE)
# ==========================================================
import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


############################################
# PATHS
############################################

CREMA_PATH = "CREMA-D/AudioWAV"
RAVDESS_PATH = "RAVDESS"
SAVEE_PATH = "SAVEE/ALL"
TESS_PATH = "TESS/TESS Toronto emotional speech set data"

SAMPLE_RATE = 16000
MAX_PAD_LEN = 200


############################################
# LABEL MAPS
############################################

crema_map = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

ravdess_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}

savee_map = {
    "a": "angry",
    "d": "disgust",
    "f": "fear",
    "h": "happy",
    "sa": "sad",
    "su": "surprise",
    "n": "neutral"
}


############################################
# DATA AUGMENTATION
############################################

def add_noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.max(data)
    noise = noise_amp * np.random.normal(size=data.shape[0])
    return data + noise


def shift(data):
    shift_range = int(np.random.uniform(-5, 5) * 1000)
    return np.roll(data, shift_range)


def pitch(data, sr):
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=0.7)


############################################
# FEATURE EXTRACTION
############################################

def extract_features(signal, sr):


    if len(signal) < 512:
        signal = np.pad(signal, (0, 512 - len(signal)), mode='constant')
    else:
        signal = signal

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)

    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
    mel = librosa.feature.melspectrogram(y=signal, sr=sr)

    contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=sr)


    features = np.vstack([mfcc, delta, chroma, mel, contrast, tonnetz])

    if features.shape[1] < MAX_PAD_LEN:
        pad = MAX_PAD_LEN - features.shape[1]
        features = np.pad(features, ((0,0),(0,pad)), mode="constant")
    else:
        features = features[:, :MAX_PAD_LEN]

    return features


############################################
# DATASET LOADERS
############################################

def load_crema():

    data = []

    for file in Path(CREMA_PATH).glob("*.wav"):

        emotion_code = file.name.split("_")[2]

        if emotion_code in crema_map:
            emotion = crema_map[emotion_code]
            data.append((str(file), emotion, "CREMA"))

    return data


def load_ravdess():

    data = []

    for actor in Path(RAVDESS_PATH).glob("Actor_*"):

        for file in actor.glob("*.wav"):

            emotion = file.name.split("-")[2]
            emotion = ravdess_map[emotion]

            data.append((str(file), emotion, "RAVDESS"))

    return data


def load_savee():
    data = []
    for file in Path(SAVEE_PATH).glob("*.wav"):
        name = file.stem  # e.g., "DC_a01"
        match = re.search(r"_([a-z]+)", name)
        if match:
            code = match.group(1)
            if code in savee_map:
                emotion = savee_map[code]
                data.append((str(file), emotion, "SAVEE"))
            else:
                print(f"Skipping {file} due to unknown emotion code: {code}")
        else:
            print(f"Skipping {file} due to no valid emotion code")
    return data

def load_tess():
    data = []
    for folder in Path(TESS_PATH).iterdir():
        if folder.is_dir():
            emotion = folder.name.split("_")[-1]  # last part
            for file in folder.glob("*.wav"):
                data.append((str(file), emotion, "TESS"))
    return data

############################################
# LOAD ALL DATA
############################################

print("Loading datasets...")

data = []
data += load_crema()
data += load_ravdess()
data += load_savee()
data += load_tess()

df = pd.DataFrame(data, columns=["path", "emotion", "dataset"])

print("Total samples:", len(df))


############################################
# DATA ANALYSIS
############################################

plt.figure()
df["emotion"].value_counts().plot(kind="bar")
plt.title("Emotion Distribution")
plt.show()

plt.figure()
df["dataset"].value_counts().plot(kind="bar")
plt.title("Dataset Distribution")
plt.show()


############################################
# FEATURE EXTRACTION WITH AUGMENTATION
############################################

print("Extracting features...")

X = []
y = []

for path, emotion in zip(df.path, df.emotion):

    try:

        signal, sr = librosa.load(path, sr=SAMPLE_RATE)

        feats = extract_features(signal, sr)
        X.append(feats)
        y.append(emotion)

        noise_signal = add_noise(signal)
        feats = extract_features(noise_signal, sr)
        X.append(feats)
        y.append(emotion)

        shift_signal = shift(signal)
        feats = extract_features(shift_signal, sr)
        X.append(feats)
        y.append(emotion)

        pitch_signal = pitch(signal, sr)
        feats = extract_features(pitch_signal, sr)
        X.append(feats)
        y.append(emotion)

    except:
        continue


X = np.array(X)
y = np.array(y)

print("Dataset after augmentation:", X.shape)


############################################
# LABEL ENCODING
############################################

encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("Classes:", encoder.classes_)


############################################
# SPLIT
############################################

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp
)


############################################
# PYTORCH DATASET
############################################

class EmotionDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


train_loader = DataLoader(
    EmotionDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    EmotionDataset(X_val, y_val),
    batch_size=32
)

test_loader = DataLoader(
    EmotionDataset(X_test, y_test),
    batch_size=32
)


############################################
# CNN1D MODEL
############################################

class CNN1D(nn.Module):

    def __init__(self, n_classes):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv1d(193, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 32, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):

        x = self.conv(x)
        x = x.squeeze(-1)

        return self.fc(x)


device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN1D(len(encoder.classes_)).to(device)


############################################
# TRAINING
############################################

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 30

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for xb, yb in train_loader:

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        preds = model(xb)

        loss = criterion(preds, yb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss {total_loss/len(train_loader):.4f}")


############################################
# SAVE MODEL
############################################

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "label_encoder": encoder
    },
    "speech_emotion_cnn1d.ckpt"
)

print("Model saved as speech_emotion_cnn1d.ckpt")

