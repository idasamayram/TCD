import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import librosa.display
import random
import soundfile as sf



sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_lengths(data_path, sr=16000):
    """Analyze audio file lengths in the dataset"""
    lengths = []
    wav_files = []


    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))

                file_path = os.path.join(root, file)
                audio, _ = librosa.load(file_path, sr=sr)
                lengths.append(len(audio))

    lengths = np.array(lengths)


    # print("Total files:", len(wav_files)) # same as below

    print(f"Number of files: {len(lengths)}")
    print(f"Min length: {lengths.min()} samples ({lengths.min() / sr:.2f} sec)")
    print(f"Max length: {lengths.max()} samples ({lengths.max() / sr:.2f} sec)")
    print(f"Mean length: {lengths.mean():.0f} samples ({lengths.mean() / sr:.2f} sec)")
    print(f"Median length: {np.median(lengths):.0f} samples ({np.median(lengths) / sr:.2f} sec)")

    # Plot distribution
    plt.figure(figsize=(10, 4))
    plt.hist(lengths / sr, bins=50, edgecolor='black')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.title('Audio Length Distribution')
    plt.show()

    return wav_files, lengths



def visualize_waveforms(files, n_examples=6, sr=16000):

    chosen = random.sample(files, n_examples)

    plt.figure(figsize=(12, 8))

    for i, file in enumerate(chosen):

        y, sr = librosa.load(file, sr=sr)

        plt.subplot(n_examples,1,i+1)

        librosa.display.waveshow(y, sr=sr)

        duration = len(y)/sr

        plt.title(f"{os.path.basename(file)}  |  duration={duration:.2f}s")

    plt.tight_layout()
    plt.show()



def collect_lengths(data_path, sr=16000):

    records = []

    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):
                path = os.path.join(root, file)

                y, _ = librosa.load(path, sr=sr)

                records.append({
                    "path": path,
                    "length": len(y),
                    "duration": len(y)/sr
                })

    return records

def sort_by_length_longest_ten(records):

    records_sorted = sorted(records, key=lambda x: x["length"], reverse=True)

    longest10 = records_sorted[:10]

    for r in longest10:
        print(r["duration"], r["path"])

    records_sorted = sorted(records, key=lambda x: x["length"], reverse=True)

    longest10 = records_sorted[:10]

    for r in longest10:
        print(r["duration"], r["path"])

    return longest10

def trim_and_save_dataset(ARCHIVE_PATH, TRIMMED_PATH, SR=16000):  #trim and normalize and save to new folder

    # Step 1: walk through folders and trim
    for root, dirs, files in os.walk(ARCHIVE_PATH):
        # compute corresponding folder in trimmed_archive
        relative_root = os.path.relpath(root, ARCHIVE_PATH)
        trimmed_root = os.path.join(TRIMMED_PATH, relative_root)
        os.makedirs(trimmed_root, exist_ok=True)

        for file in files:
            if file.endswith(".wav"):
                original_path = os.path.join(root, file)
                trimmed_path = os.path.join(trimmed_root, file)

                # load and trim
                y, sr = librosa.load(original_path, sr=SR)
                y_trim, _ = librosa.effects.trim(y, top_db=100)  # ← adjust top_db as needed

                # optional normalization  per file
                '''if max(abs(y_trim)) > 0:
                    y_trim = y_trim / max(abs(y_trim))'''

                # save trimmed file
                sf.write(trimmed_path, y_trim, SR)

    print("All files trimmed and saved to:", TRIMMED_PATH)

def normalize_and_save_dataset(TRIMMED_PATH, NORMALIZED_PATH, SR=16000):
    'normalize to global mean and std across the dataset and save to new folder'
    # Step 1: compute global mean and std
    all_samples = []
    for root, dirs, files in os.walk(TRIMMED_PATH):
        for file in files:
            if file.endswith(".wav"):
                path = os.path.join(root, file)
                y, sr = librosa.load(path, sr=SR)
                all_samples.append(y)
    all_samples = np.concatenate(all_samples)
    global_mean = np.mean(all_samples)
    global_std = np.std(all_samples)
    print(f"Global mean: {global_mean:.4f}, Global std: {global_std:.4f}")
    # Step 2: normalize each file and save
    for root, dirs, files in os.walk(TRIMMED_PATH):
        relative_root = os.path.relpath(root, TRIMMED_PATH)
        normalized_root = os.path.join(NORMALIZED_PATH, relative_root)
        os.makedirs(normalized_root, exist_ok=True)

        for file in files:
            if file.endswith(".wav"):
                original_path = os.path.join(root, file)
                normalized_path = os.path.join(normalized_root, file)

                y, sr = librosa.load(original_path, sr=SR)

                if global_std > 0:
                    y_norm = (y - global_mean) / global_std
                else:
                    y_norm = y - global_mean  # avoid division by zero

                sf.write(normalized_path, y_norm, SR)
    print("All files normalized and saved to:", NORMALIZED_PATH)




def fixed_length_and_save_dataset(TRIMMED_PATH, FIXED_PATH, TARGET_SEC=3.0, SR=16000):



    TARGET_LEN = int(TARGET_SEC * SR)

    total = 0

    for root, dirs, files in os.walk(TRIMMED_PATH):

        relative_root = os.path.relpath(root, TRIMMED_PATH)
        fixed_root = os.path.join(FIXED_PATH, relative_root)

        os.makedirs(fixed_root, exist_ok=True)

        for file in files:

            if file.endswith(".wav"):

                original_path = os.path.join(root, file)
                fixed_path = os.path.join(fixed_root, file)

                y, sr = librosa.load(original_path, sr=SR)

                if len(y) > TARGET_LEN:
                    start = (len(y) - TARGET_LEN) // 2
                    y = y[start:start + TARGET_LEN]   # ← fixed

                else:
                    pad = TARGET_LEN - len(y)
                    y = np.pad(y, (pad // 2, pad - pad // 2))

                sf.write(fixed_path, y, SR)

                total += 1

    print("Processed files:", total)



def visualize_before_after(ARCHIVE_PATH, TRIMMED_PATH, sr=16000): #random 10 files before and after trimming
    # Step 2: collect all original wav files
    all_files = []
    for root, _, files in os.walk(ARCHIVE_PATH):
        for file in files:
            if file.endswith(".wav"):
                all_files.append(os.path.join(root, file))

    # randomly pick 10
    sample_files = random.sample(all_files, 10)

    plt.figure(figsize=(12, 20))

    for i, orig_path in enumerate(sample_files):
        # corresponding trimmed file
        rel_path = os.path.relpath(orig_path, ARCHIVE_PATH)
        trimmed_path = os.path.join(TRIMMED_PATH, rel_path)

        y_orig, _ = librosa.load(orig_path, sr=sr)
        y_trim, _ = librosa.load(trimmed_path, sr=sr)

        # plot original
        plt.subplot(10, 2, i * 2 + 1)
        librosa.display.waveshow(y_orig, sr=sr)
        plt.title("Original: " + os.path.basename(orig_path))

        # plot trimmed
        plt.subplot(10, 2, i * 2 + 2)
        librosa.display.waveshow(y_trim, sr=sr)
        plt.title("Trimmed: " + os.path.basename(orig_path))

    plt.tight_layout()
    plt.show()

def visalize_before_after_longest10(records, ARCHIVE_PATH, TRIMMED_PATH, sr=16000):  #longest 10 files before and after trimming
    # we find the longest 10 files in the original dataset and visualize them before and after trimming in one figure
    longest10 = sort_by_length_longest_ten(records)
    plt.figure(figsize=(12, 20))
    for i, r in enumerate(longest10):
        orig_path = r["path"]
        rel_path = os.path.relpath(orig_path, ARCHIVE_PATH)
        trimmed_path = os.path.join(TRIMMED_PATH, rel_path)

        y_orig, _ = librosa.load(orig_path, sr=sr)
        y_trim, _ = librosa.load(trimmed_path, sr=sr)

        # plot original
        plt.subplot(10, 2, i * 2 + 1)
        librosa.display.waveshow(y_orig, sr=sr)
        plt.title(f"Original: {os.path.basename(orig_path)} | duration={len(y_orig)/sr:.2f}s")

        # plot trimmed
        plt.subplot(10, 2, i * 2 + 2)
        librosa.display.waveshow(y_trim, sr=sr)
        plt.title(f"Trimmed: {os.path.basename(orig_path)} | duration={len(y_trim)/sr:.2f}s")

    plt.tight_layout()
    plt.show()






def visualize_files(records, sr=16000):

    plt.figure(figsize=(12, 12))

    for i, r in enumerate(records):
        y, sr = librosa.load(r["path"], sr=sr)

        plt.subplot(len(records), 1, i + 1)

        librosa.display.waveshow(y, sr=sr)

        name = os.path.basename(r["path"])

        plt.title(f"{name} | duration={r['duration']:.2f}s")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    '''
    # load and analyze lengths
    files, lengths = analyze_lengths("./archive")
    '''

    '''
    # visualize_waveforms(files)
    '''
    '''
    # visualize one file 
    file = files[0]

    y, sr = librosa.load(file, sr=16000)

    # only before trimming
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Raw waveform")
    plt.show()

    # before and after trimming
    y_trim, _ = librosa.effects.trim(y, top_db=100)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title("Before trimming")

    plt.subplot(2, 1, 2)
    librosa.display.waveshow(y_trim, sr=sr)
    plt.title("After trimming")

    plt.show()
    '''

    '''
    # collect lengths for all files
    records = collect_lengths("./archive")
    '''

    '''
    # sort by length and print longest 10 and visualize them
    records_sorted = sorted(records, key=lambda x: x["length"], reverse=True)

    longest10 = records_sorted[:10]

    for r in longest10:
        print(r["duration"], r["path"])

    records_sorted = sorted(records, key=lambda x: x["length"], reverse=True)

    longest10 = records_sorted[:10]

    for r in longest10:
        print(r["duration"], r["path"])

    visualize_files(longest10)
    '''


    '''
    # analyze lengths after trimming
    trim_lengths = []

    for r in records:
        y, sr = librosa.load(r["path"], sr=16000)

        y_trim, _ = librosa.effects.trim(y, top_db=100)

        trim_lengths.append(len(y_trim))

    trim_lengths = np.array(trim_lengths)

    print("Min:", trim_lengths.min() / sr)
    print("Max:", trim_lengths.max() / sr)
    print("Mean:", trim_lengths.mean() / sr)
    print("Median:", np.median(trim_lengths) / sr)

    plt.figure(figsize=(8, 4))
    plt.hist(trim_lengths / sr, bins=40)
    plt.xlabel("Duration after trimming (sec)")
    plt.ylabel("Count")
    plt.title("Trimmed audio length distribution")
    plt.show()
    '''


    ARCHIVE_PATH = "./archive"
    TRIMMED_PATH = "./trimmed_archive"
    Fixed_path = "./fixed_archive"
    Normalized_path = "./normalized_archive"
    # New_path = "./processed_ravdess_16k_3s"
    records = collect_lengths("./archive")


    trim_and_save_dataset(ARCHIVE_PATH, TRIMMED_PATH)

    trimmed_files, trim_lengths = analyze_lengths("./trimmed_archive")

    # normalize_and_save_dataset(TRIMMED_PATH, Normalized_path)  #normalization per whole dataset (global mean and std)

    fixed_length_and_save_dataset(TRIMMED_PATH, Fixed_path, TARGET_SEC=3.0, SR=16000)

    visalize_before_after_longest10(records, ARCHIVE_PATH, Fixed_path)




    # visualize_before_after(ARCHIVE_PATH, New_path)


    '''
    trim_lengths = np.array(trim_lengths)  # samples

    for p in [90, 95, 98, 100]:
        print(p, np.percentile(trim_lengths, p) / 16000)
    '''

