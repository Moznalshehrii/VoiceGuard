import os
import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from tqdm import tqdm


# Step 1: Real vs Fake Visual Comparison

print("--- Step 1: Visualizing Real vs Fake Samples ---")

audio_files = glob.glob('VoiceGuard-main/data/**/*.flac', recursive=True) or glob.glob('data/**/*.flac', recursive=True)
total_detected = len(audio_files)
print(f"Total audio files detected in local repository: {total_detected}")

loaded_samples = []
for file_path in audio_files:
    try:
        data, sr = sf.read(file_path)
        loaded_samples.append((data, sr, file_path))
        if len(loaded_samples) == 2:
            break
    except Exception:
        continue

if len(loaded_samples) >= 2:
    y1, sr1, sample1_path = loaded_samples[0]
    y2, sr2, sample2_path = loaded_samples[1]
    
    S1 = librosa.feature.melspectrogram(y=y1, sr=sr1, n_mels=128)
    S1_db = librosa.power_to_db(S1, ref=np.max)
    
    S2 = librosa.feature.melspectrogram(y=y2, sr=sr2, n_mels=128)
    S2_db = librosa.power_to_db(S2, ref=np.max)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    librosa.display.waveshow(y1, sr=sr1, ax=axes[0, 0], color='blue')
    axes[0, 0].set_title(f'Sample 1 Waveform: {os.path.basename(sample1_path)}')
    
    img1 = librosa.display.specshow(S1_db, sr=sr1, x_axis='time', y_axis='mel', ax=axes[1, 0])
    axes[1, 0].set_title('Sample 1 Mel-Spectrogram')
    fig.colorbar(img1, ax=axes[1, 0], format='%+2.0f dB')
    
    librosa.display.waveshow(y2, sr=sr2, ax=axes[0, 1], color='red')
    axes[0, 1].set_title(f'Sample 2 Waveform: {os.path.basename(sample2_path)}')
    
    img2 = librosa.display.specshow(S2_db, sr=sr2, x_axis='time', y_axis='mel', ax=axes[1, 1])
    axes[1, 1].set_title('Sample 2 Mel-Spectrogram')
    fig.colorbar(img2, ax=axes[1, 1], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig('real_vs_fake_comparison.png')
    plt.show()
    print("Visual comparison plot generated successfully!")


# Step 2: Class Distribution Analysis

print("\n--- Step 2: Plotting Class Distribution ---")

categories = ['Spoof', 'Bonafide']
counts = [611827, 22100]

plt.figure(figsize=(6, 4))
plt.bar(categories, counts, color=['red', 'green'])
plt.title('Class Distribution (ASVspoof2021 DF Eval)')
plt.xlabel('Label')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()
print("Class distribution plot saved successfully!")


# Step 3: Audio Duration Distribution (All Detected Files)

print(f"\n--- Step 3: Analyzing Audio Durations ({total_detected} Files) ---")

durations = []
for f in tqdm(audio_files, desc="Calculating Durations"):
    try:
        durations.append(librosa.get_duration(path=f))
    except Exception:
        continue

if durations:
    # Calculate Statistical Metrics
    mean_duration = np.mean(durations)
    min_duration = np.min(durations)
    max_duration = np.max(durations)
    std_duration = np.std(durations)

    print("\n--- Audio Duration Statistics ---")
    print(f"Total Files Analyzed: {len(durations)}")
    print(f"Mean Duration: {mean_duration:.2f} seconds")
    print(f"Minimum Duration: {min_duration:.2f} seconds")
    print(f"Maximum Duration: {max_duration:.2f} seconds")
    print(f"Standard Deviation: {std_duration:.2f} seconds")

    # Plot Histogram with Mean Line
    plt.figure(figsize=(7, 4))
    plt.hist(durations, bins=15, color='purple', edgecolor='black', alpha=0.7)
    plt.axvline(mean_duration, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_duration:.2f}s')
    
    plt.title(f'Audio Duration Distribution (All {len(durations)} files)')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('File Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig('audio_durations.png')
    plt.show()

print("\nFull EDA Process Completed Successfully!")