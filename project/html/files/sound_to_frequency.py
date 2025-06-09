import pandas as pd
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

audio_path = "ag_dog_2.wav"
y, sr = librosa.load(audio_path, sr=16000, mono=True)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.figure(figsize=(10, 4))
librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", cmap="coolwarm")
plt.colorbar(label="Decibels")
plt.title("Aggressive Spectrogram of Puppy Sound")
plt.show()

audio_path = "df_dog_15.wav"
y, sr = librosa.load(audio_path, sr=16000, mono=True)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.figure(figsize=(10, 4))
librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", cmap="coolwarm")
plt.colorbar(label="Decibels")
plt.title("Defensive Spectrogram of Puppy Sound")
plt.show()

audio_path = "no_dog_1.wav"
y, sr = librosa.load(audio_path, sr=16000, mono=True)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.figure(figsize=(10, 4))
librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", cmap="coolwarm")
plt.colorbar(label="Decibels")
plt.title("Normal Spectrogram of Puppy Sound")

#comment: Extracting features from the audio file
"""Extracting features from the audio file
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform of Puppy Sound")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# Compute MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis="time", cmap="coolwarm")
plt.colorbar(label="MFCC Coefficients")
plt.title("MFCCs of Puppy Sound")
plt.show()

#Spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.figure(figsize=(10, 4))
librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", cmap="coolwarm")
plt.colorbar(label="Decibels")
plt.title("Spectrogram of Puppy Sound")
plt.show()

audio_path = "cat_0.wav"
y, sr = librosa.load(audio_path, sr=16000, mono=True)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.figure(figsize=(10, 4))
librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", cmap="coolwarm")
plt.colorbar(label="Decibels")
plt.title("Spectrogram of Cat Sound")
plt.show()
"""

base_dir = "dog_bark_train"
all_data = []

# Mapping short prefixes to full emotion labels
label_map = {
    'ag': 'aggressive',
    'df': 'defensive',
    'no': 'normal'
}

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".wav"):
            filepath = os.path.join(root, file)
            try:
                # Extract label from filename prefix
                prefix = file.split("_")[0]
                label = label_map.get(prefix, 'unknown')  # fallback if typo

                y, sr = librosa.load(filepath, sr=16000, mono=True)
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                mean_spectrum = np.mean(D, axis=1)  # shape: (freq_bins,)

                feature_row = {
                    'filepath': filepath,
                    'label': label,
                    **{f'spec_{i}': mean_spectrum[i] for i in range(len(mean_spectrum))}
                }
                all_data.append(feature_row)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

# Create DataFrame and save
df = pd.DataFrame(all_data)
df.to_csv("all_spectrum_features.csv", index=False)
print("Saved labeled spectrum features to all_spectrum_features.csv")

base_dir = "dog_bark_test"
all_data = []

# Mapping short prefixes to full emotion labels
label_map = {
    'ag': 'aggressive',
    'df': 'defensive',
    'no': 'normal'
}

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".wav"):
            filepath = os.path.join(root, file)
            try:
                # Extract label from filename prefix
                prefix = file.split("_")[0]
                label = label_map.get(prefix, 'unknown')  # fallback if typo

                y, sr = librosa.load(filepath, sr=16000, mono=True)
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                mean_spectrum = np.mean(D, axis=1)  # shape: (freq_bins,)

                feature_row = {
                    'filepath': filepath,
                    'label': label,
                    **{f'spec_{i}': mean_spectrum[i] for i in range(len(mean_spectrum))}
                }
                all_data.append(feature_row)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

# Create DataFrame and save
df = pd.DataFrame(all_data)
#df.to_csv("all_spectrum_features_test.csv", index=False)
#print("Saved labeled spectrum features to all_spectrum_features.csv")

import librosa
import librosa.display
import numpy as np

# Assume you have `mean_spectrum` (shape: [n_freq])
# We'll fake a spectrogram by repeating it over time
S = np.tile(mean_spectrum.reshape(-1, 1), 100)  # 100 frames
S = librosa.db_to_amplitude(S)  # Convert back to linear scale

# Griffin-Lim to approximate audio
y = librosa.griffinlim(S, n_iter=60)

# Save or play
import soundfile as sf
sf.write("mean_emotion_sound.wav", y, sr=14400)
