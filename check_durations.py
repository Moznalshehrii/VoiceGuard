import os
import soundfile as sf
import statistics

folder = "data/raw/LA/ASVspoof2019_LA_train/flac"

durations = []

for file in os.listdir(folder):
    if file.endswith(".flac"):
        path = os.path.join(folder, file)
        audio, sr = sf.read(path)
        duration = len(audio) / sr
        durations.append(duration)

print("Files:", len(durations))
print("Average:", round(statistics.mean(durations), 2), "seconds")
print("Median:", round(statistics.median(durations), 2), "seconds")
print("Min:", round(min(durations), 2), "seconds")
print("Max:", round(max(durations), 2), "seconds")