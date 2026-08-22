"""Audio -> Mel-spectrogram pipeline and PyTorch Dataset for real/spoof classification."""

import random

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


def load_and_fix_length(filepath: str, sample_rate: int, num_samples: int, train: bool) -> torch.Tensor:
    """Load an audio file, resample to `sample_rate`, force mono, and pad/crop to `num_samples`."""
    waveform, sr = torchaudio.load(filepath)

    if waveform.shape[0] > 1:  # stereo -> mono
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    length = waveform.shape[1]
    if length < num_samples:
        # repeat-pad: tile the clip until it's long enough, then trim
        reps = num_samples // length + 1
        waveform = waveform.repeat(1, reps)
        length = waveform.shape[1]

    if length > num_samples:
        if train:
            start = random.randint(0, length - num_samples)  # random crop = cheap augmentation
        else:
            start = (length - num_samples) // 2  # center crop = deterministic eval
        waveform = waveform[:, start : start + num_samples]

    return waveform


class AudioSpoofDataset(Dataset):
    """Wraps a normalized protocol DataFrame (see protocol.py) and yields
    (mel_spectrogram [1, n_mels, T], label) pairs.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sample_rate: int = 16000,
        duration_s: float = 4.0,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration_s)
        self.train = train

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        waveform = load_and_fix_length(row["filepath"], self.sample_rate, self.num_samples, self.train)

        spec = self.mel_spec(waveform)  # [1, n_mels, T]
        spec = self.to_db(spec)

        # per-utterance normalization
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)

        label = torch.tensor(row["label"], dtype=torch.long)
        return spec, label


class RawWaveformSpoofDataset(Dataset):
    """Like AudioSpoofDataset, but yields raw waveforms instead of spectrograms.

    Wav2Vec2/XLSR's own convolutional feature extractor operates directly on
    the raw waveform and learns its own time-frequency representation, so we
    skip the Mel-spectrogram step entirely here (step C uses this; step B's
    CNN keeps using AudioSpoofDataset above).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sample_rate: int = 16000,
        duration_s: float = 4.0,
        train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration_s)
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        waveform = load_and_fix_length(row["filepath"], self.sample_rate, self.num_samples, self.train)
        waveform = waveform.squeeze(0)  # [num_samples] -- Wav2Vec2 expects 1D per-sample input

        # zero-mean, unit-variance -- matches what Wav2Vec2FeatureExtractor
        # does internally (do_normalize=True), kept explicit here so this
        # dataset has no hidden dependency on transformers' preprocessing.
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)

        label = torch.tensor(row["label"], dtype=torch.long)
        return waveform, label
