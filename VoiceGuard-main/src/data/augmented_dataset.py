"""Person 5 -- Step E: robustness via data augmentation.

Separate file (not editing dataset.py directly) so there's zero merge
conflict risk with anyone else's work. Wraps RawWaveformSpoofDataset and
adds noise / codec-compression augmentation on top, per the brief:
"Apply data augmentation -- background noise and codec/compression effects
-- to narrow the generalization gap, comparing EER before and after."
"""

import os
import sys
import random
import torch
import torchaudio.transforms as T

# Add project root directory to system path for seamless module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.dataset import RawWaveformSpoofDataset, load_and_fix_length


def add_background_noise(waveform: torch.Tensor, snr_db: float = 15.0) -> torch.Tensor:
    """Mix in Gaussian noise at a target signal-to-noise ratio (dB)."""
    noise = torch.randn_like(waveform)
    
    s_power = waveform.norm(p=2)
    n_power = noise.norm(p=2)
    
    snr = 10 ** (snr_db / 20)
    scale = s_power / (snr * n_power + 1e-8)
    
    return waveform + scale * noise


def apply_codec_compression(waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """Simulate lossy codec compression using Mu-law encoding/decoding."""
    encoder = T.MuLawEncoding(quantization_channels=256)
    decoder = T.MuLawDecoding(quantization_channels=256)
    
    encoded = encoder(waveform)
    decoded = decoder(encoded)
    return decoded


class AugmentedRawWaveformSpoofDataset(RawWaveformSpoofDataset):
    def __init__(self, *args, augment_prob: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment_prob = augment_prob

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        waveform = load_and_fix_length(row["filepath"], self.sample_rate, self.num_samples, self.train)

        # Apply audio augmentations only during training phase based on probability
        if self.train and random.random() < self.augment_prob:
            aug_type = random.choice(["noise", "codec", "both"])
            
            if aug_type == "noise":
                waveform = add_background_noise(waveform, snr_db=15.0)
            elif aug_type == "codec":
                waveform = apply_codec_compression(waveform, self.sample_rate)
            elif aug_type == "both":
                waveform = add_background_noise(waveform, snr_db=15.0)
                waveform = apply_codec_compression(waveform, self.sample_rate)

        waveform = waveform.squeeze(0)
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)

        label = torch.tensor(row["label"], dtype=torch.long)
        return waveform, label


if __name__ == "__main__":
    print("AugmentedRawWaveformSpoofDataset defined successfully.")