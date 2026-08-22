"""Person 5 -- Step E: robustness via data augmentation.

Separate file (not editing dataset.py directly) so there's zero merge
conflict risk with anyone else's work. Wraps RawWaveformSpoofDataset and
adds noise / codec-compression augmentation on top, per the brief:
"Apply data augmentation -- background noise and codec/compression effects
-- to narrow the generalization gap, comparing EER before and after."

Plan:
1. Implement add_background_noise() and apply_codec_compression() below.
2. Implement AugmentedRawWaveformSpoofDataset, which calls the baseline
   RawWaveformSpoofDataset's loading logic then augments the waveform
   before returning it (train split only -- eval should stay clean).
3. Train the SAME model (Wav2VecSpoofClassifier / BaselineCNN) once with
   this dataset and once without, evaluate both on the same eval set, and
   compare EER. That before/after comparison is the deliverable.
"""

import torch

from src.data.dataset import RawWaveformSpoofDataset, load_and_fix_length


def add_background_noise(waveform: torch.Tensor, snr_db: float = 15.0) -> torch.Tensor:
    """Mix in Gaussian noise at a target signal-to-noise ratio (dB).

    TODO (Person 5): compute signal power, derive noise power from snr_db,
    generate torch.randn_like(waveform) scaled to that power, and add it.
    (Using real background-noise clips, e.g. from MUSAN, instead of
    synthetic Gaussian noise would be a stronger version of this -- worth
    trying if there's time.)
    """
    raise NotImplementedError


def apply_codec_compression(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Simulate lossy codec compression (the effect ASVspoof2021 DF has).

    TODO (Person 5): torchaudio.functional / torchaudio.io has codec
    application utilities (e.g. encoding to a low-bitrate format and
    decoding back) -- look at torchaudio's AudioEffector or apply_codec.
    """
    raise NotImplementedError


class AugmentedRawWaveformSpoofDataset(RawWaveformSpoofDataset):
    def __init__(self, *args, augment_prob: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment_prob = augment_prob

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        waveform = load_and_fix_length(row["filepath"], self.sample_rate, self.num_samples, self.train)

        # TODO (Person 5): only augment during training (self.train), and
        # only apply with self.augment_prob probability so the model also
        # sees some clean clips. Randomly pick noise, codec, both, or
        # neither each time.

        waveform = waveform.squeeze(0)
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)

        label = torch.tensor(row["label"], dtype=torch.long)
        return waveform, label
