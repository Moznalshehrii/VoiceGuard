"""Baseline CNN: validates the training pipeline end-to-end.

Not meant to be strong -- the brief is explicit that "expected accuracy is
modest" here. The real performance work happens later when this is swapped
for a fine-tuned Wav2Vec2/XLSR backbone (step C).
"""

import torch.nn as nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BaselineCNN(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 16),
            conv_block(16, 32),
            conv_block(32, 64),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # collapses (freq, time) -> 1x1, so
        self.classifier = nn.Linear(64, n_classes)  # input length can vary

    def forward(self, x):
        # x: [B, 1, n_mels, T]
        x = self.features(x)
        x = self.pool(x).flatten(1)  # [B, 64]
        return self.classifier(x)  # [B, n_classes] logits
