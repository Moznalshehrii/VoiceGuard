"""Step C: the main model. Swaps the baseline CNN's hand-built spectrogram
features for a pretrained Wav2Vec2/XLSR backbone -- self-supervised speech
representations trained on huge amounts of raw audio, which is what should
give the "clear jump in performance" the brief asks for.

XLS-R (facebook/wav2vec2-xls-r-300m) is used specifically because it was
pretrained across ~128 languages, which matters here: it needs to generalize
to synthesis artifacts, not just one language's phonetics.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

DEFAULT_CHECKPOINT = "facebook/wav2vec2-xls-r-300m"


class Wav2VecSpoofClassifier(nn.Module):
    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        n_classes: int = 2,
        freeze_feature_extractor: bool = True,
        freeze_transformer: bool = False,
    ):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(checkpoint)
        hidden_size = self.backbone.config.hidden_size

        if freeze_feature_extractor:
            # freezes only the low-level conv feature extractor (generic
            # acoustic features); the transformer layers stay trainable so
            # the model can still adapt to spoof-specific cues.
            self.backbone.feature_extractor._freeze_parameters()

        if freeze_transformer:
            for param in self.backbone.encoder.parameters():
                param.requires_grad = False

        # lightweight head, per the brief -- the backbone does the heavy lifting
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def unfreeze_transformer(self):
        """Call after a few linear-probe epochs to start full fine-tuning."""
        for param in self.backbone.encoder.parameters():
            param.requires_grad = True

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        # waveforms: [B, num_samples]
        outputs = self.backbone(waveforms)
        hidden = outputs.last_hidden_state  # [B, T', hidden_size]
        pooled = hidden.mean(dim=1)  # mean-pool over time -> [B, hidden_size]
        return self.classifier(pooled)  # [B, n_classes] logits
