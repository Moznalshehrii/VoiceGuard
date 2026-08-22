"""Step C training loop: fine-tune Wav2Vec2/XLSR + lightweight head.

Usage (smoke test on synthetic data):
    python3 -m src.train_wav2vec --protocol data/dummy/dummy_protocol.txt --audio_dir data/dummy/audio --epochs 3

Usage (real ASVspoof2019 LA):
    python3 -m src.train_wav2vec \
        --protocol data/raw/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
        --audio_dir data/raw/LA/ASVspoof2019_LA_train/flac \
        --epochs 10 --batch_size 4

Note: batch_size is much smaller than the baseline CNN's -- Wav2Vec2/XLSR is
~300M params, so batches need to be small (and grad_accum_steps used) to fit
in memory, especially on a laptop GPU.
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import RawWaveformSpoofDataset
from src.data.protocol import parse_asvspoof_protocol, summarize
from src.models.wav2vec_classifier import Wav2VecSpoofClassifier
from src.utils.metrics import compute_eer


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, device, criterion, optimizer=None, grad_accum_steps=1):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss, all_labels, all_scores, correct = 0.0, [], [], 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        if train:
            optimizer.zero_grad()
        for step, (waveforms, labels) in enumerate(tqdm(loader, leave=False)):
            waveforms, labels = waveforms.to(device), labels.to(device)

            logits = model(waveforms)
            loss = criterion(logits, labels)

            if train:
                (loss / grad_accum_steps).backward()
                if (step + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item() * waveforms.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]
            correct += (logits.argmax(1) == labels).sum().item()
            all_labels.extend(labels.cpu().tolist())
            all_scores.extend(probs.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    eer, _ = compute_eer(all_labels, all_scores)
    return avg_loss, acc, eer


def main(args):
    device = get_device()
    print(f"Using device: {device}")

    df = parse_asvspoof_protocol(args.protocol, audio_dir=args.audio_dir, audio_ext=args.audio_ext)
    print(f"Loaded protocol: {summarize(df)}")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_val = max(1, int(len(df) * args.val_frac))
    val_df, train_df = df.iloc[:n_val], df.iloc[n_val:]
    print(f"Train: {len(train_df)} clips | Val: {len(val_df)} clips")

    train_ds = RawWaveformSpoofDataset(train_df, train=True)
    val_ds = RawWaveformSpoofDataset(val_df, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Loading pretrained backbone: {args.checkpoint} (downloads on first run)")
    model = Wav2VecSpoofClassifier(checkpoint=args.checkpoint, freeze_feature_extractor=True).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_trainable:,} trainable / {n_total:,} total")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_eer = float("inf")
    for epoch in range(1, args.epochs + 1):
        if epoch == args.unfreeze_at_epoch:
            print("Unfreezing transformer backbone for full fine-tuning")
            model.unfreeze_transformer()
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr * 0.1
            )

        train_loss, train_acc, train_eer = run_epoch(
            model, train_loader, device, criterion, optimizer, args.grad_accum_steps
        )
        val_loss, val_acc, val_eer = run_epoch(model, val_loader, device, criterion, optimizer=None)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} EER {train_eer:.2f}% | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} EER {val_eer:.2f}%"
        )

        if val_eer < best_eer:
            best_eer = val_eer
            torch.save(model.state_dict(), args.ckpt_path)
            print(f"  -> saved new best checkpoint (val EER {val_eer:.2f}%) to {args.ckpt_path}")

    print(f"Done. Best val EER: {best_eer:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--audio_ext", default=".flac")
    parser.add_argument("--checkpoint", default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--unfreeze_at_epoch", type=int, default=3)
    parser.add_argument("--ckpt_path", default="checkpoints/wav2vec_spoof.pt")
    main(parser.parse_args())
