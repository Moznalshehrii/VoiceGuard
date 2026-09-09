"""Person 4 -- extra generalization study: hold out some ASVspoof2019 LA
attack systems entirely from training, then test on them.

The training protocol file has a system_id per spoof clip (A01-A19 -- which
TTS/voice-conversion system produced it). Train on a subset (e.g. A01-A15),
then evaluate on the clips whose system_id was held out (A16-A19). The EER
on held-out systems vs. seen systems is the interesting number here -- it's
a within-dataset version of the ASVspoof-vs-In-the-Wild question Person 3
is answering, and isolates "new synthesis method" from "totally different
recording conditions".

Usage (once implemented):
    python3 -m src.train_holdout \
        --protocol data/raw/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
        --audio_dir data/raw/LA/ASVspoof2019_LA_train/flac \
        --holdout_systems A16 A17 A18 A19
"""
import argparse
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve
from src.utils.metrics import compute_eer
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import soundfile as sf
import torchaudio
import src.data.dataset as dataset_module

from src.data.dataset import RawWaveformSpoofDataset
from src.data.protocol import parse_asvspoof_protocol, summarize
from src.models.wav2vec_classifier import Wav2VecSpoofClassifier
from src.train_wav2vec import get_device, run_epoch

def safe_load_and_fix_length(filepath, sample_rate, num_samples, train):
    audio, sr = sf.read(filepath, dtype="float32")
    waveform = torch.from_numpy(audio)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    length = waveform.shape[1]

    if length < num_samples:
        reps = num_samples // length + 1
        waveform = waveform.repeat(1, reps)
        length = waveform.shape[1]

    if length > num_samples:
        if train:
            import random
            start = random.randint(0, length - num_samples)
        else:
            start = (length - num_samples) // 2

        waveform = waveform[:, start:start + num_samples]

    return waveform


dataset_module.load_and_fix_length = safe_load_and_fix_length
def build_seen_and_heldout(train_df, eval_df, holdout_systems):
    holdout_systems = set(holdout_systems)

    train_bonafide = train_df[train_df["system_id"] == "-"]
    eval_bonafide = eval_df[eval_df["system_id"] == "-"]

    train_spoof = train_df[train_df["system_id"] != "-"]
    eval_spoof = eval_df[eval_df["system_id"] != "-"]

    seen_eval_spoof = eval_spoof[~eval_spoof["system_id"].isin(holdout_systems)]
    heldout_spoof = eval_spoof[eval_spoof["system_id"].isin(holdout_systems)]

    seen_df = pd.concat(
        [train_bonafide, train_spoof, eval_bonafide, seen_eval_spoof]
    ).reset_index(drop=True)

    heldout_df = pd.concat(
        [eval_bonafide, heldout_spoof]
    ).reset_index(drop=True)

    return seen_df, heldout_df
def evaluate_with_scores(model, loader, device, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for waveforms, labels in tqdm(loader, leave=False):
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            logits = model(waveforms)
            loss = criterion(logits, labels)

            total_loss += loss.item() * waveforms.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1]

            predictions = logits.argmax(dim=1)
            correct += torch.eq(predictions, labels).sum().item()

            all_labels.extend(labels.cpu().tolist())
            all_scores.extend(probs.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)

    eer, _ = compute_eer(all_labels, all_scores)

    return avg_loss, accuracy, eer, all_labels, all_scores


def compute_min_dcf(
    labels,
    scores,
    p_target=0.05,
    c_miss=1.0,
    c_fa=1.0
):
    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
        pos_label=1
    )

    p_miss = 1 - tpr
    p_fa = fpr

    dcf = (
        c_miss * p_target * p_miss
        + c_fa * (1 - p_target) * p_fa
    )

    normalization = min(
        c_miss * p_target,
        c_fa * (1 - p_target)
    )

    normalized_dcf = dcf / normalization

    idx = np.argmin(normalized_dcf)

    return float(normalized_dcf[idx]), float(thresholds[idx])
def evaluate_only(args):
    device = get_device()
    print(f"Using device: {device}")

    train_df = parse_asvspoof_protocol(
        args.train_protocol,
        audio_dir=args.train_audio_dir,
        audio_ext=args.audio_ext
    )

    eval_df = parse_asvspoof_protocol(
        args.eval_protocol,
        audio_dir=args.eval_audio_dir,
        audio_ext=args.audio_ext
    )

    seen_df, heldout_df = build_seen_and_heldout(
        train_df,
        eval_df,
        args.holdout_systems
    )

    seen_ds = RawWaveformSpoofDataset(seen_df, train=False)
    heldout_ds = RawWaveformSpoofDataset(heldout_df, train=False)

    seen_loader = DataLoader(
        seen_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    heldout_loader = DataLoader(
        heldout_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = Wav2VecSpoofClassifier(
        checkpoint=args.checkpoint,
        freeze_feature_extractor=True
    ).to(device)

    model.load_state_dict(
        torch.load(args.ckpt_path, map_location=device)
    )

    model.eval()
    criterion = nn.CrossEntropyLoss()

    print("\nEvaluating seen systems...")

    _, seen_acc, seen_eer, seen_labels, seen_scores = evaluate_with_scores(
        model,
        seen_loader,
        device,
        criterion
    )

    seen_mindcf, seen_dcf_threshold = compute_min_dcf(
        seen_labels,
        seen_scores
    )

    print(f"\nSeen EER: {seen_eer:.2f}%")
    print(f"Seen minDCF: {seen_mindcf:.4f}")

    print("\nEvaluating held-out systems...")

    _, heldout_acc, heldout_eer, heldout_labels, heldout_scores = evaluate_with_scores(
        model,
        heldout_loader,
        device,
        criterion
    )

    heldout_mindcf, heldout_dcf_threshold = compute_min_dcf(
        heldout_labels,
        heldout_scores
    )

    print("\n===== GENERALIZATION RESULTS =====")
    print(f"Seen systems EER:        {seen_eer:.2f}%")
    print(f"Seen systems minDCF:     {seen_mindcf:.4f}")
    print(f"Held-out systems EER:    {heldout_eer:.2f}%")
    print(f"Held-out systems minDCF: {heldout_mindcf:.4f}")
    print(f"Seen accuracy:           {seen_acc:.3f}")
    print(f"Held-out accuracy:       {heldout_acc:.3f}")
    print(
        f"EER generalization gap: "
        f"{heldout_eer - seen_eer:.2f} percentage points"
    )

    os.makedirs("results", exist_ok=True)

    np.savez(
        "results/holdout_generalization_scores.npz",
        seen_labels=np.array(seen_labels),
        seen_scores=np.array(seen_scores),
        heldout_labels=np.array(heldout_labels),
        heldout_scores=np.array(heldout_scores)
    )

    print(
        "\nSaved evaluation scores to "
        "results/holdout_generalization_scores.npz"
    )
def main(args):
    train_df = parse_asvspoof_protocol(
        args.train_protocol,
        audio_dir=args.train_audio_dir,
        audio_ext=args.audio_ext
    )

    eval_df = parse_asvspoof_protocol(
        args.eval_protocol,
        audio_dir=args.eval_audio_dir,
        audio_ext=args.audio_ext
    )

    print(f"Train protocol: {summarize(train_df)}")
    print(f"Eval protocol: {summarize(eval_df)}")

    seen_df, heldout_df = build_seen_and_heldout(
        train_df,
        eval_df,
        args.holdout_systems
    )

    print(f"Seen systems: {summarize(seen_df)}")
    print(f"Held-out systems: {summarize(heldout_df)}")

    device = get_device()
    print(f"Using device: {device}")

    train_ds = RawWaveformSpoofDataset(seen_df, train=True)
    seen_eval_ds = RawWaveformSpoofDataset(seen_df, train=False)
    heldout_ds = RawWaveformSpoofDataset(heldout_df, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    seen_loader = DataLoader(
        seen_eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    heldout_loader = DataLoader(
        heldout_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = Wav2VecSpoofClassifier(
        checkpoint=args.checkpoint,
        freeze_feature_extractor=True
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_eer = run_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            args.grad_accum_steps
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Loss {train_loss:.4f} | "
            f"Accuracy {train_acc:.3f} | "
            f"EER {train_eer:.2f}%"
        )

    print("\nEvaluating generalization...")

    _, seen_acc, seen_eer = run_epoch(
        model,
        seen_loader,
        device,
        criterion,
        optimizer=None
    )

    _, heldout_acc, heldout_eer = run_epoch(
        model,
        heldout_loader,
        device,
        criterion,
        optimizer=None
    )

    print("\n===== GENERALIZATION RESULTS =====")
    print(f"Seen systems EER:     {seen_eer:.2f}%")
    print(f"Held-out systems EER: {heldout_eer:.2f}%")
    print(f"Seen accuracy:        {seen_acc:.3f}")
    print(f"Held-out accuracy:    {heldout_acc:.3f}")
    print(
        f"EER generalization gap: "
        f"{heldout_eer - seen_eer:.2f} percentage points"
    )

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), args.ckpt_path)

    print(f"\nSaved model to {args.ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_protocol", required=True)
    parser.add_argument("--train_audio_dir", required=True)

    parser.add_argument("--eval_pr        jjjjhjhyytyyytyyyotocol", required=True)
    parser.add_argument("--eval_audio_dir", required=True)

    parser.add_argument("--audio_ext", default=".flac")

    parser.add_argument(
        "--holdout_systems",
        nargs="+",
        default=["A16", "A17", "A18", "A19"]
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint", default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ckpt_path", default="checkpoints/wav2vec_holdout.pt")
    parser.add_argument("--eval_only", action="store_true")
    cli_args = parser.parse_args()

    if cli_args.eval_only:
        evaluate_only(cli_args)
    else:
        main(cli_args)