"""Person 2 & 3 -- Step D: generic evaluation script.

Loads a trained checkpoint (from step C) and runs it on ANY dataset that's
been normalized through protocol.py (parse_asvspoof_protocol or
parse_in_the_wild_meta), reporting EER.

Person 2: fill in the TODOs below, then run this on ASVspoof 2021 DF.
Person 3: once Person 2's version works, run it on In-the-Wild (no code
changes needed -- just different --protocol/--audio_dir/--is_itw args) and
write up the ASVspoof-vs-In-the-Wild EER comparison (the project's key
result). Put that writeup in a new file, e.g. docs/generalization_results.md,
not in this script.

Usage (once implemented):
    python3 -m src.evaluate \
        --checkpoint checkpoints/wav2vec_spoof.pt \
        --protocol data/raw/ASVspoof2021_DF/keys/CM/trial_metadata.txt \
        --audio_dir data/raw/ASVspoof2021_DF/flac \
        --audio_ext .flac

    python3 -m src.evaluate \
        --checkpoint checkpoints/wav2vec_spoof.pt \
        --protocol data/raw/release_in_the_wild/meta.csv \
        --audio_dir data/raw/release_in_the_wild \
        --is_itw
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from src.data.dataset import RawWaveformSpoofDataset
from src.data.protocol import parse_asvspoof_protocol, parse_in_the_wild_meta, summarize
from src.models.wav2vec_classifier import Wav2VecSpoofClassifier
from src.utils.metrics import compute_eer, compute_min_dcf


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model, loader, device) -> tuple[float, float]:
    """Return (EER percent, normalized min-DCF) for bonafide probability."""
    all_labels, all_scores = [], []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            if batch is None:
                continue
            waveforms, labels = batch
            logits = model(waveforms.to(device))
            # Training uses class 1 for bonafide, so its softmax probability
            # is the score expected by both metric functions.
            scores = torch.softmax(logits, dim=1)[:, 1]
            all_labels.extend(labels.tolist())
            all_scores.extend(scores.cpu().tolist())

    if len(set(all_labels)) != 2:
        raise ValueError("Evaluation requires both bonafide and spoof trials.")

    print(f"Scored {len(all_labels):,} readable trials.")
    eer, _ = compute_eer(all_labels, all_scores)
    min_dcf, _ = compute_min_dcf(all_labels, all_scores)
    return eer, min_dcf


def keep_available_audio(df, audio_dir: str, audio_ext: str):
    """Keep protocol rows with audio in a (possibly partial) flat audio set."""
    audio_path = Path(audio_dir)
    available = {path.stem for path in audio_path.glob(f"*{audio_ext}")}
    if not available:
        raise FileNotFoundError(f"No {audio_ext} files found in {audio_path}")

    keep = df["filepath"].map(lambda path: Path(path).stem in available)
    filtered = df.loc[keep].reset_index(drop=True)
    if filtered.empty:
        raise FileNotFoundError(
            "None of the protocol trial IDs match the audio directory. "
            "Check --protocol, --audio_dir, and --audio_ext."
        )
    if len(filtered) != len(df):
        print(f"Using {len(filtered):,} of {len(df):,} protocol trials with available audio.")
    return filtered


def load_weights(model, checkpoint: str, device: torch.device) -> None:
    """Load a plain state_dict or a checkpoint dict containing ``state_dict``."""
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must be a model state_dict or contain a 'state_dict' key.")
    # Accommodate checkpoints saved through DataParallel.
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)


def collate_skip_decode_errors(batch):
    """Discard dataset items that could not be decoded during evaluation."""
    batch = [item for item in batch if item is not None]
    return default_collate(batch) if batch else None


def main(args):
    device = get_device()
    print(f"Using device: {device}")

    if args.is_itw:
        df = parse_in_the_wild_meta(args.protocol, audio_dir=args.audio_dir)
    else:
        df = parse_asvspoof_protocol(args.protocol, audio_dir=args.audio_dir, audio_ext=args.audio_ext)
    df = keep_available_audio(df, args.audio_dir, args.audio_ext)
    print(f"Loaded protocol: {summarize(df)}")

    dataset = RawWaveformSpoofDataset(df, train=False, skip_decode_errors=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_skip_decode_errors,
    )

    model = Wav2VecSpoofClassifier(checkpoint=args.backbone).to(device)
    load_weights(model, args.checkpoint, device)

    eer, min_dcf = evaluate(model, loader, device)
    print(f"EER: {eer:.2f}% | min-DCF: {min_dcf:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--audio_ext", default=".flac")
    parser.add_argument("--is_itw", action="store_true", help="set this flag when evaluating on In-the-Wild")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--backbone", default="facebook/wav2vec2-xls-r-300m")
    main(parser.parse_args())
