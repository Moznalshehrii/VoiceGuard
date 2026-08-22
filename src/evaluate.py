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

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import RawWaveformSpoofDataset
from src.data.protocol import parse_asvspoof_protocol, parse_in_the_wild_meta, summarize
from src.models.wav2vec_classifier import Wav2VecSpoofClassifier
from src.utils.metrics import compute_eer


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model, loader, device) -> tuple[float, float]:
    """Returns (accuracy, eer_percent). No training here -- eval only."""
    # TODO (Person 2): mirror the eval half of run_epoch() in train_wav2vec.py
    # -- forward pass only (torch.no_grad()), collect (label, P(bonafide))
    # per clip, then call compute_eer(all_labels, all_scores).
    raise NotImplementedError


def main(args):
    device = get_device()
    print(f"Using device: {device}")

    # TODO (Person 2): branch on args.is_itw to call the right parser
    if args.is_itw:
        df = parse_in_the_wild_meta(args.protocol, audio_dir=args.audio_dir)
    else:
        df = parse_asvspoof_protocol(args.protocol, audio_dir=args.audio_dir, audio_ext=args.audio_ext)
    print(f"Loaded protocol: {summarize(df)}")

    dataset = RawWaveformSpoofDataset(df, train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # TODO (Person 2): load the model architecture, then load the trained
    # weights from args.checkpoint with model.load_state_dict(...)
    model = Wav2VecSpoofClassifier().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    acc, eer = evaluate(model, loader, device)
    print(f"Accuracy: {acc:.3f} | EER: {eer:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--audio_ext", default=".flac")
    parser.add_argument("--is_itw", action="store_true", help="set this flag when evaluating on In-the-Wild")
    parser.add_argument("--batch_size", type=int, default=8)
    main(parser.parse_args())
