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

from src.data.protocol import parse_asvspoof_protocol, summarize


def split_by_system(df, holdout_systems: list[str]):
    """Split a protocol DataFrame into (train_df, heldout_df).

    train_df: all bonafide clips + spoof clips whose system_id is NOT in
              holdout_systems
    heldout_df: bonafide clips (reused for a fair EER calc) + spoof clips
                whose system_id IS in holdout_systems

    TODO (Person 4): implement this. Hint -- df["system_id"] already exists
    thanks to protocol.py; bonafide rows have system_id == "-" and should
    end up in BOTH splits (EER needs real bonafide/spoof pairs on both
    sides, not just held-out spoof clips with no bonafide to compare against).
    """
    raise NotImplementedError


def main(args):
    df = parse_asvspoof_protocol(args.protocol, audio_dir=args.audio_dir, audio_ext=args.audio_ext)
    print(f"Full set: {summarize(df)}")

    train_df, heldout_df = split_by_system(df, args.holdout_systems)
    print(f"Train (seen systems): {summarize(train_df)}")
    print(f"Held-out (unseen systems): {summarize(heldout_df)}")

    # TODO (Person 4): reuse the training loop from src/train_wav2vec.py
    # (or import run_epoch from it) to train on train_df, then evaluate on
    # heldout_df using the same compute_eer approach as src/evaluate.py.
    # Report both EERs side by side -- that comparison IS the deliverable.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--audio_ext", default=".flac")
    parser.add_argument("--holdout_systems", nargs="+", default=["A16", "A17", "A18", "A19"])
    parser.add_argument("--epochs", type=int, default=10)
    main(parser.parse_args())
