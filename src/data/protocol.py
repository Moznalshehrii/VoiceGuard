"""Parsers that normalize different dataset protocol formats into one schema:

    filepath (str), speaker_id (str), system_id (str), label (int: 1=bonafide, 0=spoof)

Downstream code (Dataset, training, eval) only ever sees this normalized schema,
so swapping ASVspoof19 -> ASVspoof21 -> In-the-Wild is a one-line change.
"""

from pathlib import Path

import pandas as pd

LABEL_TO_INT = {"bonafide": 1, "spoof": 0, "real": 1, "fake": 0, "bona-fide": 1}


def parse_asvspoof_protocol(protocol_path: str, audio_dir: str, audio_ext: str = ".flac") -> pd.DataFrame:
    """Parse an ASVspoof 2019/2021-style protocol file.

    Expected columns (whitespace separated, no header):
        speaker_id  filename  codec_or_dash  system_id  label
    Some ASVspoof2021 protocol files add extra trailing columns (e.g. compression
    info) — we only rely on the first, second, and last columns, which are stable
    across the 2019/2021 releases.
    """
    audio_dir = Path(audio_dir)
    rows = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            speaker_id, filename = parts[0], parts[1]
            # 2019 LA rows have five fields and include a system id.  The
            # labelled 2021 DF key file has only ``speaker filename label``.
            system_id = parts[-2] if len(parts) >= 5 else "-"
            label_str = parts[-1].lower()
            if label_str not in LABEL_TO_INT:
                raise ValueError(f"Unrecognized label {label_str!r} in {protocol_path}")
            rows.append(
                {
                    "filepath": str(audio_dir / f"{filename}{audio_ext}"),
                    "speaker_id": speaker_id,
                    "system_id": system_id,
                    "label": LABEL_TO_INT[label_str],
                }
            )
    return pd.DataFrame(rows)


def parse_in_the_wild_meta(meta_csv_path: str, audio_dir: str) -> pd.DataFrame:
    """Parse the In-the-Wild `meta.csv` (columns: file, speaker, label)."""
    audio_dir = Path(audio_dir)
    df = pd.read_csv(meta_csv_path)
    df = df.rename(columns={"file": "filename", "speaker": "speaker_id"})
    df["label"] = df["label"].str.lower().map(LABEL_TO_INT)
    df["system_id"] = "itw"  # no system_id concept for real-world scraped deepfakes
    df["filepath"] = df["filename"].apply(lambda fn: str(audio_dir / fn))
    return df[["filepath", "speaker_id", "system_id", "label"]]


def summarize(df: pd.DataFrame) -> str:
    n_bonafide = int((df["label"] == 1).sum())
    n_spoof = int((df["label"] == 0).sum())
    n_systems = df.loc[df["label"] == 0, "system_id"].nunique()
    return (
        f"{len(df)} clips total | bonafide={n_bonafide} spoof={n_spoof} "
        f"| {n_systems} distinct spoof system(s)"
    )
