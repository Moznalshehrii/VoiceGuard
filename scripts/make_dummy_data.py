"""Generate a tiny synthetic dataset + protocol file so the pipeline can be
smoke-tested before the real ASVspoof data is downloaded.

"bonafide" clips = clean sine tones (stand-in for structured human voice)
"spoof" clips    = white noise (stand-in for synthesis artifacts)
This is NOT a substitute for real data or a real result -- it only proves
the Dataset -> DataLoader -> CNN -> loss loop is wired correctly.
"""

import argparse
from pathlib import Path

import torch
import torchaudio

SAMPLE_RATE = 16000


def make_bonafide(duration_s: float) -> torch.Tensor:
    t = torch.arange(0, duration_s, 1 / SAMPLE_RATE)
    freq = 150 + torch.rand(1).item() * 250  # varying pitch
    tone = 0.3 * torch.sin(2 * torch.pi * freq * t)
    tone += 0.02 * torch.randn_like(tone)  # tiny bit of texture
    return tone.unsqueeze(0)


def make_spoof(duration_s: float) -> torch.Tensor:
    n = int(duration_s * SAMPLE_RATE)
    return (0.3 * torch.randn(1, n)).clamp(-1, 1)


def main(out_dir: str, n_per_class: int):
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    protocol_lines = []
    for i in range(n_per_class):
        duration = 2.0 + torch.rand(1).item() * 3.0  # 2-5s, variable length on purpose

        bona = make_bonafide(duration)
        fname = f"dummy_bonafide_{i:03d}"
        torchaudio.save(str(audio_dir / f"{fname}.flac"), bona, SAMPLE_RATE)
        protocol_lines.append(f"SPK_{i:03d} {fname} - - bonafide")

        spoof = make_spoof(duration)
        fname = f"dummy_spoof_{i:03d}"
        torchaudio.save(str(audio_dir / f"{fname}.flac"), spoof, SAMPLE_RATE)
        system_id = f"A{(i % 3) + 1:02d}"  # pretend 3 different fake "systems"
        protocol_lines.append(f"SPK_{i:03d} {fname} - {system_id} spoof")

    protocol_path = out_dir / "dummy_protocol.txt"
    protocol_path.write_text("\n".join(protocol_lines) + "\n")

    print(f"Wrote {2 * n_per_class} clips to {audio_dir}")
    print(f"Wrote protocol file to {protocol_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/dummy")
    parser.add_argument("--n_per_class", type=int, default=40)
    args = parser.parse_args()
    main(args.out_dir, args.n_per_class)
