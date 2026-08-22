# Voice Guard

Audio deepfake detector: given a speech clip, classify it as real human speech
or AI-generated (voice cloning / TTS). Core problem the project is built
around is **generalization** — most detectors fail on synthesis methods they
weren't trained on. See `voice-guard-brief.pdf` for the full brief.

## Status

- [x] Step A — data pipeline (protocol parsing, audio -> Mel-spectrogram, PyTorch Dataset)
- [x] Step B — baseline CNN + training loop, validated on synthetic data
- [ ] Step C — fine-tuned Wav2Vec2/XLSR main model
- [ ] Step D — generalization test (train on ASVspoof19, eval on In-the-Wild)
- [ ] Step E — robustness (noise/codec augmentation)
- [ ] Step F — demo + EER result tables

## Layout

```
src/
  data/
    protocol.py    parses ASVspoof + In-the-Wild protocol files into one schema
    dataset.py      audio loading, Mel-spectrogram extraction, PyTorch Dataset
  models/
    cnn.py          baseline CNN (step B)
  utils/
    metrics.py      Equal Error Rate (EER) computation
  train.py           training loop
scripts/
  make_dummy_data.py synthetic data generator, for smoke-testing the pipeline
data/
  raw/               <- put real datasets here (gitignored)
  dummy/              synthetic smoke-test data (gitignored)
checkpoints/          saved model weights (gitignored)
```

## Setup

```bash
pip3 install --user -r requirements.txt
```

## Smoke test (no real data needed)

```bash
python3 scripts/make_dummy_data.py --out_dir data/dummy --n_per_class 40
python3 -m src.train --protocol data/dummy/dummy_protocol.txt --audio_dir data/dummy/audio --epochs 5
```

## Getting real data (ASVspoof 2019 LA)

1. Download from asvspoof.org or the Edinburgh DataShare mirror. You want:
   - `LA/ASVspoof2019_LA_train/flac/` — training audio
   - `LA/ASVspoof2019_LA_dev/flac/` — dev audio
   - `LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt` — training labels
2. Place under `data/raw/` (or point `--audio_dir`/`--protocol` anywhere).
3. Train:

```bash
python3 -m src.train \
  --protocol data/raw/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
  --audio_dir data/raw/LA/ASVspoof2019_LA_train/flac \
  --epochs 20
```

Metric reported is **EER** (Equal Error Rate), not accuracy — that's the
standard metric in the anti-spoofing literature and the one the brief asks
to report.
