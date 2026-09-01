# Voice Guard

Audio deepfake detector: given a speech clip, classify it as real human speech
or AI-generated (voice cloning / TTS). Core problem the project is built
around is **generalization** — most detectors fail on synthesis methods they
weren't trained on. See `voice-guard-brief.pdf` for the full brief.

## Status

- [x] Step A — data pipeline (protocol parsing, audio -> Mel-spectrogram, PyTorch Dataset)
- [x] Step B — baseline CNN + training loop, validated on synthetic data
- [x] Step C — Wav2Vec2/XLSR main model, training in progress on real ASVspoof2019 LA (see Results below)
- [ ] Step D — generalization test (train on ASVspoof19, eval on In-the-Wild) — Person 3
- [ ] Step D-extra — holdout-attack generalization within ASVspoof19 — Person 4
- [ ] Step D-extra — evaluation on ASVspoof 2021 DF — Person 2
- [ ] Step E — robustness (noise/codec augmentation) — Person 5
- [ ] Step F — demo + final EER result tables — Person 6

## Team tasks

Each task lives in its own file with `TODO` comments marking exactly what's
needed. Clone the repo, create a branch per task, and open a PR when done —
see the task assignment table shared with the team for who owns what.

| File | Owner | Status |
|---|---|---|
| `src/evaluate.py` | Person 2 | skeleton, not implemented |
| `src/train_holdout.py` | Person 4 | skeleton, not implemented |
| `src/data/augmented_dataset.py` | Person 5 | skeleton, not implemented |
| `app.py` | Person 6 | skeleton, not implemented |

## Results

_Filled in as each task completes. Report EER and min-DCF, not accuracy._

min-DCF uses p_target=0.05, c_miss=1, c_fa=1 (see `src/utils/metrics.py`) --
normalized so a trivial always-accept/always-reject system scores 1.0.

| Eval set | EER | min-DCF | Notes |
|---|---|---|---|
| ASVspoof19 LA (held-out split of train, seen attack systems) | 0.41% | 0.0305 | After 2/3 planned epochs — near-zero expected here since it's the *same* known attack systems as training, not a generalization test |
| ASVspoof 2021 DF | — | — | pending (Person 2) |
| In-the-Wild | — | — | pending (Person 3) — **this is the project's key result** |
| ASVspoof19 LA, held-out attack systems | — | — | pending (Person 4) |
| With vs. without augmentation | — | — | pending (Person 5) |

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

## Getting real data

See [DATA_SETUP.md](DATA_SETUP.md) for exact download commands per dataset
(ASVspoof 2019 LA, ASVspoof 2021 DF, In-the-Wild) and which one each task
needs. The datasets are too large for GitHub/cloud storage, so everyone
downloads directly from the original source using a script in this repo
that handles the connection drops those servers are prone to.

Once you have ASVspoof 2019 LA:

```bash
python3 -m src.train_wav2vec \
  --protocol data/raw/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
  --audio_dir data/raw/LA/ASVspoof2019_LA_train/flac \
  --epochs 3
```

(`src/train.py` also exists for the baseline CNN from Step B, same arguments.)

Metric reported is **EER** (Equal Error Rate), not accuracy — that's the
standard metric in the anti-spoofing literature and the one the brief asks
to report.
