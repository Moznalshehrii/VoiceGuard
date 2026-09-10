# Voice Guard

Audio deepfake detector: given a speech clip, classify it as real human speech
or AI-generated (voice cloning / TTS). Core problem the project is built
around is **generalization** — most detectors fail on synthesis methods they
weren't trained on. See `voice-guard-brief.pdf` for the full brief.

## Status

- [x] Step A — data pipeline (protocol parsing, audio -> Mel-spectrogram, PyTorch Dataset)
- [x] Step B — baseline CNN + training loop, validated on synthetic data
- [x] Step C — Wav2Vec2/XLSR main model, training in progress on real ASVspoof2019 LA (see Results below)
- [x] Step D — generalization test (train on ASVspoof19, eval on In-the-Wild) — model collapses on real-world audio, see Results
- [x] Step D-extra — holdout-attack generalization within ASVspoof19 — Person 4
- [x] Step D-extra — evaluation on ASVspoof 2021 DF — Person 2
- [x] Step E — robustness (noise/codec augmentation) — Person 5
- [x] Step F — demo (Gradio, branded "Sawt / صوت") — live and tested end-to-end

## Team tasks

Each task lives in its own file with `TODO` comments marking exactly what's
needed. Clone the repo, create a branch per task, and open a PR when done —
see the task assignment table shared with the team for who owns what.

| File | Owner | Status |
|---|---|---|
| `src/evaluate.py` | Person 2 | implemented |
| `src/train_holdout.py` | Person 4 | implemented (bug fixed -- see Results notes) |
| `src/data/augmented_dataset.py` | Person 5 | implemented (moved from a stray duplicate path -- see Results notes) |
| `app.py` | Person 6 | implemented, deployed locally with a temporary public link |

## Results

_Filled in as each task completes. Report EER and min-DCF, not accuracy._

min-DCF uses p_target=0.05, c_miss=1, c_fa=1 (see `src/utils/metrics.py`) --
normalized so a trivial always-accept/always-reject system scores 1.0.

| Eval set | EER | min-DCF | Notes |
|---|---|---|---|
| ASVspoof19 LA (held-out split of train, seen attack systems) | 0.41% | 0.0305 | After 2/3 planned epochs — near-zero expected here since it's the *same* known attack systems as training, not a generalization test |
| ASVspoof 2021 DF (eval part00) | 35.42% | 0.8582 | 36,012 scored trials; 2 unreadable FLAC files skipped |
| In-the-Wild | 62.01%* | 0.9694* | 31,779 trials — **this is the project's key result, see caveat below** |
| ASVspoof19 LA dev set (seen: A01-A06, same systems as training) | 0.84% | 0.0463 | corrected seen/unseen split -- see note below |
| ASVspoof19 LA eval set (unseen: A07-A19, never in training) | 6.38% | 0.6629 | 71,237 trials; genuinely unseen attack systems |
| With vs. without augmentation, on unseen eval set (A07-A19) | With 1.84%, without 6.38% | With 0.1665, without 0.6629 | re-verified after Step E fix -- augmentation gives a real, substantial improvement on unseen-*attack* generalization |
| With vs. without augmentation, on In-the-Wild | No change (both collapse) | No change | augmentation does NOT fix the real-world domain-shift collapse -- noise/codec augmentation ≠ the actual diversity of real-world recording conditions |

**Note on the In-the-Wild numbers (*):** EER/min-DCF are reported for
completeness, but they understate what's actually happening. Direct
inspection of individual predictions (23 samples checked by hand, varied
lengths and both labels, including several clips ≥4s so padding isn't a
factor) shows the model outputs an almost-constant `P(bonafide) ≈ 0.0009`
regardless of the true label -- it isn't discriminating between real and
fake on this data at all, just defaulting to "spoof" with high confidence
every time. An EER above 50% is the tell: it means the score carries no
real signal (not an inverted one). This is a stronger version of the
project's core finding than the number alone suggests -- the model doesn't
just get *less accurate* on real-world audio, it fails to generalize
entirely, likely because the classifier head saturates when fed acoustic
conditions (different recording devices, compression, noise) far outside
the clean lab-recorded ASVspoof distribution it was fine-tuned on.

**Note on the holdout-attack numbers above:** `train_holdout.py`'s original
`build_seen_and_heldout()` mislabeled its "seen" set -- ASVspoof2019 LA's
train/dev protocols only use attack systems A01-A06, while the eval
protocol uses A07-A19 exclusively (zero overlap, confirmed directly from
the protocol files). So excluding 4 systems (A16-A19) from "the rest of
eval" doesn't produce a seen set -- none of eval's systems were ever
trained on. That's why the original run showed held-out (3.36%) EER
*lower* than "seen" (which was never reported) -- both were actually
unseen data. The two rows above replace it with the dataset's real
seen/unseen split (dev vs. eval), which shows the expected direction:
0.84% (seen) -> 6.38% (unseen).

The augmentation row hasn't been re-verified yet -- it was computed before
`src/data/augmented_dataset.py`'s real implementation was moved to the
correct path (it had been sitting unused in a stray duplicate folder), so
it's unclear which code actually produced those numbers.

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
