# Data Setup

The datasets are too large for GitHub (multi-GB each) and too large for shared
cloud storage (~24GB combined exceeds a free Google Drive account). Download
them directly using the script below -- it's already in this repo and handles
the connection drops these servers are known for (each source disconnects
every 30-90s; the script just keeps reconnecting exactly where it left off
until the file is complete).

## Which dataset do YOU need?

| Task (from the team task table) | Dataset | Download link below |
|---|---|---|
| Main model training | ASVspoof 2019 LA | #1 |
| Person 2 -- ASVspoof 2021 DF evaluation | ASVspoof 2021 DF (part00) | #2 |
| Person 3 -- In-the-Wild generalization eval | In-the-Wild | #3 |
| Person 4 -- holdout-attack generalization | ASVspoof 2019 LA | #1 |
| Person 5 -- noise/codec augmentation | ASVspoof 2019 LA | #1 |
| Person 6 -- demo | none (just needs a trained checkpoint, see below) | -- |

## Setup (everyone)

```bash
git clone https://github.com/Moznalshehrii/VoiceGuard.git
cd VoiceGuard
pip3 install --user -r requirements.txt
chmod +x scripts/resilient_download.sh
```

## 1. ASVspoof 2019 LA (~7.6 GB)

```bash
mkdir -p data/raw && cd data/raw
../../scripts/resilient_download.sh \
  "https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content" \
  "LA.zip" \
  7640952520 \
  500
unzip -q LA.zip
```

Expected result: `data/raw/LA/ASVspoof2019_LA_train/flac/` (25,380 files) and
`data/raw/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt`.

## 2. ASVspoof 2021 DF -- eval part00 (~8.6 GB)

Note: the full DF eval set is split into 4 parts (~34GB total). We're only
using part00 -- a representative subset -- to save time. If more data is
ever needed later, parts 01-03 follow the same pattern with these sizes:
part01 = 8624944890, part02 = 8616337195, part03 = 8656151261 bytes.

```bash
mkdir -p data/raw && cd data/raw
../../scripts/resilient_download.sh \
  "https://zenodo.org/api/records/4835108/files/ASVspoof2021_DF_eval_part00.tar.gz/content" \
  "ASVspoof2021_DF_eval_part00.tar.gz" \
  8637050238 \
  500
tar -xzf ASVspoof2021_DF_eval_part00.tar.gz
```

Expected result: `data/raw/ASVspoof2021_DF_eval/flac/` and
`data/raw/ASVspoof2021_DF_eval/ASVspoof2021.DF.cm.eval.trl.txt`.

## 3. In-the-Wild (~8.2 GB)

```bash
mkdir -p data/raw && cd data/raw
curl -L -C - --retry 10 --retry-delay 5 --retry-all-errors \
  -o release_in_the_wild.zip \
  "https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip"
unzip -q release_in_the_wild.zip
```

This host has been reliable (no resets observed), so plain `curl --retry` is
enough -- no need for the resilient loop script here.

Expected result: `data/raw/release_in_the_wild/` with 31,781 `.wav` files
plus `meta.csv`.

## Verifying a download finished correctly

```bash
unzip -tq <file>.zip      # for .zip files -- should print "No errors detected"
tar -tzf <file>.tar.gz | wc -l   # for .tar.gz -- just confirms it's readable
```

If a download stalls or errors out partway, just re-run the same command --
`resilient_download.sh` and `curl -C -` both resume from where they stopped
instead of starting over.

## Trained checkpoint (for Person 3 and Person 6)

Ask in the team chat for the current best checkpoint
(`checkpoints/wav2vec_spoof_real.pt`, ~1.2GB) -- it's small enough to share
directly rather than needing a fresh download.
