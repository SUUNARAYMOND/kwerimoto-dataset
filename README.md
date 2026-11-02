# KweriMoto combined engine-sound dataset

Builds one dataset of car-engine sounds with three labels, **healthy**, **engine_knock** and **misfire**,
from three public sources. Every model in the project (MFCC + XGBoost, frozen YAMNet + classifier, AST)
trains on these same prepared clips and the same train / validation / test split, so their results
can be compared fairly.

## Result

| label | source | train | validation | test | real_test | total |
|---|---|---:|---:|---:|---:|---:|
| engine_knock | AudioSet (real) | 199 | 43 | 42 | 38 | **322** |
| engine_knock | Revix (synthetic) | 200 | 43 | 43 | 0 | **286** |
| healthy | Hugging Face | 56 | 12 | 12 | 0 | **80** |
| healthy | Revix | 190 | 40 | 41 | 0 | **271** |
| misfire | Hugging Face | 76 | 16 | 16 | 0 | **108** |
| **total** | | 721 | 154 | 154 | 38 | **1,067** |

Every source is split 70 / 15 / 15. **`real_test`** holds real knock clips that no model may train or tune
on. It answers one question: *does a model trained here recognise real engine knocking?*

## Sources

| Source | Used | Label here |
|---|---:|---|
| Revix engine-sounds dataset (Lwanga Caleb, Arinda Emmanuel, Ssempija Gideon Ethan), `car_clean` | 300 | healthy |
| Revix, `car_knocking` (synthetic) | 300 listed, 299 present | engine_knock |
| Hugging Face [`RashmiKarunarathne/MyDataSet`](https://huggingface.co/datasets/RashmiKarunarathne/MyDataSet), `healthy_*` | 83 | healthy |
| Hugging Face `RashmiKarunarathne/MyDataSet`, `misfire_*` | 111 | misfire |
| Google [AudioSet strong labels](https://research.google.com/audioset/download_strong.html), "Engine knocking" clips | 338 selected, 322 downloaded | engine_knock |

Not used: Revix truck clips, and the Hugging Face `tappet_issue` and `belt_issue` files (different faults).

- **Revix** clips come from AudioSet (YouTube). `car_clean` clips are AudioSet clips tagged "Car".
  **Every `car_knocking` clip is synthetic**: Revix took a real "Engine knocking" clip, filtered out the
  knock, and laid it on top of a car recording. Our `synthetic` column copies Revix's own field.
- **Hugging Face** files are downloaded at a fixed revision
  (`07bb22b36e86762d9b06e3652ea00194b1b89cec`), so everyone gets the same files.
- **AudioSet** lists 409 YouTube clips labelled "Engine knocking" (349 in Google's training set, 60 in
  Google's evaluation set). We keep the clips tagged as a car (from both the strong and the clip-level
  labels) or not tagged with any vehicle, and leave out those tagged as another vehicle or machine
  (motorcycle, lawn mower, truck, bus, chainsaw, boat, plane). That leaves 297 + 41 = 338 clips. Each
  10-second clip is downloaded from YouTube; 16 videos are gone (deleted or private). Clips from
  Google's **evaluation** set become `real_test`. Revix never used any of them, and none shares a video
  with any other clip here.

## How to run

Needs [uv](https://docs.astral.sh/uv/). From this folder:

```
uv sync
uv run kwerimoto-build --revix-root "../AI data/AI data"
```

The first run downloads the Hugging Face files (about 200 MB), the AudioSet label files (about 150 MB) and
the AudioSet clips from YouTube (15 to 30 minutes), then builds:

```
data/
  raw/                untouched copies of the source files
    revix/car_clean/, revix/car_knocking/
    huggingface/MyDataSet/
    audioset/labels/, audioset/clips/, audioset/download_failures.csv
  prepared/           the clips to train on
    train/, validation/, test/, real_test/
      <label>/          healthy/, engine_knock/, misfire/
      metadata.jsonl    that split's rows of the manifest (read by Hugging Face)
  manifest.csv        one row per prepared clip
```

Later runs reuse what is already downloaded. Downloads that failed because of the network are retried;
videos that are gone are skipped. Rebuilding gives exactly the same clips and split (fixed seed,
`--seed 42`). Run the tests with `uv run pytest`; they use small fake datasets and never download.

## Sharing

The built dataset is on Hugging Face as
[`rayyyyy007/kwerimoto-engine-sounds`](https://huggingface.co/datasets/rayyyyy007/kwerimoto-engine-sounds);
its page text comes from `dataset_card.md`. To upload a new build (log in once first with
`uv run hf auth login` and a write token):

```
uv run kwerimoto-upload --repo rayyyyy007/kwerimoto-engine-sounds
```

This uploads `manifest.csv`, `prepared/` and the card, never `raw/`.

## Notebooks

Model notebooks live in `notebooks/`. They run in Google Colab and load the dataset from Hugging Face.

| Notebook | Model | |
|---|---|---|
| `yamnet_baseline.ipynb` | frozen YAMNet features + small three-class classifier | [Open in Colab](https://colab.research.google.com/github/SUUNARAYMOND/kwerimoto-dataset/blob/main/notebooks/yamnet_baseline.ipynb) |

## What each step does

1. **Gather.** List every Revix car clip from Revix's metadata (both of their splits), every Hugging Face
   healthy and misfire file, and every downloaded AudioSet knock clip. Copy the Revix files into
   `data/raw/` unchanged.
2. **Check.** Remove a file if it is:
   - **missing**: listed in the metadata but not on disk;
   - **unreadable**: cannot be opened as audio, or has no samples;
   - **silent**: every sample is exactly zero (quiet recordings are kept);
   - **duplicate**: byte-for-byte identical to another file with the same label (one copy is kept);
   - **duplicate with conflicting labels**: identical files carrying different labels. We cannot tell
     which label is right, so all copies are removed.
3. **Group.** Link clips that share a recording, so they always land in the same split. Revix and AudioSet
   clips are linked when they share a car recording, a knock sound, or any piece of the same YouTube video
   (AudioSet names are `<video id>_<start ms>.wav`). A real AudioSet clip therefore joins every Revix mix
   that used its video. Links chain: if A is linked to B and B to C, all three are one group. Each Hugging
   Face file is its own group (the filenames do not say which car it came from).
4. **Split.** Clips from Google's evaluation set go to `real_test`. All other groups are assigned whole to
   train / validation / test (70 / 15 / 15), keeping **each label from each source** close to those shares,
   so real and synthetic knock are both balanced. Groups are placed largest first, then a second pass
   moves single groups between splits while that brings every label and source closer to its target.
5. **Prepare.** For each kept file: average the channels to mono, resample to 16 kHz, and, following Revix,
   cut anything longer than 10 s to one random 10 s piece (the same piece on every rebuild). Shorter clips
   are **not padded**. Save as 16-bit WAV in `prepared/<split>/<label>/`.
6. **Optional windowing.** `kwerimoto_data.audio.cut_windows` cuts a clip into fixed-length windows (with
   optional overlap and padding) for window-length experiments. It is not applied to the stored dataset.

## Manifest columns

| Column | Meaning |
|---|---|
| `file_name` | path of the prepared clip, relative to `data/` |
| `label` | `healthy`, `engine_knock` or `misfire` |
| `split` | `train`, `validation`, `test` or `real_test` |
| `group` | linked-recording group, e.g. `youtube:DqOGNij7qIE` or `huggingface:misfire_007` |
| `source_dataset` | `revix`, `huggingface` or `audioset` |
| `original_file` | file name in the source dataset |
| `source_recording` | Revix: the car recording used; Hugging Face and AudioSet: the file itself |
| `knock_source` | Revix knock clips: the knocking sound laid on top; otherwise empty |
| `synthetic` | `True` for Revix knock clips, `False` for everything else |
| `vehicle_tag` | AudioSet clips: `car` or `untagged`; otherwise empty |
| `duration_s` | length of the prepared clip |
| `original_sample_rate`, `original_channels`, `original_duration_s` | format of the source file |
| `sha256` | fingerprint of the source file |

Loading the data (from Hugging Face, see the loading examples in `dataset_card.md`):

```python
import pandas as pd
manifest = pd.read_csv("data/manifest.csv", keep_default_na=False)
train = manifest[manifest["split"] == "train"]
real_test = manifest[manifest["split"] == "real_test"]   # only for the final real-knock check
```

## What was removed

| Reason | engine_knock | healthy | misfire | total |
|---|---:|---:|---:|---:|
| duplicate | 0 | 27 | 2 | 29 |
| silent | 12 | 3 | 0 | 15 |
| duplicate with conflicting labels | 1 | 2 | 1 | 4 |
| missing | 1 | 0 | 0 | 1 |
| **total** | 14 | 32 | 3 | **49** |

- Missing: Revix `car_knocking_0029.wav`.
- Conflicting labels: Revix `car_knocking_0265` is identical to `car_clean_0294` (its knock sound was never
  added); Hugging Face `healthy_062` is identical to `misfire_001`.
- Hugging Face duplicates: `healthy_009` = `healthy_002`, `healthy_064` = `healthy_063`,
  `misfire_009` = `misfire_008`, `misfire_055` = `misfire_054`.
- AudioSet: all 322 downloaded clips passed the checks. The 16 that could not be downloaded are listed in
  `data/raw/audioset/download_failures.csv`.

## Known issues

- **Labels are not verified.** Labels come from the source datasets. Our checks catch file problems, not
  wrong labels. A Revix "Car" clip means a car can be heard, not that a mechanic checked the engine, and
  the Hugging Face dataset does not say how its labels were decided.
- **Some real knock clips may not be cars.** 264 of the 322 real clips carry no vehicle tag at all. Use
  `vehicle_tag == "car"` to look at the 58 that are certainly cars.
- **Some synthetic knock clips use a non-car knock sound.** Revix did not filter knock sounds by vehicle:
  of the 181 knock sounds used in their car mixes, 28 come from clips tagged as another vehicle or machine
  (18 motorcycle, 6 lawn mower, 5 truck, 1 plane, 1 air brake).
- **Synthetic knock clips are shorter than healthy clips.** No Revix knock clip is a full 10 s (each mix is cut
  to the shorter of its two ingredients), while 154 Revix healthy clips are, and 23 clips are under 1 s (21 of
  them knock; the shortest is 0.13 s). If a model pads every clip to one fixed length, short clips get mostly
  silence, and the model could learn "more silence means knock". The real knock clips (almost all 10 s)
  weaken this, but choose the training window length with it in mind, for example by cutting fixed windows
  from real audio with `cut_windows` instead of padding everything to 10 s.
- **Real knock clips come from different videos than the healthy clips**, so their background (wind,
  speech, traffic, microphone) differs. Training on both synthetic mixes (same backgrounds as healthy) and
  real clips (no overlay processing) is meant to keep the model listening to the knock itself.
- **Background voices and music are kept on purpose.** By Google's labels, speech or music is heard in 104
  of the 271 Revix healthy clips, under 115 of the 286 Revix knock clips (in the car recording beneath the
  knock), and in 50 of the 322 real knock clips. The finished system will be used where people talk nearby,
  so models should learn to ignore these sounds. Because Revix healthy and Revix knock have about the same
  share, voices do not point to one label. The Hugging Face clips sound much cleaner, so misfire is never
  heard with voices in the background; adding background noise during training (augmentation) would
  cover that.
- **`real_test` only contains knock.** It measures how many real knocks a model catches, not how often
  it raises false alarms.
- **Hugging Face groups may be too fine.** Each Hugging Face file is treated as a separate recording.
  Neighbouring files that turned out to be exact copies (e.g. `misfire_008` and `misfire_009`) suggest some
  files may come from the same car, which the filenames cannot show.
- **Licensing.** AudioSet labels are CC BY 4.0 (Google). The audio in Revix and AudioSet clips comes from
  YouTube uploads. The Hugging Face dataset states no licence.
