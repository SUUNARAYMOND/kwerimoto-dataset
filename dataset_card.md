---
pretty_name: KweriMoto Engine Sounds
license: other
task_categories:
- audio-classification
tags:
- audio
- engine
- car
- fault-detection
- engine-knock
- misfire
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: prepared/train/**
  - split: validation
    path: prepared/validation/**
  - split: test
    path: prepared/test/**
  - split: real_test
    path: prepared/real_test/**
---

# KweriMoto Engine Sounds

Car engine sounds with three labels: **healthy**, **engine_knock** and **misfire**. The dataset combines
three public sources into one set of clips with one fixed split, so that different models can be
compared fairly. It was made for *KweriMoto, powered by EdgeKnock*, a final-year project (CSC 3118,
Makerere University) that detects engine faults from phone recordings.

All clips are **mono, 16 kHz, 16-bit WAV, at most 10 seconds** long. Shorter clips are not padded.

The code that builds the dataset: [github.com/SUUNARAYMOND/kwerimoto-dataset](https://github.com/SUUNARAYMOND/kwerimoto-dataset)

## Splits

| label | source | train | validation | test | real_test | total |
|---|---|---:|---:|---:|---:|---:|
| engine_knock | AudioSet (real) | 199 | 43 | 42 | 38 | **322** |
| engine_knock | Revix (synthetic) | 200 | 43 | 43 | 0 | **286** |
| healthy | Hugging Face | 56 | 12 | 12 | 0 | **80** |
| healthy | Revix | 190 | 40 | 41 | 0 | **271** |
| misfire | Hugging Face | 76 | 16 | 16 | 0 | **108** |
| **total** | | 721 | 154 | 154 | 38 | **1,067** |

- **train / validation / test** (70 / 15 / 15): every label from every source is split in these shares.
  Clips that share a recording (the same YouTube video, car recording or knock sound) are always in
  the same split, so a model is never tested on a recording it heard during training.
- **real_test**: 38 real engine-knock recordings from the clips Google set aside for evaluating AudioSet.
  None of them was used to make the synthetic clips, and none shares a video with any other clip here.
  It answers one question: *does a model trained here recognise real engine knocking?* It contains only
  knock, so it measures how many real knocks a model catches, not false alarms. Do not train or tune on it.

## How to load

With plain files (works anywhere, including Google Colab):

```python
from huggingface_hub import snapshot_download
import pandas as pd
import soundfile as sf

root = snapshot_download("rayyyyy007/kwerimoto-engine-sounds", repo_type="dataset")
manifest = pd.read_csv(f"{root}/manifest.csv", keep_default_na=False)
train = manifest[manifest["split"] == "train"]
audio, sample_rate = sf.read(f"{root}/{train['file_name'].iloc[0]}")  # sample_rate == 16000
```

With the `datasets` library:

```python
from datasets import load_dataset

ds = load_dataset("rayyyyy007/kwerimoto-engine-sounds")  # train, validation, test, real_test
```

With `datasets` 4 or newer, reading the audio also needs `pip install torchcodec`.

## Files

```
manifest.csv                        one row per clip (all splits)
prepared/<split>/<label>/*.wav      the clips
prepared/<split>/metadata.jsonl     that split's rows of the manifest
```

| Column | Meaning |
|---|---|
| `file_name` | path of the clip (in `manifest.csv`: from the dataset root; in `metadata.jsonl`: from the split folder) |
| `label` | `healthy`, `engine_knock` or `misfire` |
| `split` | `train`, `validation`, `test` or `real_test` |
| `group` | clips that share a recording have the same group, e.g. `youtube:DqOGNij7qIE` |
| `source_dataset` | `revix`, `huggingface` or `audioset` |
| `original_file` | file name in the source dataset |
| `source_recording` | Revix: the car recording used; others: the file itself |
| `knock_source` | Revix knock clips: the knocking sound laid on top; otherwise empty |
| `synthetic` | `True` for Revix knock clips, `False` for everything else |
| `vehicle_tag` | AudioSet clips: `car` or `untagged` (no vehicle named in Google's labels); otherwise empty |
| `duration_s` | length of the clip in seconds |
| `original_sample_rate`, `original_channels`, `original_duration_s` | format of the source file |
| `sha256` | fingerprint of the source file |

## Sources and credits

| Source | Clips used | Label here |
|---|---:|---|
| Revix engine-sounds dataset by Lwanga Caleb, Arinda Emmanuel and Ssempija Gideon Ethan: car clips | 271 + 286 | healthy, engine_knock |
| [RashmiKarunarathne/MyDataSet](https://huggingface.co/datasets/RashmiKarunarathne/MyDataSet) on Hugging Face (revision `07bb22b`): `healthy_*`, `misfire_*` | 80 + 108 | healthy, misfire |
| Google [AudioSet](https://research.google.com/audioset/) strong labels: YouTube clips labelled "Engine knocking" | 322 | engine_knock |

- **Revix knock clips are synthetic.** Revix took a real "Engine knocking" sound from AudioSet,
  filtered it, and laid it over a car recording. Revix's healthy clips are AudioSet clips tagged "Car".
- **AudioSet clips** were downloaded from YouTube. Clips tagged with another vehicle or machine
  (motorcycle, lawn mower, truck, ...) were left out; car-tagged and untagged clips were kept.
  Clips from Google's evaluation list form `real_test`.

## How it was built

1. Gather every clip from the three sources.
2. Remove files that are missing, unreadable, completely silent, exact duplicates, or exact duplicates
   with conflicting labels (49 removed out of 1,116).
3. Group clips that share a recording.
4. Split whole groups 70 / 15 / 15 (fixed seed 42); Google's evaluation clips go to `real_test`.
5. Convert to mono 16 kHz, and cut anything longer than 10 s to one random 10 s piece.

## Known issues

- **Labels are not verified.** They come from the source datasets. A Revix "Car" clip means a car can be
  heard, not that a mechanic checked the engine.
- **Background voices and music are kept on purpose.** Many YouTube clips have people talking, wind or
  music in the background (about 4 in 10 Revix clips, both healthy and knock). The final system will be used
  where people talk nearby. The Hugging Face clips are much cleaner, so misfire is never heard with
  background voices.
- **Real knock clips may not all be cars.** 264 of the 322 real clips name no vehicle at all; filter on
  `vehicle_tag == "car"` for the 58 that are certainly cars. Some Revix synthetic clips use a knock sound
  taken from a motorcycle or lawn-mower video.
- **Synthetic knock clips are shorter.** No Revix knock clip is a full 10 s, and 21 knock clips are under 1 s.
  If every clip is padded to 10 s, a model can learn "more silence means knock". Cut fixed windows from real
  audio instead of padding.
- **Sources differ in how they sound.** Revix and AudioSet clips are YouTube videos; the Hugging Face clips
  are cleaner recordings. Healthy comes from both kinds, misfire only from Hugging Face.

## Licence

This dataset is shared for research and education. The audio belongs to its original owners: Revix and
AudioSet clips come from YouTube uploads, and RashmiKarunarathne/MyDataSet states no licence. AudioSet's
labels are © Google, CC BY 4.0. Check the original sources before any other use.

## Citation

If you use the AudioSet clips, please cite:

- Gemmeke et al., "Audio Set: An ontology and human-labeled dataset for audio events", ICASSP 2017.
- Hershey et al., "The benefit of temporally-strong labels in audio event classification", ICASSP 2021.
