"""Run every step and write the combined dataset.

    uv run kwerimoto-build --revix-root "../AI data/AI data"

Output (under --out, default "data"):
    raw/          untouched copies of the source files
    prepared/     mono, 16 kHz, 16-bit WAV, at most 10 s, in <split>/<label>/ folders;
                  each split folder also has a metadata.jsonl (its rows of the manifest)
    manifest.csv  one row per prepared clip, with its label, split and source
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

from .audio import SAMPLE_RATE, prepare_audio, write_wav
from .audioset import download_audioset, read_audioset
from .checks import run_checks
from .grouping import assign_groups
from .sources import download_huggingface, read_huggingface, read_revix_cars
from .splitting import split_groups

MANIFEST_COLUMNS = ["file_name", "label", "split", "group", "source_dataset", "original_file",
                    "source_recording", "knock_source", "synthetic", "vehicle_tag", "duration_s",
                    "original_sample_rate", "original_channels", "original_duration_s", "sha256"]
REAL_TEST = "real_test"  # Google's evaluation-set knock clips: never trained on


def copy_revix(revix, raw_dir):
    """Copy Revix files into raw/revix/<their folder>/ and point raw_path at the copy."""
    raw_paths = []
    for source in map(Path, revix["source_path"]):
        target = Path(raw_dir, "revix", source.parent.name, source.name)
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        raw_paths.append(str(target))
    return revix.assign(raw_path=raw_paths)


def build_dataset(revix_root, out_dir="data", hf_dir=None, audioset_dir=None, seed=42):
    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw"

    # 1. Gather
    revix = copy_revix(read_revix_cars(revix_root), raw_dir)
    if hf_dir is None:
        hf_dir = download_huggingface(raw_dir / "huggingface")
    hf = read_huggingface(hf_dir)
    if audioset_dir is None:
        audioset_dir = download_audioset(raw_dir / "audioset")
    audioset = read_audioset(audioset_dir)
    candidates = pd.concat([revix, hf, audioset], ignore_index=True)
    candidates["raw_path"] = candidates["raw_path"].fillna(candidates["source_path"])
    for column in ["google_set", "vehicle_tag"]:  # AudioSet-only columns
        candidates[column] = candidates.get(column, pd.Series(dtype=str)).fillna("")
    if candidates["record_id"].duplicated().any():
        raise ValueError("two candidate files share a record_id")

    # 2. Check
    kept, removed = run_checks(candidates)

    # 3-4. Group linked clips, then split whole groups. Google's evaluation-set
    # knock clips are kept apart as the real-knock test.
    kept["group"] = assign_groups(kept)
    real_test = kept["google_set"] == "eval"
    shared = set(kept.loc[real_test, "group"]) & set(kept.loc[~real_test, "group"])
    if shared:
        raise ValueError(f"real_test clips share recordings with other clips: {sorted(shared)}")
    kept["stratum"] = kept["label"] + "/" + kept["source_dataset"]
    kept["split"] = REAL_TEST
    kept.loc[~real_test, "split"] = split_groups(kept[~real_test], seed=seed, by="stratum")

    # 5. Prepare audio
    prepared_dir = out_dir / "prepared"
    shutil.rmtree(prepared_dir, ignore_errors=True)  # generated; rebuilt from raw every run
    file_names, durations = [], []
    for row in kept.itertuples(index=False):
        audio = prepare_audio(row.raw_path, record_id=row.record_id, seed=seed)
        file_name = f"prepared/{row.split}/{row.label}/{row.record_id}.wav"
        write_wav(out_dir / file_name, audio)
        file_names.append(file_name)
        durations.append(round(len(audio) / SAMPLE_RATE, 3))
    kept["file_name"] = file_names
    kept["duration_s"] = durations

    manifest = kept[MANIFEST_COLUMNS].sort_values("file_name", ignore_index=True)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    # What Hugging Face reads to build each split. JSON keeps each column's type, so a column
    # that is empty in one split (knock_source in real_test) still matches the others.
    for split, part in manifest.groupby("split"):
        part.assign(file_name=part["file_name"].str.removeprefix(f"prepared/{split}/")).to_json(
            prepared_dir / split / "metadata.jsonl", orient="records", lines=True)
    print_summary(candidates, manifest, removed)
    return manifest


def print_summary(candidates, manifest, removed):
    print(f"\nCandidates: {len(candidates)}   kept: {len(manifest)}   removed: {len(removed)}")
    if len(removed):
        print("\nRemoved (reason x label):")
        print(pd.crosstab(removed["reason"], removed["label"], margins=True, margins_name="total").to_string())
    print("\nKept (label and source x split):")
    print(pd.crosstab([manifest["label"], manifest["source_dataset"]], manifest["split"],
                      margins=True, margins_name="total")
          .reindex(columns=["train", "validation", "test", REAL_TEST, "total"], fill_value=0).to_string())
    print(f"\nGroups: {manifest['group'].nunique()}   largest group: {manifest['group'].value_counts().max()} clips")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--revix-root", default="../AI data/AI data",
                        help="folder containing engine_sounds_hf_dataset/")
    parser.add_argument("--out", default="data", help="output folder")
    parser.add_argument("--hf-dir", default=None,
                        help="existing Hugging Face download (skips downloading)")
    parser.add_argument("--audioset-dir", default=None,
                        help="existing AudioSet folder with labels/ and clips/ (skips downloading)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_dataset(args.revix_root, args.out, args.hf_dir, args.audioset_dir, args.seed)


if __name__ == "__main__":
    main()
