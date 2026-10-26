"""Real engine-knock clips from Google's AudioSet (strong labels).

AudioSet lists YouTube clips with the sounds heard in them; the audio itself
has to be downloaded from YouTube. We take every clip labelled
"Engine knocking" and keep those tagged as a car or not tagged with any
vehicle; clips tagged as another vehicle or machine (motorcycle, lawn mower,
truck, ...) are left out. Vehicle tags come from both the strong labels and
AudioSet's clip-level labels.

Folder layout (under raw/audioset/):
    labels/   AudioSet label files (CC BY 4.0, Google)
    clips/    one WAV per downloaded clip, named <segment id>.wav
    download_failures.csv   clips that could not be downloaded; only videos that
                            are gone (deleted, private) are skipped on the next run
"""
import csv
import os
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

KNOCK_MID = "/m/01h82_"
GOOGLE_SETS = ["train", "eval"]
KEEP_VEHICLE_TAGS = {"car", "untagged"}

CAR_TAGS = {"Car", "Vehicle horn, car horn, honking", "Car alarm", "Car passing by", "Race car, auto racing",
            "Power windows, electric windows", "Skidding", "Tire squeal"}
OTHER_TAGS = {"Motorcycle", "Lawn mower", "Chainsaw", "Motorboat, speedboat", "Boat, Water vehicle", "Train",
              "Aircraft", "Fixed-wing aircraft, airplane", "Helicopter", "Tractor", "Truck", "Bus", "Snowmobile",
              "Dirt bike", "Moped", "Power tool", "Ship", "Rail transport", "Air brake",
              "Ice cream truck, ice cream van", "Fire engine, fire truck (siren)"}

STRONG_URL = "https://storage.googleapis.com/us_audioset/youtube_corpus/strong/"
CLIP_LEVEL_URL = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/"
LABEL_FILES = {
    "audioset_train_strong.tsv": STRONG_URL, "audioset_eval_strong.tsv": STRONG_URL,
    "mid_to_display_name.tsv": STRONG_URL, "class_labels_indices.csv": CLIP_LEVEL_URL,
    "eval_segments.csv": CLIP_LEVEL_URL, "balanced_train_segments.csv": CLIP_LEVEL_URL,
    "unbalanced_train_segments.csv": CLIP_LEVEL_URL,
}


def _clip_level_labels(labels_dir):
    """{segment id: set of MIDs} from AudioSet's 10-second clip labels."""
    labels = {}
    for name in ["eval_segments.csv", "balanced_train_segments.csv", "unbalanced_train_segments.csv"]:
        with open(Path(labels_dir, name), newline="") as f:
            for row in csv.reader(f, skipinitialspace=True):
                if row and not row[0].startswith("#"):
                    labels[f"{row[0]}_{round(float(row[1]) * 1000)}"] = set(row[3].split(","))
    return labels


def _vehicle_tag(tags):
    car, other = bool(tags & CAR_TAGS), bool(tags & OTHER_TAGS)
    if car and other:
        return "car+other"
    return "car" if car else "other" if other else "untagged"


def select_knock_clips(labels_dir):
    """Every "Engine knocking" clip, with its Google set and vehicle tag."""
    names = dict(pd.read_csv(Path(labels_dir, "mid_to_display_name.tsv"), sep="\t", header=None).values)
    indices = pd.read_csv(Path(labels_dir, "class_labels_indices.csv"))
    names.update(dict(zip(indices["mid"], indices["display_name"])))
    clip_level = _clip_level_labels(labels_dir)

    rows = []
    for google_set in GOOGLE_SETS:
        strong = pd.read_csv(Path(labels_dir, f"audioset_{google_set}_strong.tsv"), sep="\t")
        knock = sorted(set(strong.loc[strong["label"] == KNOCK_MID, "segment_id"]))
        mids = strong[strong["segment_id"].isin(knock)].groupby("segment_id")["label"].apply(set)
        for segment in knock:
            video, start_ms = segment.rsplit("_", 1)
            tags = {names.get(m, m) for m in mids[segment] | clip_level.get(segment, set())}
            rows.append({"segment_id": segment, "google_set": google_set, "video_id": video,
                         "start_s": int(start_ms) / 1000, "vehicle_tag": _vehicle_tag(tags)})
    return pd.DataFrame(rows)


def read_audioset(audioset_dir):
    """Candidate rows for the kept clips that have been downloaded."""
    audioset_dir = Path(audioset_dir)
    clips = select_knock_clips(audioset_dir / "labels")
    clips = clips[clips["vehicle_tag"].isin(KEEP_VEHICLE_TAGS)]
    candidates = []
    for clip in clips.itertuples(index=False):
        path = audioset_dir / "clips" / f"{clip.segment_id}.wav"
        if not path.exists():
            continue
        candidates.append({
            "record_id": f"audioset_{clip.segment_id}",
            "label": "engine_knock",
            "source_dataset": "audioset",
            "original_file": path.name,
            "source_path": str(path),
            "source_recording": path.name,
            "knock_source": "",
            "synthetic": False,
            "google_set": clip.google_set,
            "vehicle_tag": clip.vehicle_tag,
        })
    return pd.DataFrame(candidates)


def _ffmpeg_on_path(tools_dir):
    """yt-dlp needs a program called exactly "ffmpeg"; copy the bundled one."""
    import imageio_ffmpeg

    tools_dir = Path(tools_dir)
    target = tools_dir / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    if not target.exists():
        tools_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), target)
        target.chmod(0o755)
    os.environ["PATH"] = f"{tools_dir}{os.pathsep}{os.environ['PATH']}"


def is_permanent_failure(error):
    """True when the video itself is gone; network or ffmpeg errors are worth retrying."""
    return any(sign in error for sign in
               ["unavailable", "not available", "Private video", "terminated", "removed"])


def _download_clip(segment_id, video, start_s, clips_dir):
    """Download one 10-second piece from YouTube as WAV. Returns an error message or None."""
    import yt_dlp
    from yt_dlp.utils import download_range_func

    options = {
        "format": "bestaudio/best",
        "outtmpl": str(clips_dir / f"{segment_id}.%(ext)s"),
        "download_ranges": download_range_func([], [(start_s, start_s + 10)]),
        "force_keyframes_at_cuts": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "quiet": True, "no_warnings": True, "noprogress": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video}"])
    except Exception as exc:  # yt-dlp raises DownloadError for deleted / private videos
        return str(exc).splitlines()[0]
    return None if (clips_dir / f"{segment_id}.wav").exists() else "no WAV produced"


def download_audioset(audioset_dir, workers=4):
    """Fetch the label files, then every kept clip not already on disk."""
    audioset_dir = Path(audioset_dir)
    labels_dir, clips_dir = audioset_dir / "labels", audioset_dir / "clips"
    labels_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    for name, base in LABEL_FILES.items():
        if not (labels_dir / name).exists():
            urllib.request.urlretrieve(base + name, labels_dir / name)

    failures_file = audioset_dir / "download_failures.csv"
    failures = pd.read_csv(failures_file) if failures_file.exists() else pd.DataFrame(columns=["segment_id", "error"])
    failures = failures[failures["error"].map(is_permanent_failure)]  # others are retried
    clips = select_knock_clips(labels_dir)
    clips = clips[clips["vehicle_tag"].isin(KEEP_VEHICLE_TAGS)]
    todo = clips[~clips["segment_id"].isin(failures["segment_id"])
                 & ~clips["segment_id"].map(lambda s: (clips_dir / f"{s}.wav").exists())]
    if len(todo):
        _ffmpeg_on_path(audioset_dir / "tools")
        print(f"Downloading {len(todo)} AudioSet clips from YouTube ...", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            errors = list(pool.map(lambda c: _download_clip(c.segment_id, c.video_id, c.start_s, clips_dir),
                                   todo.itertuples(index=False)))
        new = pd.DataFrame({"segment_id": todo["segment_id"], "error": errors}).dropna()
        failures = pd.concat([failures, new], ignore_index=True)
        failures.to_csv(failures_file, index=False)
    for leftover in clips_dir.glob("*.part"):
        leftover.unlink()
    return audioset_dir
