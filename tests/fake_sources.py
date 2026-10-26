"""Tiny stand-ins for the Revix folder and the Hugging Face download."""
import numpy as np
import pyarrow as pa

from helpers import tone, write

REVIX_SCHEMA = pa.schema([
    ("audio_path", pa.string()), ("label", pa.int64()), ("vehicle_type", pa.int64()),
    ("has_knocking", pa.bool_()), ("source_file", pa.string()), ("knocking_source", pa.string()),
    ("duration", pa.float32()), ("intensity", pa.float32()), ("synthetic", pa.bool_()),
])


def revix_row(category, number, source, knock_source=""):
    knock = category.endswith("knocking")
    return {
        "audio_path": f"engine_sounds_hf_dataset/audio_files/{category}/{category}_{number:04d}.wav",
        "label": int(knock),
        "vehicle_type": 0 if category.startswith("car") else 1,
        "has_knocking": knock,
        "source_file": source,
        "knocking_source": knock_source,
        "duration": 2.0,
        "intensity": 0.8 if knock else 0.0,
        "synthetic": knock,
    }


def make_revix(root, train_rows, validation_rows=(), skip_audio=(), audio=None):
    """Write Revix-style Arrow metadata and one WAV per row under `root`."""
    for split, rows in [("train", train_rows), ("validation", validation_rows)]:
        folder = root / "engine_sounds_hf_dataset" / split
        folder.mkdir(parents=True)
        table = pa.Table.from_pylist(list(rows), schema=REVIX_SCHEMA)
        with pa.OSFile(str(folder / "data-00000-of-00001.arrow"), "wb") as sink:
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
        for i, row in enumerate(rows):
            name = row["audio_path"].rsplit("/", 1)[1]
            if name not in skip_audio:
                clip = (audio or {}).get(name, tone(2.0, frequency=200 + 7 * i + 300 * (split == "validation")))
                write(root / row["audio_path"], clip)
    return root


def make_huggingface(root, counts):
    """counts: {"healthy": 2, "misfire": 3, "tappet_issue": 1} -> MyDataSet/<name>_001.wav ..."""
    folder = root / "MyDataSet"
    for k, (prefix, n) in enumerate(counts.items()):
        for i in range(1, n + 1):
            stereo = tone(3.0, 44100, channels=2, frequency=500 + 11 * i + 150 * k)
            write(folder / f"{prefix}_{i:03d}.wav", stereo, 44100)
    return root


def silence(seconds=2.0):
    return np.zeros(int(seconds * 16000), dtype=np.float32)


KNOCK, CAR, MOTORCYCLE, ENGINE = "/m/01h82_", "/m/0k4j", "/m/04_sv", "/m/02mk9"


def make_audioset(root, clips, downloaded=None):
    """clips: {segment_id: (google_set, [extra strong MIDs], [clip-level MIDs])}.

    Writes AudioSet-style label files under root/labels and a WAV under
    root/clips for each segment in `downloaded` (default: all of them).
    """
    labels = root / "labels"
    labels.mkdir(parents=True)
    (labels / "mid_to_display_name.tsv").write_text(
        f"{KNOCK}\tEngine knocking\n{CAR}\tCar\n{MOTORCYCLE}\tMotorcycle\n{ENGINE}\tEngine\n")
    (labels / "class_labels_indices.csv").write_text(
        f'index,mid,display_name\n0,{CAR},Car\n1,{MOTORCYCLE},Motorcycle\n2,{KNOCK},"Engine knocking"\n')
    for google_set in ["train", "eval"]:
        lines = ["segment_id\tstart_time_seconds\tend_time_seconds\tlabel"]
        for seg, (where, strong, _) in clips.items():
            if where == google_set:
                lines.append(f"{seg}\t0.000\t10.000\t{KNOCK}")
                lines += [f"{seg}\t1.000\t2.000\t{mid}" for mid in strong]
        (labels / f"audioset_{google_set}_strong.tsv").write_text("\n".join(lines) + "\n")
    weak = {"eval": [], "train": []}
    for seg, (where, _, clip_level) in clips.items():
        video, ms = seg.rsplit("_", 1)
        weak[where].append(f'{video}, {int(ms) / 1000:.3f}, {int(ms) / 1000 + 10:.3f}, "{",".join([KNOCK] + clip_level)}"')
    header = "# Segments csv\n# num_ytids=0\n# YTID, start_seconds, end_seconds, positive_labels\n"
    (labels / "eval_segments.csv").write_text(header + "\n".join(weak["eval"]) + "\n")
    (labels / "balanced_train_segments.csv").write_text(header)
    (labels / "unbalanced_train_segments.csv").write_text(header + "\n".join(weak["train"]) + "\n")
    for i, seg in enumerate(clips if downloaded is None else downloaded):
        write(root / "clips" / f"{seg}.wav", tone(10.0, 48000, channels=2, frequency=900 + 13 * i), 48000)
    return root
