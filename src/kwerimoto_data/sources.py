"""Step 1: list the candidate files from each source dataset.

Every source becomes the same table of candidates:
record_id, label, source_dataset, original_file, source_path,
source_recording, knock_source, synthetic.
"""
from pathlib import Path

import pandas as pd
import pyarrow as pa

HF_REPO = "RashmiKarunarathne/MyDataSet"
# Pinned so everyone downloads exactly the same files.
HF_REVISION = "07bb22b36e86762d9b06e3652ea00194b1b89cec"
HF_LABELS = {"healthy": "healthy", "misfire": "misfire"}  # tappet_issue and belt_issue are left out

REVIX_SPLITS = ["train", "validation"]
REVIX_LABELS = {0: "healthy", 1: "engine_knock"}
REVIX_CAR = 0


def _read_arrow(path):
    with pa.memory_map(str(path)) as f:
        return pa.ipc.open_stream(f).read_all()


def read_revix_cars(revix_root):
    """Car clips from Revix's dataset (both of their splits); trucks are left out."""
    revix_root = Path(revix_root)
    rows = []
    for split in REVIX_SPLITS:
        table = _read_arrow(revix_root / "engine_sounds_hf_dataset" / split / "data-00000-of-00001.arrow")
        rows += table.select([c for c in table.column_names if c != "audio"]).to_pylist()

    candidates = []
    for row in rows:
        if row["vehicle_type"] != REVIX_CAR:
            continue
        audio_path = Path(row["audio_path"])
        label = REVIX_LABELS[row["label"]]
        folder_label = "engine_knock" if audio_path.parent.name.endswith("knocking") else "healthy"
        if label != folder_label:
            raise ValueError(f"{audio_path.name}: metadata label {label!r} disagrees with folder "
                             f"{audio_path.parent.name!r}")
        candidates.append({
            "record_id": f"revix_{audio_path.stem}",
            "label": label,
            "source_dataset": "revix",
            "original_file": audio_path.name,
            "source_path": str(revix_root / audio_path),
            "source_recording": row["source_file"],
            "knock_source": row["knocking_source"] or "",
            "synthetic": bool(row["synthetic"]),
        })
    return pd.DataFrame(candidates)


def download_huggingface(dest_dir):
    """Download only the healthy and misfire files, at the pinned revision."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        revision=HF_REVISION,
        allow_patterns=[f"MyDataSet/{prefix}_*.wav" for prefix in HF_LABELS] + ["README.md"],
        local_dir=str(dest_dir),
    )
    return Path(dest_dir)


def read_huggingface(hf_dir):
    """Healthy and misfire recordings from the Hugging Face download."""
    candidates = []
    for path in sorted(Path(hf_dir, "MyDataSet").glob("*.wav")):
        prefix = path.stem.rsplit("_", 1)[0]
        if prefix not in HF_LABELS:
            continue
        candidates.append({
            "record_id": f"hf_{path.stem}",
            "label": HF_LABELS[prefix],
            "source_dataset": "huggingface",
            "original_file": path.name,
            "source_path": str(path),
            "source_recording": path.name,
            "knock_source": "",
            "synthetic": False,
        })
    return pd.DataFrame(candidates)
