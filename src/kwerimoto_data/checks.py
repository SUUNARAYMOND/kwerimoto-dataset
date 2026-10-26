"""Step 2: find files we cannot use (missing, unreadable, silent, duplicate)."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_audio(path):
    """Open one file and report its format, and whether it is usable."""
    try:
        data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile raises several error types for bad files
        return {"readable": False, "error": str(exc)}
    if data.shape[0] == 0:
        return {"readable": False, "error": "file has no audio samples"}
    return {
        "readable": True,
        "error": None,
        "sample_rate": sample_rate,
        "channels": data.shape[1],
        "duration_s": data.shape[0] / sample_rate,
        # Silent means every sample is exactly zero; quiet recordings are kept.
        "silent": not np.any(data),
    }


def run_checks(candidates):
    """Split candidate files into the ones we keep and the ones we remove.

    Each removed row gets a `reason` (missing, unreadable, silent, duplicate,
    duplicate_conflicting_labels) and a `detail`.
    """
    kept, removed = [], []
    for row in candidates.to_dict("records"):
        path = Path(row["raw_path"])
        if not path.exists():
            removed.append({**row, "reason": "missing", "detail": "file not found"})
            continue
        info = inspect_audio(path)
        if not info["readable"]:
            removed.append({**row, "reason": "unreadable", "detail": info["error"]})
        elif info["silent"]:
            removed.append({**row, "reason": "silent", "detail": "every sample is zero"})
        else:
            kept.append({
                **row,
                "sha256": file_sha256(path),
                "original_sample_rate": info["sample_rate"],
                "original_channels": info["channels"],
                "original_duration_s": info["duration_s"],
            })

    unique = []
    by_hash = {}
    for row in kept:
        by_hash.setdefault(row["sha256"], []).append(row)
    for copies in by_hash.values():
        copies.sort(key=lambda r: r["record_id"])
        if len({r["label"] for r in copies}) > 1:
            names = ", ".join(r["record_id"] for r in copies)
            removed += [{**r, "reason": "duplicate_conflicting_labels",
                         "detail": f"identical files with different labels: {names}"} for r in copies]
            continue
        unique.append(copies[0])
        removed += [{**r, "reason": "duplicate", "detail": f"identical to {copies[0]['record_id']}"}
                    for r in copies[1:]]

    order = {rid: i for i, rid in enumerate(candidates["record_id"])}
    unique.sort(key=lambda r: order[r["record_id"]])
    removed.sort(key=lambda r: order[r["record_id"]])
    kept_df = pd.DataFrame(unique, columns=list(candidates.columns) + [
        "sha256", "original_sample_rate", "original_channels", "original_duration_s"])
    removed_df = pd.DataFrame(removed, columns=list(candidates.columns) + ["reason", "detail"])
    return kept_df.reset_index(drop=True), removed_df.reset_index(drop=True)
