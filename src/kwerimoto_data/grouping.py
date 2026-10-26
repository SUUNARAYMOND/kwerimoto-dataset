"""Step 4a: link clips that share a recording, so they land in the same split.

Revix knock clips are a car recording with a knock sound laid on top. Two clips
are linked when they share the car recording, the knock sound, or any piece of
the same YouTube video (AudioSet names are "<video id>_<start ms>.wav").
Revix and our real AudioSet knock clips both come from YouTube, so a real clip
is linked to every Revix mix that used a piece of the same video.
Links chain: if A is linked to B and B to C, all three form one group.
"""
from pathlib import Path

import pandas as pd

YOUTUBE_DATASETS = {"revix", "audioset"}


def video_id(source_name):
    return Path(source_name).stem.rsplit("_", 1)[0]


def _recording_key(dataset, name):
    # Hugging Face files carry no video id, so each one is its own recording.
    if dataset in YOUTUBE_DATASETS:
        return f"youtube:{video_id(name)}"
    return f"{dataset}:{Path(name).stem}"


def assign_groups(df):
    """Return one group name per row, e.g. "youtube:DqOGNij7qIE"."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            # Keep the alphabetically first key as the root, so names are stable.
            parent[max(ra, rb)] = min(ra, rb)

    keys = []
    for row in df.itertuples(index=False):
        key = _recording_key(row.source_dataset, row.source_recording)
        find(key)
        if isinstance(row.knock_source, str) and row.knock_source:
            union(key, _recording_key(row.source_dataset, row.knock_source))
        keys.append(key)
    return pd.Series([find(k) for k in keys], index=df.index, name="group")
