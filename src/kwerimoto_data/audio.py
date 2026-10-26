"""Step 5: turn a raw file into a prepared clip, plus optional windowing."""
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SAMPLE_RATE = 16000
MAX_SECONDS = 10.0


def prepare_audio(path, record_id, sample_rate=SAMPLE_RATE, max_seconds=MAX_SECONDS, seed=42):
    """Mono, `sample_rate` Hz, at most `max_seconds` long (following Revix).

    A longer recording is cut to one random piece. The piece's position
    depends only on `seed` and `record_id`, so reruns give the same clip.
    Shorter recordings are left as they are (no padding).
    """
    data, original_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = data.mean(axis=1)
    if original_rate != sample_rate:
        audio = soxr.resample(audio, original_rate, sample_rate)
    audio = audio.astype(np.float32)

    max_len = int(max_seconds * sample_rate)
    if len(audio) > max_len:
        rng = np.random.default_rng([seed, zlib.crc32(record_id.encode())])
        start = int(rng.integers(0, len(audio) - max_len + 1))
        audio = audio[start:start + max_len]
    return audio


def write_wav(path, audio, sample_rate=SAMPLE_RATE):
    """Save as 16-bit WAV; samples beyond full scale are clipped."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -1.0, 1.0), sample_rate, subtype="PCM_16")


def cut_windows(audio, window_seconds, hop_seconds=None, sample_rate=SAMPLE_RATE, pad=True):
    """Cut a clip into fixed-length windows for training experiments.

    `hop_seconds` defaults to `window_seconds` (no overlap). With `pad`, the
    last short piece is padded with silence; without it, the piece is dropped.
    """
    size = int(window_seconds * sample_rate)
    hop = int((hop_seconds or window_seconds) * sample_rate)
    windows = []
    covered = 0  # end of the audio already inside a full window
    for start in range(0, len(audio), hop):
        piece = audio[start:start + size]
        if len(piece) < size:
            # Only pad a piece that holds audio no earlier window has.
            if pad and len(audio) > covered:
                windows.append(np.pad(piece, (0, size - len(piece))))
            break
        windows.append(piece)
        covered = start + size
    return windows
