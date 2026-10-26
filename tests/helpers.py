"""Small helpers for making test audio files."""
from pathlib import Path

import numpy as np
import soundfile as sf


def tone(seconds, sample_rate=16000, channels=1, frequency=440.0, amplitude=0.5):
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    wave = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    if channels == 1:
        return wave
    return np.stack([wave] * channels, axis=1)


def write(path, audio, sample_rate=16000, subtype="PCM_16"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype=subtype)
    return path
