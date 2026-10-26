import numpy as np
import soundfile as sf

from helpers import tone, write
from kwerimoto_data.audio import cut_windows, prepare_audio, write_wav


def test_stereo_44k_becomes_mono_16k(tmp_path):
    path = write(tmp_path / "s.wav", tone(2.0, 44100, channels=2), 44100)
    audio = prepare_audio(path, record_id="s")
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert len(audio) == 2 * 16000


def test_mono_is_the_average_of_the_channels(tmp_path):
    left = tone(1.0, amplitude=0.4)
    stereo = np.stack([left, np.zeros_like(left)], axis=1)
    path = write(tmp_path / "s.wav", stereo, subtype="FLOAT")
    audio = prepare_audio(path, record_id="s")
    np.testing.assert_allclose(audio, left / 2, atol=1e-6)


def test_32_bit_float_input_is_accepted(tmp_path):
    path = write(tmp_path / "f.wav", tone(1.0, 48000), 48000, subtype="FLOAT")
    assert len(prepare_audio(path, record_id="f")) == 16000


def test_recordings_up_to_10_seconds_are_kept_whole(tmp_path):
    path = write(tmp_path / "ten.wav", tone(10.0))
    original, _ = sf.read(path, dtype="float32")
    np.testing.assert_array_equal(prepare_audio(path, record_id="ten"), original)


def test_short_recordings_are_not_padded(tmp_path):
    path = write(tmp_path / "short.wav", tone(3.2))
    assert len(prepare_audio(path, record_id="short")) == int(3.2 * 16000)


def test_longer_recordings_are_cut_to_one_10_second_piece(tmp_path):
    # A ramp makes every sample unique, so we can find where the piece came from.
    ramp = np.linspace(-0.9, 0.9, 25 * 16000, dtype=np.float32)
    path = write(tmp_path / "long.wav", ramp, subtype="FLOAT")
    audio = prepare_audio(path, record_id="long")
    assert len(audio) == 10 * 16000
    start = int(np.argmin(np.abs(ramp - audio[0])))
    np.testing.assert_array_equal(audio, ramp[start:start + 10 * 16000])


def test_cut_is_repeatable_for_the_same_file_and_seed(tmp_path):
    path = write(tmp_path / "long.wav", np.linspace(-0.9, 0.9, 25 * 16000, dtype=np.float32), subtype="FLOAT")
    first = prepare_audio(path, record_id="long", seed=42)
    np.testing.assert_array_equal(first, prepare_audio(path, record_id="long", seed=42))
    assert not np.array_equal(first, prepare_audio(path, record_id="long", seed=43))


def test_write_wav_saves_16_bit_and_clips_overloud_samples(tmp_path):
    out = tmp_path / "out" / "x.wav"
    write_wav(out, np.array([0.0, 0.5, 1.7, -2.0], dtype=np.float32))
    info = sf.info(out)
    assert info.subtype == "PCM_16" and info.samplerate == 16000 and info.channels == 1
    data, _ = sf.read(out, dtype="float32")
    assert data.max() <= 1.0 and data.min() >= -1.0
    assert data[2] > 0.99 and data[3] == -1.0


def test_cut_windows_splits_and_pads_the_last_piece():
    audio = np.ones(int(7.5 * 16000), dtype=np.float32)
    windows = cut_windows(audio, window_seconds=3.0)
    assert [len(w) for w in windows] == [48000, 48000, 48000]
    assert windows[2][int(1.5 * 16000):].sum() == 0


def test_cut_windows_with_overlap():
    audio = np.arange(6 * 16000, dtype=np.float32)
    windows = cut_windows(audio, window_seconds=4.0, hop_seconds=2.0)
    assert [w[0] for w in windows] == [0, 32000]


def test_cut_windows_can_drop_the_short_last_piece():
    audio = np.ones(int(7.5 * 16000), dtype=np.float32)
    assert len(cut_windows(audio, window_seconds=3.0, pad=False)) == 2


def test_cut_windows_pads_a_recording_shorter_than_one_window():
    audio = np.ones(16000, dtype=np.float32)
    windows = cut_windows(audio, window_seconds=3.0)
    assert len(windows) == 1 and len(windows[0]) == 48000
