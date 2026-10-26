import numpy as np
import pandas as pd

from helpers import tone, write
from kwerimoto_data.checks import inspect_audio, run_checks


def test_inspect_audio_reads_format_details(tmp_path):
    path = write(tmp_path / "a.wav", tone(2.0, 44100, channels=2), 44100)
    info = inspect_audio(path)
    assert info["readable"] and not info["silent"]
    assert info["sample_rate"] == 44100
    assert info["channels"] == 2
    assert info["duration_s"] == 2.0


def test_inspect_audio_flags_all_zero_audio_as_silent(tmp_path):
    path = write(tmp_path / "quiet.wav", np.zeros(16000, dtype=np.float32))
    assert inspect_audio(path)["silent"]


def test_inspect_audio_keeps_very_quiet_audio(tmp_path):
    path = write(tmp_path / "faint.wav", tone(1.0, amplitude=0.001))
    assert not inspect_audio(path)["silent"]


def test_inspect_audio_flags_broken_file(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"not really audio")
    info = inspect_audio(path)
    assert not info["readable"]
    assert info["error"]


def candidates(tmp_path, specs):
    """specs: list of (record_id, label, audio or None for a missing file)."""
    rows = []
    for record_id, label, audio in specs:
        path = tmp_path / f"{record_id}.wav"
        if audio is not None:
            write(path, audio)
        rows.append({"record_id": record_id, "label": label, "raw_path": str(path)})
    return pd.DataFrame(rows)


def test_run_checks_removes_each_problem_with_its_reason(tmp_path):
    broken = tmp_path / "broken.wav"
    df = candidates(tmp_path, [
        ("good", "healthy", tone(1.0)),
        ("gone", "healthy", None),
        ("quiet", "healthy", np.zeros(16000, dtype=np.float32)),
    ])
    broken.write_bytes(b"junk")
    df.loc[len(df)] = {"record_id": "broken", "label": "misfire", "raw_path": str(broken)}

    kept, removed = run_checks(df)

    assert list(kept["record_id"]) == ["good"]
    assert dict(zip(removed["record_id"], removed["reason"])) == {
        "gone": "missing", "quiet": "silent", "broken": "unreadable"}


def test_run_checks_keeps_one_copy_of_identical_files(tmp_path):
    same = tone(1.0)
    df = candidates(tmp_path, [
        ("b_copy", "healthy", same),
        ("a_first", "healthy", same),
        ("other", "healthy", tone(1.0, frequency=300)),
    ])

    kept, removed = run_checks(df)

    assert sorted(kept["record_id"]) == ["a_first", "other"]
    row = removed.set_index("record_id").loc["b_copy"]
    assert row["reason"] == "duplicate"
    assert "a_first" in row["detail"]


def test_run_checks_drops_identical_files_with_different_labels(tmp_path):
    same = tone(1.0)
    df = candidates(tmp_path, [
        ("x", "healthy", same),
        ("y", "engine_knock", same),
    ])

    kept, removed = run_checks(df)

    assert kept.empty
    assert set(removed["reason"]) == {"duplicate_conflicting_labels"}


def test_run_checks_records_original_format_for_kept_files(tmp_path):
    df = candidates(tmp_path, [("good", "healthy", tone(1.5))])
    kept, _ = run_checks(df)
    row = kept.iloc[0]
    assert row["original_sample_rate"] == 16000
    assert row["original_channels"] == 1
    assert row["original_duration_s"] == 1.5
    assert len(row["sha256"]) == 64
