import pytest

from fake_sources import CAR, ENGINE, MOTORCYCLE, make_audioset
from kwerimoto_data.audioset import is_permanent_failure, read_audioset, select_knock_clips


@pytest.mark.parametrize("error", [
    "ERROR: [youtube] 1D2q6StYdtY: This video is unavailable",
    "ERROR: [youtube] FGAcC29fUHo: This video is not available",
    "ERROR: [youtube] 3nlTDoIP2Oo: Private video. If the owner of this video has granted you access, please sign in.",
    "ERROR: [youtube] BWkVK480uNI: Video unavailable. This video is no longer available because the YouTube "
    "account associated with this video has been terminated.",
])
def test_gone_videos_are_permanent_failures(error):
    assert is_permanent_failure(error)


@pytest.mark.parametrize("error", [
    "ERROR: [youtube] sBHtYqJ0ArE: Unable to download API page: [Errno 11001] getaddrinfo failed",
    "ERROR: ffmpeg exited with code 3436169992",
    "no WAV produced",
])
def test_network_and_ffmpeg_errors_are_retried(error):
    assert not is_permanent_failure(error)


def test_vehicle_tag_comes_from_strong_and_clip_level_labels(tmp_path):
    root = make_audioset(tmp_path / "as", {
        "carstrong_10000": ("train", [CAR], []),
        "carweak_0": ("train", [], [CAR]),
        "moto_0": ("train", [MOTORCYCLE], []),
        "both_0": ("train", [CAR], [MOTORCYCLE]),
        "plain_0": ("eval", [ENGINE], []),
    })

    tags = select_knock_clips(root / "labels").set_index("segment_id")["vehicle_tag"].to_dict()

    assert tags == {"carstrong_10000": "car", "carweak_0": "car", "moto_0": "other",
                    "both_0": "car+other", "plain_0": "untagged"}


def test_select_knock_clips_records_google_set_and_start_time(tmp_path):
    root = make_audioset(tmp_path / "as", {"-Nc_T1FvPx0_70000": ("eval", [], [])})
    row = select_knock_clips(root / "labels").iloc[0]
    assert row["google_set"] == "eval"
    assert row["video_id"] == "-Nc_T1FvPx0"
    assert row["start_s"] == 70.0


def test_read_audioset_keeps_car_and_untagged_clips_that_downloaded(tmp_path):
    root = make_audioset(tmp_path / "as", {
        "car_0": ("train", [CAR], []),
        "plain_0": ("eval", [], []),
        "moto_0": ("train", [MOTORCYCLE], []),
        "gone_0": ("train", [CAR], []),
    }, downloaded=["car_0", "plain_0", "moto_0"])

    df = read_audioset(root).set_index("record_id")

    assert sorted(df.index) == ["audioset_car_0", "audioset_plain_0"]
    row = df.loc["audioset_plain_0"]
    assert row["label"] == "engine_knock"
    assert row["source_dataset"] == "audioset"
    assert row["original_file"] == "plain_0.wav"
    assert row["source_recording"] == "plain_0.wav"
    assert row["knock_source"] == ""
    assert not row["synthetic"]
    assert row["google_set"] == "eval"
    assert row["vehicle_tag"] == "untagged"
    assert row["source_path"].endswith("plain_0.wav")
