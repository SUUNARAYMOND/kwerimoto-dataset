import pytest

from fake_sources import make_huggingface, make_revix, revix_row
from kwerimoto_data.sources import read_huggingface, read_revix_cars


def test_revix_keeps_only_car_clips_with_our_labels(tmp_path):
    root = make_revix(tmp_path / "revix", [
        revix_row("car_clean", 1, "AAA_0.wav"),
        revix_row("car_knocking", 1, "BBB_0.wav", "KKK_0.wav"),
        revix_row("truck_clean", 1, "TTT_0.wav"),
    ])

    df = read_revix_cars(root).set_index("record_id")

    assert list(df.index) == ["revix_car_clean_0001", "revix_car_knocking_0001"]
    assert df.loc["revix_car_clean_0001", "label"] == "healthy"
    assert df.loc["revix_car_knocking_0001", "label"] == "engine_knock"


def test_revix_carries_source_information_across(tmp_path):
    root = make_revix(tmp_path / "revix", [revix_row("car_knocking", 7, "BBB_30000.wav", "KKK_1000.wav")])

    row = read_revix_cars(root).iloc[0]

    assert row["source_dataset"] == "revix"
    assert row["original_file"] == "car_knocking_0007.wav"
    assert row["source_recording"] == "BBB_30000.wav"
    assert row["knock_source"] == "KKK_1000.wav"
    assert row["synthetic"] is True or row["synthetic"] == True  # noqa: E712 (numpy bool)
    assert row["source_path"].endswith("car_knocking_0007.wav")


def test_revix_reads_both_of_its_splits(tmp_path):
    root = make_revix(tmp_path / "revix",
                      [revix_row("car_clean", 1, "AAA_0.wav")],
                      [revix_row("car_clean", 2, "BBB_0.wav")])
    assert len(read_revix_cars(root)) == 2


def test_revix_lists_a_clip_even_when_its_file_is_missing(tmp_path):
    # The checks step reports it as missing; reading must not hide it.
    root = make_revix(tmp_path / "revix", [revix_row("car_knocking", 29, "AAA_0.wav", "KKK_0.wav")],
                      skip_audio={"car_knocking_0029.wav"})
    assert len(read_revix_cars(root)) == 1


def test_revix_refuses_a_label_that_disagrees_with_its_folder(tmp_path):
    bad = revix_row("car_clean", 1, "AAA_0.wav")
    bad["label"] = 1
    root = make_revix(tmp_path / "revix", [bad])
    with pytest.raises(ValueError, match="car_clean_0001"):
        read_revix_cars(root)


def test_huggingface_keeps_healthy_and_misfire_only(tmp_path):
    root = make_huggingface(tmp_path / "hf", {"healthy": 2, "misfire": 3, "tappet_issue": 1, "belt_issue": 1})

    df = read_huggingface(root).set_index("record_id")

    assert sorted(df.index) == ["hf_healthy_001", "hf_healthy_002",
                                "hf_misfire_001", "hf_misfire_002", "hf_misfire_003"]
    assert df.loc["hf_healthy_001", "label"] == "healthy"
    assert df.loc["hf_misfire_002", "label"] == "misfire"
    assert df.loc["hf_misfire_002", "source_dataset"] == "huggingface"
    assert df.loc["hf_misfire_002", "source_recording"] == "misfire_002.wav"
    assert df.loc["hf_misfire_002", "knock_source"] == ""
    assert not df["synthetic"].any()
