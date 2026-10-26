import pandas as pd
import pyarrow.json as paj
import soundfile as sf

from fake_sources import CAR, MOTORCYCLE, make_audioset, make_huggingface, make_revix, revix_row, silence
from helpers import tone
from kwerimoto_data.pipeline import build_dataset

AUDIOSET_CLIPS = {
    "REAL1_0": ("train", [CAR], []),
    "REAL2_0": ("train", [], []),
    "MOTO_0": ("train", [MOTORCYCLE], []),
    "EVAL1_0": ("eval", [], []),
}


def build(tmp_path, train_rows, validation_rows=(), skip_audio=(), audio=None, audioset_clips=AUDIOSET_CLIPS):
    revix = make_revix(tmp_path / "revix", train_rows, validation_rows, skip_audio, audio)
    hf = make_huggingface(tmp_path / "hf", {"healthy": 2, "misfire": 3, "tappet_issue": 1})
    audioset = make_audioset(tmp_path / "audioset", audioset_clips)
    out = tmp_path / "out"
    manifest = build_dataset(revix_root=revix, out_dir=out, hf_dir=hf, audioset_dir=audioset, seed=42)
    return out, manifest


def test_builds_manifest_and_prepared_audio(tmp_path):
    out, manifest = build(tmp_path, [
        revix_row("car_clean", 1, "AAA_0.wav"),
        revix_row("car_knocking", 1, "BBB_0.wav", "KKK_0.wav"),
        revix_row("truck_clean", 1, "TTT_0.wav"),
    ])

    saved = pd.read_csv(out / "manifest.csv", keep_default_na=False)
    assert len(saved) == len(manifest) == 2 + 5 + 3
    assert saved["label"].value_counts().to_dict() == {"engine_knock": 4, "misfire": 3, "healthy": 3}
    assert set(saved["split"]) <= {"train", "validation", "test", "real_test"}
    for row in saved.itertuples():
        info = sf.info(out / row.file_name)
        assert info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16"
        assert row.file_name.startswith(f"prepared/{row.split}/{row.label}/")


def test_manifest_has_the_agreed_columns(tmp_path):
    out, _ = build(tmp_path, [revix_row("car_clean", 1, "AAA_0.wav")])
    columns = list(pd.read_csv(out / "manifest.csv").columns)
    assert columns == ["file_name", "label", "split", "group", "source_dataset", "original_file",
                       "source_recording", "knock_source", "synthetic", "vehicle_tag", "duration_s",
                       "original_sample_rate", "original_channels", "original_duration_s", "sha256"]


def test_each_split_folder_lists_its_own_clips(tmp_path):
    # Hugging Face reads prepared/<split>/metadata.jsonl to build each split
    out, manifest = build(tmp_path, [revix_row("car_clean", i, f"V{i:03d}_0.wav") for i in range(1, 12)])
    for split, part in manifest.groupby("split"):
        metadata = pd.read_json(out / "prepared" / split / "metadata.jsonl", lines=True)
        assert list(metadata.columns) == list(manifest.columns)
        assert sorted(f"prepared/{split}/" + metadata["file_name"]) == sorted(part["file_name"])
        assert (out / "prepared" / split / metadata["file_name"].iloc[0]).exists()


def test_split_metadata_files_agree_on_column_types(tmp_path):
    # Hugging Face refuses splits whose metadata columns differ in type; real_test has
    # no synthetic clips, so its knock_source column is empty but must still be text.
    rows = [revix_row("car_knocking", i, f"V{i:03d}_0.wav", f"K{i:03d}_0.wav") for i in range(1, 12)]
    out, manifest = build(tmp_path, rows)
    assert "real_test" in set(manifest["split"])
    schemas = {split: paj.read_json(out / "prepared" / split / "metadata.jsonl").schema
               for split in manifest["split"].unique()}
    assert len({str(schema) for schema in schemas.values()}) == 1
    assert str(schemas["real_test"].field("knock_source").type) == "string"


def test_real_knock_clips_are_added_as_non_synthetic_knock(tmp_path):
    _, manifest = build(tmp_path, [revix_row("car_clean", 1, "AAA_0.wav")])
    real = manifest[manifest["source_dataset"] == "audioset"].set_index("original_file")
    assert sorted(real.index) == ["EVAL1_0.wav", "REAL1_0.wav", "REAL2_0.wav"]  # motorcycle left out
    assert (real["label"] == "engine_knock").all()
    assert not real["synthetic"].any()
    assert real.loc["REAL1_0.wav", "vehicle_tag"] == "car"
    assert real.loc["REAL2_0.wav", "vehicle_tag"] == "untagged"


def test_google_evaluation_clips_form_the_real_test_split(tmp_path):
    _, manifest = build(tmp_path, [revix_row("car_clean", 1, "AAA_0.wav")])
    real_test = manifest[manifest["split"] == "real_test"]
    assert list(real_test["original_file"]) == ["EVAL1_0.wav"]
    others = manifest[manifest["original_file"].isin(["REAL1_0.wav", "REAL2_0.wav"])]
    assert set(others["split"]) <= {"train", "validation", "test"}


def test_real_clip_stays_with_the_revix_mixes_that_used_its_video(tmp_path):
    rows = [revix_row("car_knocking", i, f"V{i:03d}_0.wav", "REAL1_5000.wav") for i in range(1, 12)]
    rows += [revix_row("car_clean", i, f"C{i:03d}_0.wav") for i in range(1, 12)]
    _, manifest = build(tmp_path, rows)
    linked = manifest[manifest["original_file"].str.startswith("car_knocking") | (manifest["original_file"] == "REAL1_0.wav")]
    assert linked["group"].nunique() == 1
    assert linked["split"].nunique() == 1


def test_original_files_are_kept_untouched_in_raw(tmp_path):
    out, _ = build(tmp_path, [revix_row("car_clean", 1, "AAA_0.wav")])
    original = tmp_path / "revix/engine_sounds_hf_dataset/audio_files/car_clean/car_clean_0001.wav"
    copy = out / "raw/revix/car_clean/car_clean_0001.wav"
    assert copy.read_bytes() == original.read_bytes()


def test_removed_files_are_left_out_of_the_manifest(tmp_path):
    same = tone(2.0, frequency=321)
    out, manifest = build(tmp_path, [
        revix_row("car_clean", 1, "AAA_0.wav"),
        revix_row("car_clean", 2, "BBB_0.wav"),
        revix_row("car_clean", 3, "CCC_0.wav"),
        revix_row("car_knocking", 29, "DDD_0.wav", "KKK_0.wav"),
    ], skip_audio={"car_knocking_0029.wav"},
        audio={"car_clean_0002.wav": silence(), "car_clean_0001.wav": same, "car_clean_0003.wav": same})

    revix_files = set(manifest.loc[manifest["source_dataset"] == "revix", "original_file"])
    assert revix_files == {"car_clean_0001.wav"}
    assert not list(out.glob("prepared/**/revix_car_clean_0002.wav"))


def test_linked_clips_share_a_split(tmp_path):
    rows = [revix_row("car_clean", i, f"V{i:03d}_0.wav") for i in range(1, 30)]
    rows += [revix_row("car_knocking", i, f"V{i:03d}_0.wav", "KKK_0.wav") for i in range(1, 30)]
    _, manifest = build(tmp_path, rows)
    revix = manifest[manifest["source_dataset"] == "revix"]
    assert revix["group"].nunique() == 1
    assert revix["split"].nunique() == 1


def test_rebuilding_gives_identical_results(tmp_path):
    rows = [revix_row("car_clean", i, f"V{i:03d}_0.wav") for i in range(1, 12)]
    out, first = build(tmp_path, rows)
    second = build_dataset(revix_root=tmp_path / "revix", out_dir=out, hf_dir=tmp_path / "hf",
                           audioset_dir=tmp_path / "audioset", seed=42)
    pd.testing.assert_frame_equal(first, second)
