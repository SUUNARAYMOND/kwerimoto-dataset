from pathlib import Path

from huggingface_hub import metadata_load

import kwerimoto_data.publish as publish

CARD = Path(__file__).parents[1] / "dataset_card.md"


class FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create_repo", repo_id, kwargs))

    def upload_file(self, **kwargs):
        self.calls.append(("upload_file", kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))


def test_uploads_the_card_and_prepared_clips_but_not_raw_files(tmp_path, monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(publish, "HfApi", lambda: api)
    card = tmp_path / "card.md"
    card.write_text("---\n---\n")

    publish.upload("me/sounds", data_dir=tmp_path / "data", card=card)

    (_, repo_id, repo), (_, card_upload), (_, data_upload) = api.calls
    assert repo_id == "me/sounds"
    assert repo == {"repo_type": "dataset", "private": False, "exist_ok": True}
    assert card_upload["path_or_fileobj"] == str(card)
    assert card_upload["path_in_repo"] == "README.md"
    assert data_upload["folder_path"] == str(tmp_path / "data")
    assert data_upload["allow_patterns"] == ["manifest.csv", "prepared/**"]
    assert data_upload["delete_patterns"] == ["prepared/**"]  # clips gone from a rebuild are removed too
    assert card_upload["repo_type"] == data_upload["repo_type"] == "dataset"


def test_dataset_card_points_each_split_at_its_folder():
    configs = metadata_load(CARD)["configs"]
    data_files = {f["split"]: f["path"] for f in configs[0]["data_files"]}
    assert data_files == {s: f"prepared/{s}/**" for s in ["train", "validation", "test", "real_test"]}
