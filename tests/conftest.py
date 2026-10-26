import pytest

import kwerimoto_data.pipeline as pipeline


@pytest.fixture(autouse=True)
def no_downloads(monkeypatch):
    """Tests use fake folders; fail fast if one would reach the internet."""
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to download real data; pass hf_dir / audioset_dir")

    monkeypatch.setattr(pipeline, "download_huggingface", refuse)
    monkeypatch.setattr(pipeline, "download_audioset", refuse)
