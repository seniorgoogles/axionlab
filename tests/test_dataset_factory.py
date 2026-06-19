"""Spec for the dataset factory (tasks #13, #16): name registration, unknown
handling, injectable transform passthrough, and .zip extraction."""

import zipfile

import pytest

from src.datasets.base import BaseDataset
from src.datasets import factory


class _Dummy(BaseDataset):
    def _do_preprocessing(self, **kwargs):
        # record what was injected so the test can assert it
        self.seen_transform = self.transform
        return [(0, 0)], [(1, 1)]


def test_register_and_build():
    factory.register("dummy", _Dummy)
    assert "dummy" in factory.list_datasets()
    ds = factory.build_dataset("dummy", {"batch_size": [2, 2]})
    assert isinstance(ds, _Dummy)


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        factory.build_dataset("does-not-exist", {})


def test_transform_is_injected():
    factory.register("dummy2", _Dummy)
    sentinel = object()
    ds = factory.build_dataset("dummy2", {"batch_size": [2, 2]}, transform=sentinel)
    assert ds.seen_transform is sentinel


def test_maybe_extract_zip(tmp_path):
    torchvision = pytest.importorskip("torchvision")  # vision module needs it
    from src.datasets.vision import _maybe_extract

    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("class_a/img.txt", "x")
    out = _maybe_extract(str(archive))
    assert out == str(tmp_path / "data")
    assert (tmp_path / "data" / "class_a" / "img.txt").exists()
