"""Spec for the experiment-config builder. Pure stdlib + yaml -- no torch."""

import yaml
import pytest

from src.webapp import expbuilder


def _spec(**over):
    base = {"name": "demo", "model": "configs/models/jsc_2l.yaml", "dataset": "jsc"}
    base.update(over)
    return base


def test_build_minimal_has_defaults():
    cfg = expbuilder.build_exp_config(_spec())
    assert cfg["name"] == "demo"
    assert cfg["model"] == "configs/models/jsc_2l.yaml"
    assert cfg["dataset"] == "jsc"
    assert cfg["output_dir"] == "./outputs/demo"
    assert cfg["tracking_db"] == "./outputs/demo/experiments.db"
    assert cfg["batch_size"] == [128, 256]
    assert cfg["best_metric_name"] == "accuracy"


def test_build_overrides_and_int_batch():
    cfg = expbuilder.build_exp_config(_spec(epochs=5, learning_rate=0.03, batch_size=64,
                                            output_dir="./out/x"))
    assert cfg["epochs"] == 5 and cfg["learning_rate"] == 0.03
    assert cfg["batch_size"] == [64, 64]
    assert cfg["output_dir"] == "./out/x"
    assert cfg["tracking_db"] == "./out/x/experiments.db"


def test_train_path_only_when_given():
    assert "train_path" not in expbuilder.build_exp_config(_spec())
    cfg = expbuilder.build_exp_config(_spec(train_path="./tmp/datasets/cifar10"))
    assert cfg["train_path"] == "./tmp/datasets/cifar10"


@pytest.mark.parametrize("missing", ["name", "model", "dataset"])
def test_required_fields(missing):
    spec = _spec()
    spec[missing] = ""
    with pytest.raises(ValueError):
        expbuilder.build_exp_config(spec)


def test_yaml_roundtrips():
    cfg = expbuilder.build_exp_config(_spec())
    text = expbuilder.to_yaml(cfg)
    assert text.startswith("#")                    # has the header comment
    parsed = yaml.safe_load(text)
    assert parsed["name"] == "demo" and parsed["model"] == cfg["model"]


def test_save_writes_and_guards_overwrite(tmp_path):
    repo = tmp_path
    cfg = expbuilder.build_exp_config(_spec())
    res = expbuilder.save_exp_config(str(repo), cfg)
    written = repo / res["path"]
    assert written.exists() and res["overwritten"] is False
    assert yaml.safe_load(written.read_text())["name"] == "demo"

    with pytest.raises(FileExistsError):
        expbuilder.save_exp_config(str(repo), cfg)
    res2 = expbuilder.save_exp_config(str(repo), cfg, overwrite=True)
    assert res2["overwritten"] is True


def test_save_rejects_path_escape(tmp_path):
    cfg = expbuilder.build_exp_config(_spec())
    with pytest.raises(ValueError):
        expbuilder.save_exp_config(str(tmp_path), cfg, rel_path="../escape.yaml")


def test_list_datasets_nonempty():
    names = expbuilder.list_datasets()
    assert "cifar10" in names and "jsc" in names
