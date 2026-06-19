"""Spec for experiment tracking DB (task #19). Pure sqlite -- no torch needed."""

from types import SimpleNamespace

from src.tracking import DBTrackerCallback, ExperimentDB


def test_experiment_roundtrip(tmp_path):
    db = ExperimentDB(str(tmp_path / "exp.db"))
    exp_id = db.start_experiment("run1", model_yaml="name: m", experiment_yaml="name: e")
    db.log_metric(exp_id, 1, "train", loss=0.5)
    db.log_metric(exp_id, 1, "val", loss=0.4, accuracy=0.8)

    row = db.get_experiment(exp_id)
    assert row["name"] == "run1" and "name: m" in row["model_yaml"]
    metrics = db.get_metrics(exp_id)
    assert len(metrics) == 2
    assert any(m["split"] == "val" and m["accuracy"] == 0.8 for m in metrics)


def test_weights_blob_roundtrip(tmp_path):
    db = ExperimentDB(str(tmp_path / "exp.db"))
    exp_id = db.start_experiment("run2")
    ckpt = tmp_path / "best.pth"
    ckpt.write_bytes(b"\x00\x01weights")
    db.save_weights(exp_id, "best", str(ckpt))
    assert db.load_weights(exp_id, "best") == b"\x00\x01weights"
    assert db.load_weights(exp_id, "last") is None


def test_callback_logs_and_saves(tmp_path):
    db = ExperimentDB(str(tmp_path / "exp.db"))
    (tmp_path / "best.pth").write_bytes(b"w")
    (tmp_path / "last.pth").write_bytes(b"w")
    runner = SimpleNamespace(config=SimpleNamespace(save_dir=str(tmp_path)))

    cb = DBTrackerCallback(db, "run3", model_yaml="m", experiment_yaml="e")
    cb.on_epoch_end(runner, 1, {"train_loss": 0.7, "loss": 0.6, "accuracy": 0.5})
    cb.on_train_end(runner, {})

    assert cb.exp_id is not None
    assert len(db.get_metrics(cb.exp_id)) == 2          # train + val
    assert db.load_weights(cb.exp_id, "best") == b"w"
    assert db.load_weights(cb.exp_id, "last") == b"w"
