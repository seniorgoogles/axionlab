"""Runner: one class for train / validate / evaluate.

Ties together the model builder, the dataset factory and the CheckpointManager.
Writes best.pth + last.pth every run. Built for classification (CrossEntropy +
top-1 accuracy); pass a custom `criterion` for other tasks. Detection (YOLO)
needs its own loss/metrics -> plug them in via `criterion` and a custom metric.

    from src.core.runner import Runner
    runner = Runner.from_experiment("configs/experiments/imagenet/resnet18.yaml")
    runner.train()
    print(runner.evaluate())            # loads best.pth, runs the test loader
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from src.training.checkpoint_manager import CheckpointManager
from src.training.trainer_config import TrainerConfig
from src.utils.device_selector import DeviceSelector


def _auto_device(pref: Optional[str]) -> torch.device:
    """Pick CUDA (NVIDIA), ROCm (AMD, via cuda path), MPS (Apple) or CPU."""
    return DeviceSelector.get_device(prefer=pref, enable_mps=True)


@dataclass
class Phase:
    """One stage of a multi-phase training plan.

    Lets you express recipes like "10 epochs @ lr1, 15 epochs @ lr2, then
    quantize and fine-tune": each phase runs `epochs` epochs, optionally with its
    own `lr`, and an optional `on_start(runner)` hook that mutates the model
    (e.g. quantize / freeze layers) BEFORE the phase. The optimizer is rebuilt at
    every phase start, so a changed lr and changed parameters both take effect.
    """
    name: str
    epochs: int
    lr: Optional[float] = None
    on_start: Optional[Callable[["Runner"], None]] = None
    callbacks: list = field(default_factory=list)  # callbacks active only in this phase


def build_network(config_file: str):
    """Build a model from a net yaml, picking the graph builder for YOLO-style
    configs (have `head:` or `scales:`) and the sequential builder otherwise."""
    from src.config.reader import ConfigReader
    from src.core.build import build_graph, build_model

    cfg = ConfigReader.read_config(config_file, validate=False)
    if "head" in cfg or "scales" in cfg:
        return build_graph(cfg)
    return build_model(cfg)


class Runner:
    def __init__(self, model: nn.Module, config: TrainerConfig,
                 train_loader=None, val_loader=None,
                 criterion: Optional[nn.Module] = None,
                 optimizer_factory: Optional[Callable] = None,
                 device: Optional[str] = None,
                 metric_fn: Optional[Callable] = None):
        self.config = config
        self.device = _auto_device(device or config.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        # factory(params, lr) -> Optimizer. Rebuilt per phase so lr changes and
        # post-quantization parameter changes both take effect.
        self._opt_factory = optimizer_factory or (
            lambda params, lr: torch.optim.Adam(params, lr=lr))
        self.optimizer = self._opt_factory(self.model.parameters(), config.lr)
        self.metric_fn = metric_fn or self._top1_accuracy

        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        self.ckpt = CheckpointManager(config.save_dir, config.model_name, config)
        self._epochs_no_improve = 0
        self.default_callbacks: list = []        # always-on callbacks (e.g. DB tracking)
        self.show_progress = True                # per-batch tqdm bar (a TUI may disable it)
        self.max_steps: Optional[int] = None     # cap batches/epoch for a quick smoke run

        # optional resume
        if config.resume_from:
            p = Path(config.resume_from)
            if p.exists():
                self.model.load_state_dict(torch.load(p, map_location=self.device))
                print(f"[runner] resumed weights from {p}")

    # ---- factory ----------------------------------------------------------
    @classmethod
    def from_experiment(cls, experiment_file: str, overrides: Optional[dict] = None) -> "Runner":
        from src.config.reader import ConfigReader
        from src.datasets.factory import build_dataset

        cfg = ConfigReader.read_config(experiment_file, validate=False)
        if overrides:  # CLI flags override the yaml (yaml is the base)
            cfg.update({k: v for k, v in overrides.items() if v is not None})

        print(f"→ building model from {cfg['model']}")
        model = build_network(cfg["model"])
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model: {n_params/1e6:.3f}M params")
        print(f"→ loading dataset '{cfg['dataset']}' (first run may download)…")
        dataset = build_dataset(cfg["dataset"], cfg)
        print("  dataset ready")

        tcfg = TrainerConfig(
            save_dir=cfg.get("output_dir", "./outputs"),
            experiment_name=cfg.get("name", "experiment"),
            model_name=cfg.get("name", "model"),
            epochs=cfg.get("epochs", 100),
            lr=cfg.get("learning_rate", cfg.get("lr", 1e-3)),
            best_metric_name=cfg.get("best_metric_name", "accuracy"),
            best_metric_mode=cfg.get("best_metric_mode", "max"),
            validate_every=cfg.get("validate_every", 1),
            device=cfg.get("device"),
            save_best=cfg.get("save_best", True),
            save_last=cfg.get("save_last", True),
        )
        runner = cls(model, tcfg, dataset.get_train_loader(), dataset.get_test_loader())

        # optional experiment tracking: enable by adding `tracking_db: path.db` to the yaml
        if cfg.get("tracking_db"):
            from pathlib import Path

            from src.tracking import DBTrackerCallback, ExperimentDB
            model_yaml = Path(cfg["model"]).read_text() if Path(cfg["model"]).exists() else ""
            exp_yaml = Path(experiment_file).read_text()
            db = ExperimentDB(cfg["tracking_db"])
            runner.default_callbacks.append(
                DBTrackerCallback(db, cfg.get("name", "experiment"), model_yaml, exp_yaml))
        return runner

    # ---- core loops -------------------------------------------------------
    def train(self, plan: Optional[List[Phase]] = None, callbacks: Optional[list] = None) -> Dict[str, list]:
        """Run training. Without a `plan` it trains config.epochs in one phase.

        Pass a `plan` (list of Phase) for multi-stage recipes, e.g. float warmup
        then quantize-and-fine-tune:

            runner.train(plan=[
                Phase("warmup", 10, lr=1e-3),
                Phase("higher_lr", 15, lr=2e-3),
                Phase("qat", 20, lr=1e-4, on_start=lambda r: r.replace_model(quantize(r.model))),
            ])

        `callbacks` (run-level) and `Phase.callbacks` (phase-only) are objects with
        optional on_phase_start(runner, phase) / on_epoch_end(runner, epoch, metrics)
        methods (Observer hooks). For each phase the two lists are merged.
        """
        callbacks = [*(callbacks or []), *self.default_callbacks]
        plan = plan or [Phase("train", self.config.epochs, lr=self.config.lr)]
        history: Dict[str, list] = {"epoch": [], "phase": [], "train_loss": [],
                                    "val_loss": [], "val_metric": []}
        epoch = 0
        for phase in plan:
            phase_callbacks = [*callbacks, *phase.callbacks]   # run-level + phase-only
            if phase.on_start is not None:
                phase.on_start(self)                 # may quantize / mutate the model
                self.model.to(self.device)
                self.ckpt.reset_best()               # arch may have changed -> best.pth restarts
            lr = phase.lr if phase.lr is not None else self.config.lr
            self._rebuild_optimizer(lr)
            print(f"\n=== phase '{phase.name}': {phase.epochs} epochs, lr={lr}, device={self.device} ===")
            self._epochs_no_improve = 0              # patience resets per phase
            for cb in phase_callbacks:
                if hasattr(cb, "on_phase_start"):
                    cb.on_phase_start(self, phase)
            stop = False
            for _ in range(phase.epochs):
                epoch += 1
                stop = self._run_epoch(epoch, phase, history, phase_callbacks)
                if stop:
                    break
            if stop:
                break
        for cb in callbacks:                       # run-level end-of-training hook
            if hasattr(cb, "on_train_end"):
                cb.on_train_end(self, history)
        return history

    def _run_epoch(self, epoch: int, phase: Phase, history: Dict[str, list], callbacks: list) -> bool:
        """Run one epoch; returns True if early stopping should halt training."""
        tag = f"[{phase.name}] epoch {epoch}"
        tr_loss = self._train_one_epoch(desc=f"{tag} train")

        val: Dict[str, float] = {}
        improved = False
        if self.config.validate_every and epoch % self.config.validate_every == 0 and self.val_loader:
            val = self.validate(desc=f"{tag} val")
            improved = self._maybe_save_best(epoch, val)

        history["epoch"].append(epoch)
        history["phase"].append(phase.name)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val.get("loss"))
        history["val_metric"].append(val.get(self.config.best_metric_name))

        if self.config.save_last:
            self.ckpt.save_last(self.model, epoch, metrics=val)
        if self.config.save_metrics_history:
            self._save_history(history)

        msg = f"[{phase.name}] epoch {epoch}  train_loss={tr_loss:.4f}"
        if val:
            msg += (f"  val_loss={val['loss']:.4f}"
                    f"  {self.config.best_metric_name}={val[self.config.best_metric_name]:.4f}")
            if improved:
                msg += "  *best"
        print(msg)

        for cb in callbacks:
            if hasattr(cb, "on_epoch_end"):
                cb.on_epoch_end(self, epoch, {"train_loss": tr_loss, **val})

        if self.config.early_stopping_patience and val:
            self._epochs_no_improve = 0 if improved else self._epochs_no_improve + 1
            if self._epochs_no_improve >= self.config.early_stopping_patience:
                print(f"[runner] early stopping at epoch {epoch} "
                      f"(no improvement for {self._epochs_no_improve} epochs)")
                return True
        return False

    def _rebuild_optimizer(self, lr: float) -> None:
        self.optimizer = self._opt_factory(self.model.parameters(), lr)

    def replace_model(self, model: nn.Module) -> None:
        """Swap in a transformed model (e.g. a quantized version) mid-plan."""
        self.model = model.to(self.device)

    def _save_history(self, history: Dict[str, list]) -> None:
        with open(Path(self.config.save_dir) / "metrics_history.json", "w") as f:
            json.dump(history, f, indent=2)

    def _progress(self, loader, desc):
        """Wrap a loader in a tqdm bar if enabled and available, else return as-is."""
        if not self.show_progress:
            return loader
        try:
            from tqdm import tqdm
            return tqdm(loader, desc=desc, leave=False)
        except Exception:
            return loader

    def _train_one_epoch(self, desc: str = "train") -> float:
        self.model.train()
        total, n = 0.0, 0
        bar = self._progress(self.train_loader, desc)
        for step, (x, y) in enumerate(bar):
            if self.max_steps is not None and step >= self.max_steps:
                break
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(loss=f"{total / max(n, 1):.4f}")
        return total / max(n, 1)

    @torch.no_grad()
    def validate(self, loader=None, desc: str = "val") -> Dict[str, float]:
        loader = loader or self.val_loader
        self.model.eval()
        total_loss, correct, n = 0.0, 0, 0
        for step, (x, y) in enumerate(self._progress(loader, desc)):
            if self.max_steps is not None and step >= self.max_steps:
                break
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x)
            total_loss += self.criterion(out, y).item() * x.size(0)
            correct += self.metric_fn(out, y) * x.size(0)
            n += x.size(0)
        n = max(n, 1)
        return {"loss": total_loss / n, "accuracy": correct / n}

    def evaluate(self, checkpoint: str = "best.pth", loader=None) -> Dict[str, float]:
        """Load a checkpoint (default best.pth) and run the test/val loader."""
        path = Path(self.config.save_dir) / checkpoint
        if path.exists():
            state = torch.load(path, map_location=self.device)
            try:
                self.model.load_state_dict(state)
            except RuntimeError as e:
                print(f"[warn] strict load of {checkpoint} failed ({type(e).__name__}); "
                      "loading non-strict — checkpoint architecture may differ")
                self.model.load_state_dict(state, strict=False)
        else:
            print(f"[warn] {path} not found, evaluating current weights")
        return self.validate(loader or self.val_loader, desc="eval")

    # ---- helpers ----------------------------------------------------------
    def _maybe_save_best(self, epoch: int, val: Dict[str, float]) -> bool:
        if not self.config.save_best:
            return False
        value = val.get(self.config.best_metric_name)
        if value is None:
            return False
        # delegate best-tracking + persistence to the CheckpointManager (single owner)
        return self.ckpt.save_best(self.model, value, self.config.best_metric_name,
                                   self.config.best_metric_mode, epoch=epoch)

    @staticmethod
    def _top1_accuracy(out: torch.Tensor, y: torch.Tensor) -> float:
        return (out.argmax(1) == y).float().mean().item()
