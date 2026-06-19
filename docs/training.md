# Training — the Runner

`src.core.Runner` is the single entry point for `train` / `validate` / `evaluate`.
It ties together the model builder, the dataset factory and the CheckpointManager.

```python
from src.core import Runner
runner = Runner.from_experiment("configs/experiments/jsc/jsc_2l_qat.yaml")
runner.train()                 # writes best.pth + last.pth + metrics_history.json
print(runner.evaluate())       # loads best.pth, runs the val/test loader
```

Construct directly for full control:

```python
Runner(model, config, train_loader, val_loader,
       criterion=None,          # default: CrossEntropyLoss
       optimizer_factory=None,  # default: Adam; called (params, lr) per phase
       device=None,             # default: auto CUDA/ROCm → MPS → CPU
       metric_fn=None)          # default: top-1 accuracy
```

## Phases (multi-stage training)

`train(plan=[...])` runs a list of `Phase`s. Each phase has its own epoch count and
lr, and an optional `on_start(runner)` hook that may **mutate the model** (e.g.
quantize). The optimizer is rebuilt at every phase start, so a changed lr and
changed parameters both take effect. Patience resets per phase, and `best` is reset
when a phase changes the architecture (so `best.pth` always matches the final net).

```python
from src.core import Phase, quantize_model
runner.train(plan=[
    Phase("warmup", epochs=10, lr=1e-3),
    Phase("qat",    epochs=20, lr=1e-4,
          on_start=lambda r: r.replace_model(quantize_model(r.model, 4, 4))),
])
```

For a full bit-width schedule with rollback, see [qat.md](qat.md).

## Callbacks

Run-level (`train(callbacks=[...])`) and phase-level (`Phase(..., callbacks=[...])`)
hooks are merged per phase. A callback is any object with optional methods:

- `on_phase_start(runner, phase)`
- `on_epoch_end(runner, epoch, metrics)`
- `on_train_end(runner, history)`

Built-in callbacks: `src.ui.RichDashboardCallback` (live terminal table),
`src.tracking.DBTrackerCallback` (SQLite logging).

## Checkpoints

`best.pth` (best `best_metric_name`) and `last.pth` are written every run, plus
`metrics_history.json`. `--dry-run` disables writing. The CheckpointManager owns
best-tracking (`save_best` only writes on improvement).

## Devices

Auto-selected via `DeviceSelector`: NVIDIA CUDA and AMD ROCm share the `cuda` path
(ROCm reports through it), then Apple MPS, then CPU. Override with `--device` or
`Runner(..., device=...)`.

## Experiment tracking

Set `tracking_db: path.db` in the experiment yaml (or `--track-db`) to record
per-epoch metrics, the best/last weights (as BLOBs) and the yamls into SQLite for
reproducibility. Query later:

```sh
sqlite3 outputs/.../experiments.db "SELECT epoch, split, loss, accuracy FROM metrics;"
```
