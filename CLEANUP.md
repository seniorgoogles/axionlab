# Cleanup report

Survey of dead / superseded code after the refactor. I could not delete files
from here (no shell), so this lists exactly what is safe to remove and why. Run
the `rm` block from the repo root, then re-run `pytest` to confirm green.

## Dead / orphaned (nothing live imports these)

| File / dir | Superseded by | Why dead |
| --- | --- | --- |
| `src/datasets/cifar10.py`, `mnist.py`, `fashionMnist.py`, `imagenet.py` | `src/datasets/vision.py` | old, non-`BaseDataset`, hardcoded transforms (cifar10.py even had a `datasets.Cifar10` typo) |
| `src/datasets/dataset_builder.py` | `src/datasets/factory.py` | enum-based factory replaced by the name-keyed factory |
| `src/datasets/jetSubstructure/` | `src/datasets/tabular.py` (OpenML `jsc`) | JSC now loaded from OpenML; remove only if you don't need local-file JSC |
| `src/core/inject/` | — | `enum.py` (DatasetTypes/ModelTypes) only used by the dead `dataset_builder` |
| `src/utils/mapper.py` | — | only used by the dead `dataset_builder` |
| `src/models/` (`base.py`, `registry.py`, `config.py`) | `src/core/build/` | model building is done by the builders; this registry is unused |

```sh
# from repo root
rm -f src/datasets/cifar10.py src/datasets/mnist.py src/datasets/fashionMnist.py src/datasets/imagenet.py
rm -f src/datasets/dataset_builder.py
rm -f src/utils/mapper.py
rm -rf src/core/inject
rm -rf src/models
# optional (only if you don't need local-file JSC; OpenML 'jsc' replaces it):
rm -rf src/datasets/jetSubstructure
```

## Deprecated but kept (backward compatibility)

`src/training/trainer.py`, `model.py`, `validator.py`, `callbacks.py`,
`metric_tracker.py` are superseded by `src.core.Runner`. They now emit
`DeprecationWarning` and are loaded lazily (so importing the Runner no longer
pulls them or `tqdm`). Remove once nothing external depends on them.

## Architecture changes made

- `src/training/__init__.py` now imports only the lightweight pieces eagerly
  (`TrainerConfig`, `CheckpointManager`) and lazy-loads the rest (PEP 562) →
  the Runner import path is decoupled from the deprecated `Trainer`/`Model`/
  `Validator`. Guarded by `tests/test_imports.py`. (Note: `tqdm` is still pulled
  transitively by `import torch` itself; that's outside our control and not
  asserted.)
- Best-checkpoint tracking lives only in `CheckpointManager` now; the Runner
  delegates (single source of truth).

## Known remaining smell (not changed — low value / churn risk)

`src/config/types.py` and `src/config/validator.py` define overlapping
dataclasses (`LayerDefinition`, `DatasetConfig`, ...). The current
build/run flow uses raw dicts via `ConfigReader`, so this validation API is
largely unused. Consolidate to one module if you start relying on it.
