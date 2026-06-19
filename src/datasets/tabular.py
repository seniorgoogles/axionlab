"""OpenML / tabular datasets (e.g. JSC jet-substructure) as BaseDataset.

Fetches an OpenML dataset by id or name, standardizes features (default
preprocessing) and exposes train/test TensorDatasets. An injected `self.transform`
(callable on the feature matrix) overrides the default standardization.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.datasets.base import BaseDataset
from src.datasets.factory import register_dataset


def _standardize(x_train: np.ndarray, x_test: np.ndarray):
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-8
    return (x_train - mean) / std, (x_test - mean) / std


@register_dataset("openml", "jsc")
class OpenMLDataset(BaseDataset):
    """Tabular classification from OpenML. Configure via `openml_id` or `openml_name`."""

    def _do_preprocessing(self, openml_id=None, openml_name="hls4ml_lhc_jets_hlf",
                          test_split: float = 0.2, seed: int = 0, **kwargs):
        from sklearn.datasets import fetch_openml  # heavy import kept local

        # parser="liac-arff" returns numpy arrays without requiring pandas
        ds = fetch_openml(data_id=openml_id, name=None if openml_id else openml_name,
                          as_frame=False, parser="liac-arff")
        x = np.asarray(ds.data, dtype=np.float32)
        y_raw = np.asarray(ds.target)
        classes = sorted(set(y_raw.tolist()))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y = np.array([class_to_idx[v] for v in y_raw], dtype=np.int64)

        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(x))
        n_test = int(len(x) * test_split)
        test_idx, train_idx = perm[:n_test], perm[n_test:]
        x_train, x_test = x[train_idx], x[test_idx]

        if self.transform is not None:
            x_train, x_test = self.transform(x_train), self.transform(x_test)
        else:
            x_train, x_test = _standardize(x_train, x_test)

        self.num_classes = len(classes)
        train = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y[train_idx]))
        test = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y[test_idx]))
        return train, test
