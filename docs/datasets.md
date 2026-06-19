# Datasets

A name-keyed factory with injectable preprocessing. Each dataset ships a sensible
default transform; an injected `transform` overrides it.

```python
from src.datasets import build_dataset, list_datasets
ds = build_dataset("cifar10", config, transform=None)
train_loader, test_loader = ds.get_train_loader(), ds.get_test_loader()
```

`config` is the experiment dict (keys `train_path`, `test_path`, `batch_size`,
`num_workers`, `distributed`, and for some datasets `dataset_path` /
`dataset_root_path`, `openml_name`). The factory passes only the kwargs a dataset's
constructor accepts.

## Built-in datasets

| name(s) | source | notes |
| --- | --- | --- |
| `cifar10` | torchvision CIFAR-10 | downloads to `train_path` root |
| `mnist` | torchvision MNIST | grayscale default transform |
| `fashion_mnist` / `fashionmnist` | torchvision FashionMNIST | |
| `imagenet` / `imagefolder` | folder of class subdirs, or a **`.zip`** archive | auto-extracted on first use |
| `openml` / `jsc` | OpenML (default `hls4ml_lhc_jets_hlf`) | tabular; needs scikit-learn; standardized features |
| `yolo` / `coco` / `detection` | YOLO-format images + label txts | returns `(image, {boxes, labels})` + a detection collate |

## Default preprocessing

`src.datasets.preprocessing` holds the reusable defaults:
`classification_transform`, `cifar_transform`, `gray_transform`,
`detection_transform`. Override per experiment by passing a `transform` to
`build_dataset`.

## Adding a dataset

Subclass `BaseDataset`, implement `_do_preprocessing` (use `self.transform` if set),
and register it:

```python
from src.datasets.base import BaseDataset
from src.datasets.factory import register_dataset

@register_dataset("my_data")
class MyData(BaseDataset):
    def _do_preprocessing(self, **kwargs):
        tf = self.transform or my_default_transform()
        return train_dataset, test_dataset
```

Import-register it from `factory._register_builtin_datasets` (or import the module
once) so the name is available. Detection training needs a detection-capable
trainer (the classification Runner expects `(x, y)` batches).
