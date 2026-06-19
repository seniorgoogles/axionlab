"""Object-detection dataset (YOLO/COCO-style) as a data source.

Reads a YOLO-format layout: an images dir and a labels dir with one .txt per
image (`class cx cy w h`, normalized). Returns (image_tensor, target) where
target = {"boxes": [N,4] xywh-norm, "labels": [N]}.

NOTE: detection TRAINING needs a detection loss + assigner + box-aware collate
(ultralytics provides these); the classification Runner does not fit. This class
gives you the data + a collate_fn so a detection-capable trainer can consume it.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.datasets.base import BaseDataset
from src.datasets.factory import register_dataset
from src.datasets.preprocessing import detection_transform

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


class _YoloFolder(Dataset):
    def __init__(self, images_dir, labels_dir, transform):
        self.images = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in _IMG_EXT)
        self.labels_dir = Path(labels_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_path = self.images[i]
        img = self.transform(Image.open(img_path).convert("RGB"))
        label_path = self.labels_dir / (img_path.stem + ".txt")
        boxes, labels = [], []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                c, cx, cy, w, h = (float(v) for v in line.split())
                labels.append(int(c))
                boxes.append([cx, cy, w, h])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return img, target


def detection_collate(batch):
    """Keep variable-length targets as a list (don't stack boxes)."""
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


@register_dataset("yolo", "coco", "detection")
class YoloDetectionDataset(BaseDataset):
    def _do_preprocessing(self, image_size: int = 640, labels_subdir: str = "labels",
                          images_subdir: str = "images", **kwargs):
        tf = self.transform or detection_transform(image_size)

        def split(root):
            root = Path(root)
            imgs = root / images_subdir if (root / images_subdir).exists() else root
            lbls = root / labels_subdir
            return _YoloFolder(imgs, lbls, tf)

        return split(self.train_path), split(self.test_path)

    def get_train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size_train, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=detection_collate)

    def get_test_loader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size_test, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=detection_collate)
