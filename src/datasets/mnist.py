import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import distributed
from src.datasets.base import BaseDataset


class BorderCrop:
    """Crop borders from images for MNIST preprocessing."""

    def __init__(self, border):
        self.border = border

    def __call__(self, img):
        width, height = img.size
        return T.functional.crop(img, self.border, self.border, height - 2 * self.border, width - 2 * self.border)


class Mnist(BaseDataset):
    """MNIST dataset wrapper."""

    def _do_preprocessing(self, crop_border_pixels=0):
        """Create MNIST datasets with optional border cropping."""
        if crop_border_pixels > 0:
            transform = T.Compose([
                BorderCrop(border=2),
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,)),
            ])
        else:
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,)),
            ])

        train_dataset = datasets.MNIST(
            '../tmp/dataset/mnist',
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.MNIST(
            '../tmp/dataset/mnist',
            train=False,
            transform=transform
        )

        print(f"{train_dataset.data.shape=}")
        return train_dataset, test_dataset