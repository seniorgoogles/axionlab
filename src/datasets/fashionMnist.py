import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import distributed
from src.datasets.base import BaseDataset


class FashionMnist(BaseDataset):
    """FashionMNIST dataset wrapper."""

    def _do_preprocessing(self):
        """Create FashionMNIST datasets."""
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,)),
        ])

        train_dataset = datasets.FashionMNIST(
            '../tmp/dataset/fashionmnist',
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            '../tmp/dataset/fashionmnist',
            train=False,
            transform=transform
        )

        return train_dataset, test_dataset