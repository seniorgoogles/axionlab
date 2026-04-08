import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import distributed
from src.datasets.base import BaseDataset


class ImageNet(BaseDataset):
    """ImageNet dataset wrapper."""

    def _do_preprocessing(self):
        """Create ImageNet datasets with appropriate transforms."""
        # Data transformations
        normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        train_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])
        test_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize,
        ])

        train_dataset = datasets.ImageFolder(
            root=self.train_path,
            transform=train_transform
        )
        test_dataset = datasets.ImageFolder(
            root=self.test_path,
            transform=test_transform
        )

        return train_dataset, test_dataset