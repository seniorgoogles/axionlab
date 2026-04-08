"""Base dataset class with shared DataLoader methods."""

from torch.utils.data import DataLoader, distributed
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


class BaseDataset(ABC):
    """Abstract base class for datasets with shared DataLoader creation logic."""

    def __init__(
        self,
        train_path: str,
        test_path: str,
        batch_size: Tuple[int, int],
        distributed_training: bool,
        num_workers: int,
        **kwargs
    ):
        """
        Initialize base dataset parameters.

        Args:
            train_path: Path to training data
            test_path: Path to testing data
            batch_size: Tuple of (train_batch_size, test_batch_size)
            distributed_training: Whether to use distributed sampling
            num_workers: Number of workers for data loading
            **kwargs: Additional dataset-specific parameters
        """
        self.batch_size_train = batch_size[0]
        self.batch_size_test = batch_size[1]
        self.distributed_training = distributed_training
        self.num_workers = num_workers
        self.train_path = train_path
        self.test_path = test_path

        # Do preprocessing (to be implemented by subclasses)
        self.train_dataset, self.test_dataset = self._do_preprocessing(**kwargs)

        # If distributed, use distributed sampler for multiple GPUs
        if self.distributed_training:
            self.train_sampler = distributed.DistributedSampler(self.train_dataset)
            self.test_sampler = distributed.DistributedSampler(self.test_dataset)

    @abstractmethod
    def _do_preprocessing(self, **kwargs) -> Tuple:
        """
        Do dataset-specific preprocessing.

        Args:
            **kwargs: Dataset-specific parameters

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        pass

    def get_train_loader(self) -> DataLoader:
        """Create and return a DataLoader for training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size_train,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=self.train_sampler if self.distributed_training else None
        )

    def get_test_loader(self) -> DataLoader:
        """Create and return a DataLoader for testing."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_test,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=self.test_sampler if self.distributed_training else None
        )