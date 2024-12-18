import os

from src.core.inject.enum import DatasetTypes
from src.datasets.imagenet import ImageNet
from src.datasets.jetSubstructure.dataloader import JetSubstructureDataset
from src.datasets.mnist import Mnist
from src.datasets.fashionMnist import FashionMnist
from src.utils.mapper import Mapper
import yaml

class DatasetBuilder:
    @staticmethod
    def build(dataset, config=None, crop_border_pixels=0):
        batch_size = [0,0]
        num_workers = 0
        distributed = False
        train_path = ""
        test_path = ""

        # Check if config is a path or an object
        config_path = config if isinstance(config, str) else None
        config = config if not isinstance(config, str) else None

        if config == None:
            # Check if config file exists, if not raise Exception
            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    config = yaml.safe_load(file)
            else:
                raise Exception("Config does not exist")

        train_path = config["train_path"]
        test_path = config["test_path"]

        batch_size = config["batch_size"]
        num_workers = config["num_workers"]
        distributed = config["distributed"]


        # Build model
        if dataset == DatasetTypes.MNIST:
            return Mnist(train_path, test_path, batch_size, distributed, num_workers, crop_border_pixels)
        if dataset == DatasetTypes.FASHION_MNIST:
            return FashionMnist(train_path, test_path, batch_size, distributed, num_workers)
        if dataset == DatasetTypes.CIFAR10:
            raise NotImplementedError
        if dataset == DatasetTypes.IMAGENET:
            return ImageNet(train_path, test_path, batch_size, distributed, num_workers)
        if dataset == DatasetTypes.COCO:
            raise NotImplementedError
        if dataset == DatasetTypes.JSC:
            dataset_path = config["dataset_root_path"]
            project_root = os.getenv("PROJECT_ROOT")

            if project_root is not None:
                # If dataset path has a leading slash, remove it
                dataset_path = dataset_path[1:] if dataset_path.startswith("/") else dataset_path
                dataset_path = os.path.join(project_root, dataset_path)

            return JetSubstructureDataset(dataset_path, batch_size, distributed, num_workers)

        else:
            raise Exception(f"{dataset} not implemented.")
