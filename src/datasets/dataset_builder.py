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
    def build(dataset, config_path):
        config = None
        batch_size = [0,0]
        num_workers = 0
        distributed = False
        train_path = ""
        test_path = ""
        dataset_root_path = ""

        # Load config file
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            train_path = config["train_path"]
            test_path = config["test_path"]

            if Mapper.has_key(config, "val_path") is True:
                val_path = config["val_path"]

            batch_size = config["batch_size"]
            num_workers = config["num_workers"]
            distributed = config["distributed"]


        # Build model
        if dataset == DatasetTypes.MNIST:
            return Mnist(train_path, test_path, batch_size, distributed, num_workers)
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