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
            dataset_root_path = config["dataset_root_path"]

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
            abs_dataset_path = os.path.join(os.path.dirname(__file__), dataset_root_path)
            print("abs_dataset_path: ", abs_dataset_path)
            dataset_path = os.path.join(abs_dataset_path,
                                        "processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z")
            print("dataset_path: ", dataset_path)
            return JetSubstructureDataset(dataset_path, batch_size, distributed, num_workers)

        else:
            raise Exception(f"{dataset} not implemented.")