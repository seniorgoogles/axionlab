from src.core.inject.enum import DatasetTypes
from src.utils.datasets.imagenet import ImageNet

import yaml

def has_key(yaml, key):
    try:
        return True if key in yaml else False
    except yaml.YAMLError as exc:
        print(exc)
        return False

class DatasetBuilder:
    @staticmethod
    def build(dataset, config_path):
        config = None
        batch_size = [0,0]
        num_workers = 0
        distributed = False
        train_path = ""
        test_path = ""

        # Load config file
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            train_path = config["train_path"]
            test_path = config["test_path"]

            if has_key(config, "val_path") is True:
                val_path = config["val_path"]

            batch_size = config["batch_size"]
            num_workers = config["num_workers"]
            distributed = config["distributed"]

        # Build model
        if dataset == DatasetTypes.MNIST:
            raise NotImplementedError
        if dataset == DatasetTypes.FASHION_MNIST:
            raise NotImplementedError
        if dataset == DatasetTypes.CIFAR10:
            raise NotImplementedError
        if dataset == DatasetTypes.IMAGENET:
            return ImageNet(train_path, test_path, batch_size, distributed, num_workers)
        if dataset == DatasetTypes.COCO:
            raise NotImplementedError
        else:
            raise Exception(f"{dataset} not implemented.")