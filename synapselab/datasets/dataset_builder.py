import os
import yaml

from synapselab.engine.config import Configuration
from synapselab.datasets.imagenet import ImageNet
from synapselab.datasets.jetSubstructure.dataloader import JetSubstructureDataset
from synapselab.datasets.mnist import Mnist
from synapselab.datasets.fashionMnist import FashionMnist
from synapselab.utils.mapper import Mapper

class DatasetBuilder:
    @staticmethod
    def build(config: Configuration, crop_border_pixels=0):
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

        train_path = config.train_path
        test_path = config.test_path

        batch_size = config.batch_size
        num_workers = config.num_workers
        distributed = config.distributed

        # Build model
        if config.dataset == "MNIST":
            return Mnist(train_path, test_path, batch_size, distributed, num_workers, crop_border_pixels)
        if config.dataset == "FASHION_MNIST":
            return FashionMnist(train_path, test_path, batch_size, distributed, num_workers)
        if config.dataset == "CIFAR10":
            raise NotImplementedError
        if config.dataset == "IMAGENET":
            return ImageNet(train_path, test_path, batch_size, distributed, num_workers)
        if config.dataset == "COCO":
            raise NotImplementedError
        if config.dataset == "JSC":
            dataset_path = config.dataset_root_path
            project_root = os.getenv("PROJECT_ROOT")

            if project_root is not None:
                # If dataset path has a leading slash, remove it
                dataset_path = dataset_path[1:] if dataset_path.startswith("/") else dataset_path
                dataset_path = os.path.join(project_root, dataset_path)

            return JetSubstructureDataset(dataset_path, batch_size, distributed, num_workers)

        else:
            raise Exception(f"{config.dataset} not implemented.")
