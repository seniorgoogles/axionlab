from src.core.inject.enum import ModelTypes
from src.models.resnet import ResNet
from src.models.lenet import LeNet5

import yaml

class ModelBuilder:
    @staticmethod
    def build(model, config_path, preload_weights=False):
        config = None

        # Load config file
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Build model
        if model == ModelTypes.LENET:
            return LeNet5(config, preload_weights)
        elif model == ModelTypes.VGG:
            raise NotImplementedError
        elif model == ModelTypes.RESNET:
            return ResNet(config, preload_weights)
        else:
            raise Exception(f"{model} not implemented.")