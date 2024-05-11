import src.models.jsc
from src.core.inject.enum import ModelTypes
from src.models.resnet import ResNet
from src.models.lenet import LeNet5
from src.models.vgg import Vgg
from src.models.jsc import Jsc

import yaml

class ModelBuilder:

    @staticmethod
    def build(modeltype, config_path, preload_weights=False):

        # Load config file
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Build model
        if modeltype == ModelTypes.LENET:
            model = LeNet5(config, preload_weights)
            return model
        elif modeltype == ModelTypes.VGG:
            model = Vgg(config, preload_weights)
            return model
        elif modeltype == ModelTypes.RESNET:
            model = ResNet(config, preload_weights)
            return model
        elif modeltype == ModelTypes.JSC:
            model = Jsc(config, preload_weights)
            return model
        elif modeltype == ModelTypes.HDR:
            #model = Hdr(config, preload_weights)
            model = None
            return model
        else:
            raise Exception(f"{modeltype} not implemented.")