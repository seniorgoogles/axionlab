import os
import torch
import typing
import yaml

from src.core.inject.enum import ModelTypes
from src.models.resnet import ResNet
from src.models.lenet import LeNet5
from src.models.vgg import Vgg
from src.models.hdr import Hdr
from src.models.jsc import Jsc
from src.models.continuousflow_mnist import ContinuousflowMnist
from typing import Union

class ModelBuilder:

    @staticmethod
    def load_config(config_path: str) -> dict:
        with open(config_path) as config_yaml:
            return yaml.load(config_yaml, Loader=yaml.FullLoader)

    @staticmethod
    def get_model_instance(modeltype: ModelTypes, config: dict, preload_weights: bool):
        model_map = {
            ModelTypes.LENET: LeNet5,
            ModelTypes.VGG: Vgg,
            ModelTypes.RESNET: ResNet,
            ModelTypes.JSC: Jsc,
            ModelTypes.HDR: Hdr,
            ModelTypes.CONTINUOUSFLOW_MNIST: ContinuousflowMnist
        }

        if modeltype not in model_map:
            raise ValueError(f"Model type {modeltype} not implemented.")

        return model_map[modeltype](config, preload_weights)

    @staticmethod
    def load_model_weights(model, weights_path: str, weights_only: bool, weights_strict_loading: bool):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Could not load weights from {weights_path}. File does not exist.")

        model.load_state_dict(torch.load(weights_path, weights_only=weights_only), strict=weights_strict_loading)

    @staticmethod
    def build(modeltype: ModelTypes, config: Union[dict, str] = None, preload_weights: bool = False,
              weights_path: str = None, weights_only: bool = False, weights_strict_loading: bool = False):

        if isinstance(config, str):
            config = ModelBuilder.load_config(config)

        model = ModelBuilder.get_model_instance(modeltype, config, preload_weights)

        if weights_path:
            ModelBuilder.load_model_weights(model, weights_path, weights_only, weights_strict_loading)

        return model
