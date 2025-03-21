import os
import torch
import typing
import yaml

from synapselab.core.inject.enum import ModelTypes
from synapselab.engine.config import Configuration
from synapselab.models.resnet import ResNet
from synapselab.models.lenet import LeNet5
from synapselab.models.vgg import Vgg
from synapselab.models.hdr import Hdr
from synapselab.models.jsc import Jsc
from synapselab.models.continuousflow_mnist import ContinuousflowMnist
from typing import Union

class ModelBuilder:
    """
    @staticmethod
    def load_config(config_path: str) -> dict:
        with open(config_path) as config_yaml:
            return yaml.load(config_yaml, Loader=yaml.FullLoader)
    """
    @staticmethod
    def get_model_instance(config, preload_weights: bool):
        model_type = config.model_type
        model_map = {
            "LENET": LeNet5,
            "VGG": Vgg,
            "RESNET": ResNet,
            "JSC": Jsc,
            "HDR": Hdr
        }

        if model_type not in model_map:
            raise ValueError(f"Model type {model_type} not implemented.")

        return model_map[model_type](config, preload_weights)

    @staticmethod
    def load_model_weights(model, weights_path: str, weights_only: bool, weights_strict_loading: bool):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Could not load weights from {weights_path}. File does not exist.")

        model.load_state_dict(torch.load(weights_path, weights_only=weights_only), strict=weights_strict_loading)

    @staticmethod
    def build(config: Configuration = None, preload_weights: bool = False,
              weights_path: str = None, weights_only: bool = False, weights_strict_loading: bool = False):

        model = ModelBuilder.get_model_instance(config, preload_weights)

        if weights_path:
            ModelBuilder.load_model_weights(model, weights_path, weights_only, weights_strict_loading)

        return model
