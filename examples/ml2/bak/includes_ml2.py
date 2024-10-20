import sys
import torch
import os
import yaml
import logging
import random
import traceback
from torchsummary import summary
import numpy as np

# Add the parent directory to the path so project modules can be imported
from pathlib import Path

current_script_path = Path(__file__).parent
parent_directory = current_script_path.parent.parent

sys.path.append(str(parent_directory))

from src.datasets.dataset_builder import DatasetBuilder
from src.models.model_builder import ModelBuilder
from src.core.inject.enum import ModelTypes, DatasetTypes
from src.engine.validator import Validator
from src.engine.trainer import Trainer
from src.engine.tuner import Tuner
from src.engine.config_manager import ConfigurationManager

from src.quantizer.learned_bitwidth_quantizer import LearnedBitWidthQuantizer


from examples.ml2.helpers import *

model_configurations = [
    #"configs/jsc/jsc_2l.yaml",
    #"configs/jsc/jsc_5l.yaml",
    #"configs/jsc/jsc_lite.yaml",
    #"configs/jsc/jsc_m_lite_floating_point.yaml",
    f"{parent_directory}/configs/jsc/jsc_xl.yaml",
    #f"{parent_directory}/configs/jsc/jsc_xl_floating_point.yaml"
]
