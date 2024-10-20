import json
import shutil
from includes_ml2 import *
import numpy as np
import matplotlib.pyplot as plt
from src.engine.config_manager import ConfigurationManager
from src.engine.validator import Validator
from src.datasets.dataset_builder import DatasetBuilder
from src.core.inject.enum import DatasetTypes, ModelTypes


if __name__ == "__main__":

    
    checkpoint = 331 
    best_weights = f"/home/mmecik/repositories/synapselab/train/jsc_xl_floating_point/run_{checkpoint}/best_weights.pth"

    config_manager = ConfigurationManager(config_file_path=f"{parent_directory}/configs/jsc/quant_jsc_xl_updated.yaml")
    
    config_manager.set_layer_param("backbone", "dense1", "weight_sparse_eps", 0.0)
    model = ModelBuilder().build(ModelTypes.JSC, config=config_manager.config, weights_path=best_weights)

    dataset= DatasetBuilder().build(DatasetTypes.JSC, config=config_manager.config)
    validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())

    validator.validate(model)
    
    
    
    
    config_manager.write(f"{parent_directory}/configs/jsc/quant_jsc_xl_updated.yaml")
