import json
import shutil
from includes_ml2 import *
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    
    checkpoint = 331 

    model_config = f"{parent_directory}/configs/jsc/quant_jsc_xl.yaml"
    best_weights = f"/home/mmecik/repositories/synapselab/train/jsc_xl_floating_point/run_{checkpoint}/best_weights.pth"
    
    # Layers
    layer_str_list = ["dense1", "dense2", "dense3", "dense4", "dense5"]

    # Sparsity
    sparsity_range = np.arange(0.0, 1.0, 0.1)   
    
    # Weights 
    weights_range = range(8, 1, -1)
    
    model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=best_weights)
    sorted_layer_str_list = sort_layers_by_param_num(model, layer_str_list)
        
    model_config_changed = set_config_value(model_config, "dense1", "weight_sparse_eps", 0.3)
    
    for key in model_config_changed.keys():
        
        if isinstance(model_config_changed[key], list):
            print(f"{key}:")
            for i, value in enumerate(model_config_changed[key]):
                print(f" - {value}")
        else:   
            print(f"{key} = {model_config_changed[key]}")
        