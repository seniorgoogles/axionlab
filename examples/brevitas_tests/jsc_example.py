import sys
import torch 
import numpy as np
from torch.nn import Linear

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
from src.engine.config_manager import ConfigurationManager

if __name__ == "__main__":
    
    base_quant_model_config = f"{parent_directory}/examples/ml2/jsc_xl_quant_base_config.yaml"
    
    base_quant_model_config_percentile = f"{parent_directory}/examples/brevitas_tests/jsc_xl_quant_base_config_percentile.yaml"
    base_quant_model_config_threshold = f"{parent_directory}/examples/brevitas_tests/jsc_xl_quant_base_config_threshold.yaml"

    float_model_weight_path = f"{parent_directory}/train/jsc_xl/run_25/best_weights.pth"
    float_model_config_path = f"{parent_directory}/configs/jsc/jsc_xl.yaml"
    
    float_model = ModelBuilder().build(ModelTypes.JSC, config=float_model_config_path, weights_path=float_model_weight_path)
    quant_model_percentile = ModelBuilder().build(ModelTypes.JSC, config=base_quant_model_config_percentile, weights_path=float_model_weight_path)
    quant_model_threshold = ModelBuilder().build(ModelTypes.JSC, config=base_quant_model_config_threshold, weights_path=float_model_weight_path)

    dataset = DatasetBuilder().build(DatasetTypes.JSC, config=float_model_config_path)
    validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())
    
    acc_float_model,_ = validator.validate(float_model)
    acc_quant_perc_model,_ = validator.validate(quant_model_percentile)
    acc_quant_thres_model,_ = validator.validate(quant_model_threshold)

    acc_float_model = round(acc_float_model, 2)
    acc_quant_perc_model = round(acc_quant_perc_model, 2)
    acc_quant_thres_model = round(acc_quant_thres_model, 2)

    # Clear terminal
    #print("\033[H\033[J")
    
    print("|----------------------------------------------------")
    print("| Results: ")
    print("|----------------------------------------------------")
    print(f"| Accuracy Float Model: {acc_float_model}%")
    print(f"| Accuracy Quant Perc Model: {acc_quant_perc_model}%")
    print(f"| Accuracy Quant Thres Model: {acc_quant_thres_model}%")
    print("|----------------------------------------------------\n\n\n")
    

    print("Percentile model")
    print("----------------------------------------------------")  
    # Iterate over model and get the weights of dense layers
    for name, module in quant_model_percentile.named_modules():
        if isinstance(module, Linear):
            
            weights = module.quant_weight()[0].cpu().detach().numpy()
            
            # Get the number of params from weight
            num_params = weights.flatten().size
            num_zeros = np.count_nonzero(weights == 0)


            percentage_zeros = (num_zeros / num_params) * 100
            print(f"Layer: {name}")
            print(f"Number of parameters: {num_params}")
            print(f"Number of zeros: {num_zeros}")
            print(f"Percentage of zeros: {percentage_zeros}%")
            print("------------------------------------")
            
    print("\n\n\n")
    print("Threshold model")
    print("----------------------------------------------------")       
    for name, module in quant_model_threshold.named_modules():
        if isinstance(module, Linear):
            
            weights = module.quant_weight()[0].cpu().detach().numpy()
            
            # Get the number of params from weight
            num_params = weights.flatten().size
            num_zeros = np.count_nonzero(weights == 0)

            # Get max abs value
            max_abs = np.max(np.abs(weights))


            percentage_zeros = (num_zeros / num_params) * 100
            print(f"Layer: {name}")
            print(f"Number of parameters: {num_params}")
            print(f"Number of zeros: {num_zeros}")
            print(f"Percentage of zeros: {percentage_zeros}%")
            print("Max abs value: ", max_abs)
            print("------------------------------------")