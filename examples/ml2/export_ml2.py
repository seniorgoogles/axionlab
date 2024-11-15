import os
import json
import shutil
import torch
import numpy as np
from includes_ml2 import *
from brevitas.nn import QuantConv2d, QuantLinear  # Replace 'some_module' with the actual module name where QuantConv2d is defined
from tqdm import tqdm
import time 

from torch.nn import Linear

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def export_model(model, output_path, validation_dataset, num_samples=100):
    
    # Iterate over the model, save layer, activation function and weights
    model_dict = {}
    
    model_dict["layer0"] = {
        "layerId": 0,
        "numNeurons": 16,
        "activationFunction": "null"
    }
    # Enumerate over the model's named modules
    
    index = 0
    for name, module in model.named_modules():
        if isinstance(module, Linear):
            index += 1
            name = name.replace("dense", "layer")
            model_dict[name] = {
                "layerId": index,
                "numNeurons": module.weight.shape[0],
                "weight": module.weight.cpu().detach().numpy().tolist(),
                "bias": module.bias.cpu().detach().numpy().tolist(),
                "activationFunction": "ReLU"
            }
            
    # Add validation data here
    model_dict["validation"] = {
        "numSamples": num_samples,
        "input": [],
        "results": []
    }
    
    # Iterate over the validation dataset and the batches, until we have num_samples
    for i, (inputs, targets) in enumerate(validation_dataset):
        # Iterate over batch
        for j in range(inputs.shape[0]):
            if len(model_dict["validation"]["input"]) >= num_samples:
                break
            model_dict["validation"]["input"].append(inputs[j].flatten().numpy().tolist())
            model_dict["validation"]["results"].append(targets[j].numpy().tolist())
        
    model_json = json.dumps(model_dict)
    
    # Write to file 
    with open(f"{output_path}.json", "w") as f:
        f.write(model_json)
            
def visualize_layer_differences(layer, weight1, weight2):
    """
    Visualizes the differences between two sets of weights for each layer,
    with square aspect ratio for each cell to ensure consistent spacing.

    Parameters:
    - layers: List of layer names.
    - weights1: List of numpy arrays representing the weights for the first model.
    - weights2: List of numpy arrays representing the weights for the second model.
    """

    # Ensure the weights have the same shape
    if weight1.shape != weight2.shape:
        print(f"Shape mismatch for layer {layer}. Skipping visualization.")
        return

    # Calculate the difference
    difference = weight1 - weight2

    # Plot the difference as a heatmap with square cells
    plt.figure(figsize=(8, 8))
    sns.heatmap(difference, cmap="coolwarm", center=0, annot=False, square=True,
                cbar_kws={'label': 'Difference'})
    plt.title(f"Difference in weights for layer: {layer}")
    plt.xlabel("Input Features")
    plt.ylabel("Output Features")
    plt.show()
    plt.savefig(f"layer_{layer}_difference.png")


#base_model_config = f"{parent_directory}/configs/jsc/quant_jsc_xl_updated_quant.yaml"
float_model_weight_path = f"weights/jsc/jsc_xl_weights.pth"
float_model_config = "/home/mmecik/repositories/synapselab/configs/jsc/jsc_xl.yaml"

        
model_acc_drop = f"50.0"

quant_model_config = f"/home/mmecik/repositories/synapselab/tmp_data/experiment_quant_first/1_pruning/allowed_acc_4.0/config.yaml"
quant_model_weight_path = f"/home/mmecik/repositories/synapselab/tmp_data/experiment_quant_first/1_pruning/allowed_acc_4.0/best_weights.pth"

dataset = DatasetBuilder().build(DatasetTypes.JSC, config=quant_model_config)
quant_model = ModelBuilder().build(ModelTypes.JSC, config=quant_model_config, weights_path=quant_model_weight_path)

validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())

state_dict = copy.deepcopy(quant_model.state_dict())

# Set cache for bias
for name, module in quant_model.named_modules():
    if isinstance(module, QuantLinear):
        module.cache_inference_quant_bias = True
        
# Run inference
model_float = ModelBuilder().build(ModelTypes.JSC, config=float_model_config, weights_path=float_model_weight_path)
model_float.load_state_dict(state_dict)

acc, loss = validator.validate(quant_model)
acc, loss = validator.validate(model_float)

# Iterate over layers and print weights
for name, module in quant_model.named_modules():
    if isinstance(module, QuantLinear):
        
        try: 
            print(f"Layer: {name}")
            print(f"Weight: {module.quant_weight()[0].shape}")

            visualize_layer_differences(name, module.quant_weight()[0].detach().numpy(), module.weight.detach().numpy())
            state_dict[name + ".weight"] = module.quant_weight()[0]
            state_dict[name + ".bias"] = module.quant_bias()[0]
        except Exception as e:
            print(name)
            print(e)

for name, module in model_float.named_modules():
    if isinstance(module, Linear):
        print(f"Layer: {name}")
        print(f"Weight: {module.weight}")
  
  
export_model(model_float, "ml2_jsc_weights", dataset.get_test_loader(), num_samples=100)

