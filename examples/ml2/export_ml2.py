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
    
    model_dict["numLayers"] = 6
    
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
            
            weights = module.quant_weight()[0].cpu().detach().numpy()
            biases = module.quant_bias()[0].cpu().detach().numpy()
            #Change shape from (output, input) to (input, output)
            
            weights = np.transpose(weights)
            
            # Reshape biases in n x 1
            biases = biases.reshape(-1, 1)
                        
            name = name.replace("dense", "layer")
            model_dict[name] = {
                "layerId": index,
                "numNeurons": module.weight.shape[0],
                "activationFunction": "ReLU",
                "weights":weights.tolist(),
                "bias": biases.tolist(),
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
    with open(f"{output_path}", "w") as f:
        f.write(model_json)
        
def export_configfile(model, model_config, output_path, dataset):
    
    layers = ["dense1", "dense2", "dense3", "dense4", "dense5"]
    bias_bitwidth = [8 for x in range(0, 5)]
    activation_bitwidth = [8 for x in range(0, 5)]
    weight_bitwidth = []
    
    msb_act_per_layer = [0 for x in range(0, 5)]
    lsb_act_per_layer = [-8 for x in range(0, 5)]
    
    msb_weights_per_layer = []
    lsb_weights_per_layer = []
    
    msb_bias_per_layer = []
    lsb_bias_per_layer = []
    
    # Iterate over the dataset and get the max value for the input
    for inputs, targets in dataset:
        print(f"{inputs.shape=}")
        max_input = torch.max(torch.abs(inputs))
        print(f"{max_input=}")
            
    # Get MSB by calculate the log2 of the highest absolute value round up
    msb_input = np.ceil(np.log2(max_input.cpu().detach().numpy()))
    print(f"{msb_input=}")
    
    # Read model config as yaml 
    with open(model_config, "r") as f:
        config_file = f.read()
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        
    for layer in config["backbone"]:
        if layer[2] == "QuantLinear":
            weight_bitwidth.append(layer[4]["weight_bit_width"])
            

    print(f"Layers: {layers}")
    print(f"Weight bitwidth: {weight_bitwidth}")
    print(f"Bias bitwidth: {bias_bitwidth}")
    print(f"Activation bitwidth: {activation_bitwidth}")
    print("---------")
    print("---------")
    

    index = 0
    # Iterate over layers and print weights
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            try: 
                print(f"Layer: {name}")
                print(f"Weight: {module.quant_weight()[0].shape}")

                #visualize_layer_differences(name, module.quant_weight()[0].detach().numpy(), module.weight.detach().numpy())
                
                # Get highest absolute value in weight matrix
                max_weight = torch.max(torch.abs(module.quant_weight()[0]))
                print(f"{max_weight=}")                
                # Get MSB by calculate the log2 of the highest absolute value round up
                msb_weight = np.ceil(np.log2(max_weight.cpu().detach().numpy()))
                print(f"{msb_weight=}")                
                # Get highest absolute value in bias matrix
                max_bias = torch.max(torch.abs(module.quant_bias()[0]))
                print(f"{max_bias=}")
        
                msb_bias = np.ceil(np.log2(max_bias.cpu().detach().numpy()))
                print(f"{msb_bias=}")
                print("---------")
                # Get highest absolute value in activation matrix
                # max_activation = torch.max(torch.abs(module.quant_activation()[0]))
                
                msb_weights_per_layer.append(msb_weight)
                msb_bias_per_layer.append(msb_bias)
                
                lsb_weights_per_layer.append(msb_weight - weight_bitwidth[index] + 1)
                lsb_bias_per_layer.append(msb_bias - bias_bitwidth[index] + 1)
                
                index += 1
            except Exception as e:
                print(e)
    print("---------")
    print("Weights")
    for msb_weight, lsb_weight in zip(msb_weights_per_layer, lsb_weights_per_layer):
        print(f"MSB: {msb_weight}, LSB: {lsb_weight}")
    
    print("---------")
    print("Bias")
    for msb_bias, lsb_bias in zip(msb_bias_per_layer, lsb_bias_per_layer):
        print(f"MSB: {msb_bias}, LSB: {lsb_bias}")

    print(f"{msb_input=}")

    def create_json_file(output_path):
        try: 
            data = {
                "numLayers": 6,
                "lsbOut": -8,
                "layer0": {
                    "layerId": 0,
                    "msbIn": 1,
                    "lsbIn": -8,
                    "msbWeights": 0,
                    "lsbWeights": 0,
                    "method": "generic"
                }
            }
                        
            for i, (msbAct, lsbAct, msbWeight, lsbWeight, msbBias, lsbBias) in enumerate(zip(msb_act_per_layer, lsb_act_per_layer, msb_weights_per_layer, lsb_weights_per_layer, msb_bias_per_layer, lsb_bias_per_layer), start=1):
                data[f"layer{i}"] = {
                "layerId": i,
                "msbIn": int(msbAct),
                "lsbIn": int(lsbAct),
                "lsbInt": -8,
                "msbWeights": int(msbWeight),
                "lsbWeights": int(lsbWeight),
                "msbBias": int(msbBias),
                "lsbBias": int(lsbBias),
                "method": "generic"
            }

            with open(output_path, "w") as f:
                print(f"Writing to {output_path}")
                json.dump(data, f, indent=4)
        except Exception as e:
            print(json.dumps(data, indent=4))
            print(e)

    create_json_file(output_path)
            
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
float_model_config = "configs/jsc/jsc_xl.yaml"

model_acc_drop = f"50.0"

quant_model_config = f"tmp_data/quant_first/2_pruning/config.yaml"
quant_model_weight_path = f"tmp_data/quant_first/3_retrain/best_weights.pth"

dataset = DatasetBuilder().build(DatasetTypes.JSC, config=quant_model_config)
quant_model = ModelBuilder().build(ModelTypes.JSC, config=quant_model_config, weights_path=quant_model_weight_path)

validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())

state_dict = copy.deepcopy(quant_model.state_dict())

# Set cache for bias
for name, module in quant_model.named_modules():
    if isinstance(module, QuantLinear):
        module.cache_inference_quant_bias = True
        
# Run inference
#model_float = ModelBuilder().build(ModelTypes.JSC, config=float_model_config, weights_path=float_model_weight_path)

acc, loss = validator.validate(quant_model)
#acc, loss = validator.validate(model_float)

# Iterate over layers and print weights
for name, module in quant_model.named_modules():
    if isinstance(module, QuantLinear):
        try: 
            print(f"Layer: {name}")
            print(f"Weight: {module.quant_weight()[0].shape}")
            
            # Print shape of bias
            print(f"Bias: {module.quant_bias()[0].shape}")
            
            # Print number of parameters
            print(f"Number of parameters: {module.quant_weight()[0].shape[0] * module.quant_weight()[0].shape[1]}")

            #visualize_layer_differences(name, module.quant_weight()[0].detach().numpy(), module.weight.detach().numpy())
            state_dict[name + ".weight"] = module.quant_weight()[0]
            state_dict[name + ".bias"] = module.quant_bias()[0]
            
            #print(f"Weight: {module.quant_weight()[0]}")
            #print(f"Bias: {module.quant_bias()[0]}")
            
            # Print Shape of the layer
            #print(f"Weight: {module.quant_weight()[0].shape}")
  
        except Exception as e:
            print(name)
            print(e)
  
export_model(quant_model, "ml2_jsc_weights.json", dataset.get_test_loader(), num_samples=100)
#export_configfile(quant_model, quant_model_config, "ml2_jsc_config.json", dataset.get_test_loader())
