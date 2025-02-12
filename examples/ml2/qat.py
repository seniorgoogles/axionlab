import json
import shutil
from includes_ml2 import *
import numpy as np
from helpers import clear_folder, train_model, print_colored_window, get_model_sparsity, set_config_value

def evaluate_sensitivity(model, base_config, weights_path, validator, layer_str_list):
    # Get the base model accuracy
    base_acc, _ = validator.validate(model)

    # Get the base model sparsity
    base_sparsity = get_model_sparsity(model, layer_str_list)
    
    # Go through all layers and set the sparsity to 1.0 from 0.0 by 0.1
    for layer_str in layer_str_list:
        
        config = base_config

        for i in np.arange(0.0, 1.0, 0.1):
            # Set the sparsity
            config = set_config_value(config, layer_str, "weight_threshold", i)
            config = set_config_value(config, layer_str, "weight_enable_prune", True)

            # Build the model
            model = ModelBuilder().build(ModelTypes.JSC, config=config, weights_path=weights_path)

            # Get the accuracy
            acc, _ = validator.validate(model)
            sparsity = get_model_sparsity(model, layer_str_list)

            print(f"Layer: {layer_str} Sparsity: {sparsity} Accuracy: {acc}")

        # Reload model
        model = ModelBuilder().build(ModelTypes.JSC, config=base_config, weights_path=weights_path)


float_model_config = f"/home/fry/Documents/repositories/synapselab/examples/ml2/float_model/config.yaml"
quant_model_config = f"/home/fry/Documents/repositories/synapselab/examples/ml2/quant_base_model/jsc_base_quant_config.yaml"

float_model_weight_path = f"/home/fry/Documents/repositories/synapselab/examples/ml2/float_model/best_weights.pth"

dataset = DatasetBuilder.build(DatasetTypes.JSC, config=float_model_config)
validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())

model = ModelBuilder().build(ModelTypes.JSC, config=float_model_config, weights_path=float_model_weight_path)
quant_model = ModelBuilder().build(ModelTypes.JSC, config=quant_model_config, weights_path=float_model_weight_path)

acc, _ = validator.validate(model)
quant_acc, _ = validator.validate(quant_model)

# Clear console
os.system('clear')

# Print accuracies
print(f"Float model accuracy: {acc}")
print(f"Quant model accuracy: {quant_acc}")

layer_str_list = ["dense1", "dense2", "dense3", "dense4", "dense5"]



##################################################################################################
## PRUNING
##################################################################################################
# Enable sparsity
quant_model_config = set_config_value(quant_model_config, "dense", "weight_enable_prune", True)
quant_model = ModelBuilder().build(ModelTypes.JSC, config=quant_model_config, weights_path=float_model_weight_path)


os.system('clear')
quant_acc, _ = validator.validate(quant_model)

print(f"Quant model accuracy with sparsity: {quant_acc}")


# Initialize objects
trainer = Trainer()

# If folder exists delete it and recreate it
if os.path.exists("qat"):
    clear_folder("qat")

os.makedirs("qat")
os.makedirs("qat/pruned")
os.makedirs("qat/quant")

train_model(dataset, trainer, quant_model, 0.00087923, 70, save_model=True, save_dir="qat/pruned")                 

acc, _ = validator.validate(quant_model)
#os.system('clear')

# If weightts folder exists, clear it and create it
if os.path.exists("weights"):
    clear_folder("weights")

os.makedirs("weights")

# Write weights of every dense layer to file as string
for layer_str in layer_str_list:
    
    # Get layer by name
    
    layer = getattr(quant_model, layer_str)
    weights = layer.quant_weight()
    #print(weights)
    


    with open(f"weights/{layer_str}_pruned.txt", "w") as f:
        f.write(str(weights))
        
        # print zero values per layer
        zero_values = torch.sum(layer.quant_weight().tensor == 0).item()
        # all values
        all_values = layer.quant_weight().tensor.numel()
        
        # get max and min value
        max_value = torch.max(layer.quant_weight().tensor)
        min_value = torch.min(layer.quant_weight().tensor)
        
        print(f"Layer: {layer_str} Zero values: {zero_values} All values: {all_values} Max value: {max_value} Min value: {min_value}")

    print(acc)

input("Yo Bruder, willste weitermachen?")
##################################################################################################
## QUANTIZATION
##################################################################################################

# Enable sparsity
quant_model_config = set_config_value(quant_model_config, "dense", "weight_enable_quant", True)
quant_model = ModelBuilder().build(ModelTypes.JSC, config=quant_model_config, weights_path="qat/pruned/best_weights.pth")


quant_acc, _ = validator.validate(quant_model)

print(f"Quant model accuracy with sparsity: {quant_acc}")

train_model(dataset, trainer, quant_model, 0.00087923, 70, save_model=True, save_dir="qat/quant")                 

acc, _ = validator.validate(quant_model)
os.system('clear')
# Write weights of every dense layer to file as string
for layer_str in layer_str_list:
    
    # Get layer by name
    
    layer = getattr(quant_model, layer_str)
    weights = layer.quant_weight()
    
    with open(f"weights/{layer_str}_quant.txt", "w") as f:
        f.write(str(weights))
        
        # print zero values per layer
        zero_values = torch.sum(layer.quant_weight().tensor == 0).item()
        # all values
        all_values = layer.quant_weight().tensor.numel()
        
        # get max and min value
        max_value = torch.max(layer.quant_weight().tensor)
        min_value = torch.min(layer.quant_weight().tensor)
        
        print(f"Layer: {layer_str} Zero values: {zero_values} All values: {all_values} Max value: {max_value} Min value: {min_value}")

        print(acc)
        
# Save config
ConfigurationManager(quant_model_config).write("qat/quant/config.yaml")