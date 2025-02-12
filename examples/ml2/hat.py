import json
import shutil
from includes_ml2 import *
import numpy as np
from helpers import clear_folder, train_model, print_colored_window, get_model_sparsity, set_config_value

layer_str_list = ["dense1", "dense2", "dense3", "dense4", "dense5"]

float_model_config = f"/home/fry/Documents/repositories/synapselab/examples/ml2/float_model/config.yaml"

quant_model_config = f"/home/fry/Documents/repositories/synapselab/qat/quant/config.yaml"
quant_shiftadd_model_config = f"/home/fry/Documents/repositories/synapselab/examples/ml2/quant_base_model/jsc_base_quant_shiftadd_config.yaml"


quant_weights = "/home/fry/Documents/repositories/synapselab/qat/quant/best_weights.pth"


dataset = DatasetBuilder.build(DatasetTypes.JSC, config=float_model_config)
validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())

# Initialize objects
trainer = Trainer()
dataset = DatasetBuilder.build(DatasetTypes.JSC, config=quant_shiftadd_model_config)
validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())

# Validate quant model
quant_model = ModelBuilder().build(ModelTypes.JSC, config=quant_model_config, weights_path=quant_weights)
shiftadd_model = ModelBuilder().build(ModelTypes.JSC, config=quant_shiftadd_model_config, weights_path=quant_weights)

acc, _ = validator.validate(quant_model)
shiftadd_acc, _ = validator.validate(shiftadd_model)

# Clear console
os.system('clear')

# Print accuracies
print(f"Quant model accuracy: {acc} ShiftAdd model accuracy: {shiftadd_acc}")

input("Press Enter to continue...")


train_model(dataset, trainer, shiftadd_model, 0.0003923, 70, save_model=False, save_dir=None)                 

acc, _ = validator.validate(shiftadd_model)
os.system('clear')


# Write weights of every dense layer to file as string
for layer_str in layer_str_list:
    
    # Get layer by name
    
    layer = getattr(shiftadd_model, layer_str)
    weights = layer.quant_weight()
    #print(weights)
    
    # If weightts folder exists, clear it and create it
    if os.path.exists("weights"):
        clear_folder("weights")

    os.makedirs("weights")

    # Clear

    with open(f"weights/{layer_str}_adder_aware.txt", "w") as f:
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