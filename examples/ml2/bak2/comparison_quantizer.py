import shutil

from matplotlib import pyplot as plt
from includes_ml2 import *    

def get_weights_model(model, layer_list):

    # Iterate over model layers
    # If the layer is in the layer_list, get the weights
    weights = []
    for name, module in model.named_modules():
        if name in layer_list:
            
            # Copy to cpu and detach from graph
            weights.append(module.weight.cpu().detach().clone().flatten())
    
    return weights

def get_max_abs_values(model, layers):
        
        # Get weights as list
        weights = get_weights_model(model, layers)
        
        # Flatten weights, and convert tensor to value
        weights = [w.numpy() for w in weights]
        
        # Get Max Abs Value
        max_abs_values = [np.max(np.abs(w)) for w in weights]
        
        return max_abs_values

def explore_threshold_quantization_globally(model, layer_list, factor=0.7):
    
    # Get weights as list
    weights = get_weights_model(model, layer_list)
    
    # Flatten weights, and convert tensor to value
    weights = [w.numpy() for w in weights]
    
    # Get Max Abs Value
    max_abs_value = max([np.max(np.abs(w)) for w in weights])
    
    print(f"Max Abs Value: {max_abs_value}")    
    print(f"Threshold: {max_abs_value * factor} at {factor * 100}%")
    
    max_abs_values = get_max_abs_values(model, layer_list)

    print("\n")
    print("Max Abs Values:")
    for max_val, layer in zip(max_abs_values, layer_list):
        print(f"|> Layer: {layer} - Max Abs Value: {max_val}")


if __name__ == "__main__":
    
    processing_sequence = ["dense1", "dense2", "dense3", "dense4", "dense5"]
    #processing_sequence = ["dense2", "dense3", "dense4", "dense1", "dense5"]

    base_quant_model_config = f"{parent_directory}/examples/ml2/jsc_xl_quant_base_config.yaml"
    base_quant_model_config_threshold = f"{parent_directory}/examples/ml2/jsc_xl_quant_base_config_threshold.yaml"
    
    float_model_weight_path = f"{parent_directory}/train/jsc_xl/run_25/best_weights.pth"
    float_model_config_path = f"{parent_directory}/configs/jsc/jsc_xl.yaml"
    
    float_model = ModelBuilder().build(ModelTypes.JSC, config=float_model_config_path, weights_path=float_model_weight_path)
    quant_model = ModelBuilder().build(ModelTypes.JSC, config=base_quant_model_config, weights_path=float_model_weight_path)

    dataset = DatasetBuilder().build(DatasetTypes.JSC, config=float_model_config_path)
    validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())
    
    acc_float_model,_ = validator.validate(float_model)
    acc_quant_model,_ = validator.validate(quant_model)
    
    acc_float_model = round(acc_float_model, 2)
    acc_quant_model = round(acc_quant_model, 2)
    
    # Clear terminal
    print("\033[H\033[J")
    
    print("|----------------------------------------------------")
    print("| Results: ")
    print("|----------------------------------------------------")
    print(f"| Accuracy Float Model: {acc_float_model}%")
    print(f"| Accuracy Quant Model: {acc_quant_model}%")
    print("|----------------------------------------------------\n")
    
    explore_threshold_quantization_globally(float_model, processing_sequence)