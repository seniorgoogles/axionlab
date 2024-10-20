import json
import shutil
from includes_ml2 import *
import numpy as np
import matplotlib.pyplot as plt

def plot_sensitivity_by_key(data, title, xlabel, ylabel, save_path, flip_x_axis=False):
        
    # Plot each dense layer's data as a separate line
    for key in data:
        x = list(map(float, data[key].keys()))
            
        y = list(data[key].values())
        plt.plot(x, y, label=key)

    if flip_x_axis:
        plt.gca().invert_xaxis()
        
    # Adding labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    # Save the plot
    plt.savefig(save_path)
    plt.close()
    
def plot_acc_vs_sparsity(x, y, title, xlabel, ylabel, save_path, flip_x_axis=False):
        
    # Plot each dense layer's data as a separate line
    plt.plot(x, y, label="Accuracy vs Sparsity")

    if flip_x_axis:
        plt.gca().invert_xaxis()
        
    # Adding labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    # Save the plot
    plt.savefig(save_path)
    plt.close()

def eval_sensitivity_by_key(model_config_path, weights_path, layer_str_list, key, value_range):
    
    result = {}
    
    dataset = DatasetBuilder.build(DatasetTypes.JSC, config=quant_model_config)
    validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())
    
    model = ModelBuilder().build(ModelTypes.JSC, config=model_config_path, weights_path=weights_path)
    baseline_acc, _ = validator.validate(model)
        
    print(f"Baseline accuracy: {baseline_acc}")
    
    
    for layer_str in layer_str_list:
        
        result[layer_str] = dict()
        
        for val in value_range:
            
            model_config = set_config_value(model_config_path, layer_str, key, val)
            model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=weights_path)
            
            acc, _ = validator.validate(model)
            result[layer_str][val] = acc
        
        # Set back to default value
        model_config = set_config_value(model_config, layer_str, key, value_range[0])

        model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=weights_path)
        acc, _ = validator.validate(model) 
        
    return result

def evaluate_sparsity_max_acc_drop(model_config_path, weights_path, layer_str_list, key, value_range, max_acc_drop):

    result = {}
    
    dataset = DatasetBuilder.build(DatasetTypes.JSC, config=quant_model_config)
    validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())
    
    model = ModelBuilder().build(ModelTypes.JSC, config=model_config_path, weights_path=weights_path)
    baseline_acc, _ = validator.validate(model)
    
    prev_model_sparsity = get_sparsity_overview(model, layer_str_list)
        
    print(f"Baseline accuracy: {baseline_acc}")
    
    model_config = load_config(model_config_path)

    for layer_str in layer_str_list:
        
        result[layer_str] = dict()
        reload_index = 0
        
        for index, val in enumerate(value_range):
            
            model_config = set_config_value(model_config, layer_str, key, val)
            model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=weights_path)
            
            acc, _ = validator.validate(model)
            
            if baseline_acc - acc > max_acc_drop:
                reload_index = index - 1
                print(f"Reload index: {reload_index}")
                break
            else:
                reload_index = index
                result[layer_str][val] = acc

        
        # Set back to previous "good" model configuration 
        model_config = set_config_value(model_config, layer_str, key, value_range[reload_index])

        model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=weights_path)
        acc, _ = validator.validate(model) 
        
    model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=weights_path)
    acc, _ = validator.validate(model)         
    curr_model_sparsity = get_sparsity_overview(model, layer_str_list)

        
        
    return result, model_config, prev_model_sparsity, curr_model_sparsity, acc

def evaluate_sparsity_max_acc_max_min_params(model_config_path, weights_path, layer_str_list, key, value_range, max_acc_drop):
    pass
     

def evaluate_sparsity_min_max_params(model_config_path, weights_path, layer_str_list, key, value_range, max_acc_drop):
    pass


if __name__ == "__main__":

    
    checkpoint = 331 

    quant_model_config = f"{parent_directory}/configs/jsc/quant_jsc_xl.yaml"
    best_weights = f"/home/mmecik/repositories/synapselab/train/jsc_xl_floating_point/run_{checkpoint}/best_weights.pth"
    
    # Layers
    layer_str_list = ["dense1", "dense2", "dense3", "dense4", "dense5"]

    # Sparsity
    sparsity_range = np.arange(0.0, 1.0, 0.1)   
    
    # Weights 
    weights_range = range(8, 1, -1)
    
    model = ModelBuilder().build(ModelTypes.JSC, config=quant_model_config, weights_path=best_weights)
    sorted_layer_str_list = sort_layers_by_param_num(model, layer_str_list)
    
    ''' Intepentend Sensitivity '''
    # Evaluate the sensitivity of the model to weight word width
    #results = eval_sensitivity_by_key(quant_model_config, best_weights, layer_str_list, "weight_bit_width", weights_range)
    #plot_sensitivity_by_key(results, f"Sensitivity Weight Bit Width", "Bit Width", "Accuracy", f"Sensitivity_Weight_Bit_Width.png", flip_x_axis=True)
    
    # Evaluate the sensitivity of the model to weight word width
    #results = eval_sensitivity_by_key(quant_model_config, best_weights, layer_str_list, "output_bit_width", weights_range)
    #plot_sensitivity_by_key(results, f"Sensitivity Activation Bit Width", "Bit Width", "Accuracy", f"Sensitivity_Activation_Bit_Width.png", flip_x_axis=True)
    
  
    # Evaluate the sensitivity of the model to weight sparsity
    #results = eval_sensitivity_by_key(quant_model_config, best_weights, layer_str_list, "weight_sparse_eps", sparsity_range)
   
    #plot_sensitivity_by_key(results, f"Sensitivity Pruning", "Sparsity Level", "Accuracy", f"Sensitivity_Pruning.png")
    
    ''' Depentent Sensitivity with max allowed acc drop '''
    #results, model_config, _, _ = evaluate_sparsity_max_acc_drop(quant_model_config, best_weights, layer_str_list, "weight_bit_width", weights_range, 3.0)
    #plot_sensitivity_by_key(results, f"Sensitivity Weight Quant Max Acc Drop 3.0%", "Bit_Width", "Accuracy", f"Sensitivity_Weight_Quant_Max_Drop_3.png", flip_x_axis=True)
    
    #results, model_config, _, _ = evaluate_sparsity_max_acc_drop(quant_model_config, best_weights, layer_str_list, "output_bit_width", weights_range, 3.0)
    #plot_sensitivity_by_key(results, f"Sensitivity Act Quant Max Acc Drop 3.0%", "Bit-Width", "Accuracy", f"Sensitivity_Act_Quant_Max_Drop_3.png", flip_x_axis=True)
    
    # Sorted by max-min params
    #results, model_config, _, _ = evaluate_sparsity_max_acc_drop(quant_model_config, best_weights, sorted_layer_str_list, "weight_bit_width", weights_range, 3.0)
    #plot_sensitivity_by_key(results, f"Sensitivity Weight Quant Max Acc Drop 3.0% Sorted Max-Min", "Bit_Width", "Accuracy", f"Sensitivity_Weight_Quant_Max_Drop_3_Sorted.png", flip_x_axis=True)
    
    #results, model_config, _, _ = evaluate_sparsity_max_acc_drop(quant_model_config, best_weights, sorted_layer_str_list, "output_bit_width", weights_range, 3.0)
    #plot_sensitivity_by_key(results, f"Sensitivity Act Quant Max Acc Drop 3.0% Sorted Max-Min", "Bit-Width", "Accuracy", f"Sensitivity_Act_Quant_Max_Drop_3_Sorted.png", flip_x_axis=True)
    """
    results, model_config, prev_model_sparsity, current_model_sparsity = evaluate_sparsity_max_acc_drop(quant_model_config, best_weights, layer_str_list, "weight_sparse_eps", sparsity_range, 3.0)
    data = json.loads(current_model_sparsity)
    total_weights = data['total']['num_weights']
    total_zero_weights = data['total']['num_zero_weights']
    total_zero_weight_percentage = (total_zero_weights / total_weights) * 100
    plot_sensitivity_by_key(results, f"Sens. Prun. Max Acc Drop 3.0% -> {total_zero_weight_percentage:.2f}% Sparsity", "Sparsity Level", "Accuracy", f"Sensitivity_Pruning_Max_Drop_3.png")

    results, model_config, prev_model_sparsity, current_model_sparsity = evaluate_sparsity_max_acc_drop(quant_model_config, best_weights, layer_str_list, "weight_sparse_eps", sparsity_range, 5.0)
    data = json.loads(current_model_sparsity)
    total_weights = data['total']['num_weights']
    total_zero_weights = data['total']['num_zero_weights']
    total_zero_weight_percentage = (total_zero_weights / total_weights) * 100
    plot_sensitivity_by_key(results, f"Sens. Prun. Max Acc Drop 5.0% -> {total_zero_weight_percentage:.2f}% Sparsity", "Sparsity Level", "Accuracy", f"Sensitivity_Pruning_Max_Drop_5.png")
    
    """
    model = ModelBuilder().build(ModelTypes.JSC, config=quant_model_config, weights_path=best_weights)
    sorted_layer_str_list = sort_layers_by_param_num(model, layer_str_list)
    
    x_data = []
    y_data = []
    allowed_acc_drop = []
    
    # make dir tmp_data
    if os.path.exists("tmp_data"):
        shutil.rmtree("tmp_data")
        
    os.mkdir("tmp_data")
    os.makedirs(f"tmp_data/configs/jsc")
    
    for max_allowed_acc_drop in np.arange(0.0, 50.25, 0.25):
        
        # round to two decimal places
        max_allowed_acc_drop = round(max_allowed_acc_drop, 2)
        
        results, model_config, prev_model_sparsity, current_model_sparsity, acc = evaluate_sparsity_max_acc_drop(quant_model_config, best_weights, sorted_layer_str_list, "weight_sparse_eps", sparsity_range, max_allowed_acc_drop)
        data = json.loads(current_model_sparsity)
        total_weights = data['total']['num_weights']
        total_zero_weights = data['total']['num_zero_weights']
        total_zero_weight_percentage = (total_zero_weights / total_weights) * 100
        plot_sensitivity_by_key(results, f"Sens. Prun. Max Acc Drop {max_allowed_acc_drop}% Sorted Max-Min -> {total_zero_weight_percentage:.2f}% Sparsity", "Sparsity Level", "Accuracy", f"tmp_data/Sensitivity_Pruning_Max_Drop_{max_allowed_acc_drop}_Sorted_Max_Min.png")
        
        x_data.append(total_zero_weight_percentage)
        y_data.append(acc)
        allowed_acc_drop.append(max_allowed_acc_drop)
        
        # Write model_config to file
        ConfigurationManager(model_config).write(f"tmp_data/configs/jsc/quant_jsc_xl_acc_drop_{max_allowed_acc_drop}.yaml")
        
    print(f"{x_data=}")
    print(f"{y_data=}")
    print(f"{allowed_acc_drop=}")
    
    # Write to file 
    with open("tmp_data/acc_vs_sparsity_jsc_xl.json", "w") as f:
        json.dump({"sparsity": x_data, "accuracy": y_data, "allowed_acc_drop": allowed_acc_drop}, f)
    
    plot_acc_vs_sparsity(x_data, allowed_acc_drop, "Acc vs Sparsity", "Sparsity Level", "Accuracy", "tmp_data/acc_vs_sparsity_jsc_xl.png")

    """
    max_allowed_acc_drop = 0.5
    results, model_config, prev_model_sparsity, current_model_sparsity = evaluate_sparsity_max_acc_drop(quant_model_config, best_weights, sorted_layer_str_list, "weight_sparse_eps", sparsity_range, 3.0)
    data = json.loads(current_model_sparsity)
    total_weights = data['total']['num_weights']
    total_zero_weights = data['total']['num_zero_weights']
    total_zero_weight_percentage = (total_zero_weights / total_weights) * 100
    plot_sensitivity_by_key(results, f"Sens. Prun. Max Acc Drop {max_allowed_acc_drop}% Sorted Max-Min -> {total_zero_weight_percentage:.2f}% Sparsity", "Sparsity Level", "Accuracy", f"Sensitivity_Pruning_Max_Drop_{max_allowed_acc_drop}_Sorted_Max_Min.png")
    
    max_allowed_acc_drop = 0.1
    results, model_config, prev_model_sparsity, current_model_sparsity = evaluate_sparsity_max_acc_drop(quant_model_config, best_weights, sorted_layer_str_list, "weight_sparse_eps", sparsity_range, 5.0)
    data = json.loads(current_model_sparsity)
    total_weights = data['total']['num_weights']
    total_zero_weights = data['total']['num_zero_weights']
    total_zero_weight_percentage = (total_zero_weights / total_weights) * 100
    plot_sensitivity_by_key(results, f"Sens. Prun. Max Acc Drop {max_allowed_acc_drop}% Sorted Max-Min -> {total_zero_weight_percentage:.2f}% Sparsity", "Sparsity Level", "Accuracy", f"Sensitivity_Pruning_Max_Drop_{max_allowed_acc_drop}_Sorted_Max_Min.png")
    """
    #def plot_acc_vs_sparsity(x, y, title, xlabel, ylabel, save_path, flip_x_axis=False):
