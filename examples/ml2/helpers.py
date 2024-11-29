import copy
import shutil
import os 
import yaml
import torch
from torchviz import make_dot
import json

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def set_config_value(model_config, config, layer_str, key, value, logger):
    layers = None

    # If no model configuration is provided, load the configuration from the file
    if model_config is None:
        model_config = load_config(config)
        layers = model_config['backbone']

    # If a model configuration is provided, use the provided configuration
    else:
        layers = model_config['backbone']

    # Go through all layers and check if the layer is a dense layer
    for index, layer in enumerate(layers):
        layer_name = layer[3]

        # Go through all layers and check if the layer is a dense layer
        if layer_str in layer_name:
            # Set the value of the key to the new value
            if isinstance(layer[-1], dict) and key in layer[-1]:

                logger.debug(f"Setting {key} to {value} for {layer_name}")

                layer[-1][key] = value

    return model_config

def get_zero_params_per_layer(model, logger=None):
    layers = []
    zero_params = []
    params = dict()

    # Iterate over model and get the parameters
    for name, param in model.named_parameters():
        layer_name = name.split('.')[0]

        if layer_name not in params:
            params[layer_name] = dict()
        params[layer_name][name.split('.')[1]] = torch.sum(param == 0).item()

        layers.append(layer_name)
        zero_params.append(torch.sum(param == 0).item())

    if logger is not None:
        logger.debug(f"Layers: {layers}")
        logger.debug(f"Zero params: {zero_params}")

    return layers, zero_params

def get_params_count_per_layer(model, logger=None):
    layers = []
    params_count = []
    params = dict()

    # Iterate over model and get the parameters
    for name, param in model.named_parameters():
        layer_name = name.split('.')[0]

        if layer_name not in params:
            params[layer_name] = dict()
        params[layer_name][name.split('.')[1]] = len(param)

        layers.append(layer_name)
        params_count.append(len(param))

    if logger is not None:
        logger.debug(f"Layers: {layers}")
        logger.debug(f"Params: {params_count}")

    return layers, params_count

def get_weights_per_layer(model, layer_str):
    
    layer = getattr(model, layer_str)
    num_weights = layer.quant_weight().tensor.numel()            
            
    return num_weights 

def get_weights_zero_value_per_layer(model, layer_str):

    layer = getattr(model, layer_str)
    num_zero_weights = torch.sum(layer.quant_weight().tensor == 0).item()
            
    return num_zero_weights

def plot_model(model, logger, input):

    y = model(input)
    g = make_dot(y, params=dict(model.named_parameters()))
    g.render('model', format='png')


def clear_folder(folder):
    try:
        shutil.rmtree(folder)
    except Exception as ex:
        print(ex)

def train_model(dataset, trainer, validator, weights_path, model, lr, epochs, results_file_path, reload_weights, reload_weights_acc, logger):
        
    lr_list = []
    current_acc = 0.0
    previous_acc = 0.0
    
    # Make a deep copy of the model state
    bak_model = copy.deepcopy(model.state_dict())
    
    # Load weights if provided
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path,  weights_only=False), strict=False)
        bak_model = copy.deepcopy(model.state_dict())
        previous_acc, loss = validator.validate(model)
        
    # If learning rate is not a list, convert it to a list
    if not isinstance(lr, list):
        lr_list = [lr]
    else:
        lr_list = lr
        
    print(f"{lr_list=}")
    
    for lr in lr_list:
        
        logger.info(f"Training model with learning rate {lr}")
        
        if reload_weights:
            model.load_state_dict(bak_model)
        
        # If current acc is better than the previous acc, deep copy the model state
        if current_acc > previous_acc:
            bak_model = copy.deepcopy(model.state_dict())
        else:
            logger.info(f"Previous acc: {previous_acc} Current acc: {current_acc}")
            previous_acc = current_acc
            
        optimizer = torch.optim.Adam(model.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=14)
        
        
        
        trainer.train(model, None, dataset, torch.nn.CrossEntropyLoss(), optimizer, lr, epochs, 200, scheduler)
        
        # Validate the model
        current_acc, loss = validator.validate(model)
        
        logger.info(f"Accuracy: {current_acc}")
        
        # Get the last run the folder
        run = max([int(f.split('_')[-1]) for f in os.listdir(f"train/{model.name}") if os.path.isdir(os.path.join(f"train/{model.name}", f))])
        
        with open(results_file_path, "a") as f:
            f.write(f"{run}\t{lr}= {current_acc=}\n")

def train_student_teacher(dataset, trainer, validator, teacher, student, lr, epochs, logger):
        
    lr_list = []
    current_acc = 0.0
    previous_acc = 0.0
    
    # Make a deep copy of the model state
    bak_model = copy.deepcopy(student.state_dict())
    
    # If learning rate is not a list, convert it to a list
    if not isinstance(lr, list):
        lr_list = [lr]
    else:
        lr_list = lr
        
    print(f"{lr_list=}")
    
    for lr in lr_list:
        
        logger.info(f"Training model with learning rate {lr}")
        
        # If current acc is better than the previous acc, deep copy the model state
        #if current_acc > previous_acc:
        #    bak_model = copy.deepcopy(student.state_dict())
        #else:
        #    logger.info(f"Previous acc: {previous_acc} Current acc: {current_acc}")
        #    previous_acc = current_acc
        
        trainer.train_teacher_student(teacher, student, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(student.parameters(), lr), epochs, T=2, teacher_is_pretrained=True)
        
        # Validate the model
        current_acc, loss = validator.validate(student)
        
        logger.info(f"Accuracy: {current_acc}")
        
        # Get the last run the folder
        run = max([int(f.split('_')[-1]) for f in os.listdir(f"train/{student.name}") if os.path.isdir(os.path.join(f"train/{student.name}", f))])
        
        with open("result_lr_acc.txt", "a") as f:
            f.write(f"{run}\t{lr}= {current_acc=}\n")
            
def set_config_value(model_config, layer_str, key, value):
    layers = None
    
    if isinstance(model_config, str):
        model_config = load_config(model_config)
        layers = model_config['backbone']

    # If a model configuration is provided, use the provided configuration
    else:
        layers = model_config['backbone']

    # Go through all layers and check if the layer is a dense layer
    for index, layer in enumerate(layers):
        layer_name = layer[3]

        # Go through all layers and check if the layer is a dense layer
        if layer_str in layer_name:
            # Set the value of the key to the new value
            if isinstance(layer[-1], dict) and key in layer[-1]:

                print(f"Setting {key} to {value} for {layer_name}")
                layer[-1][key] = value
            else:
                print(f"Layer {layer_name} does not have a {key} key")

    return model_config

def get_config_value(model_config, layer_str, key):
    
    layers = None
    ret_val = None
    
    if isinstance(model_config, str):
        model_config = load_config(model_config)
        layers = model_config['backbone']

    # If a model configuration is provided, use the provided configuration
    else:
        layers = model_config['backbone']

    # Go through all layers and check if the layer is a dense layer
    for index, layer in enumerate(layers):
        layer_name = layer[3]

        # Go through all layers and check if the layer is a dense layer
        if layer_str in layer_name:
            # Set the value of the key to the new value
            if isinstance(layer[-1], dict) and key in layer[-1]:
                ret_val = layer[-1][key]
                print(f"Getting {key} to {ret_val} for {layer_name}")
            else:
                print(f"Layer {layer_name} does not have a {key} key")

    return ret_val

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def get_sparsity_overview(model, layer_str_list):
    sparsity_overview = dict()
    sum_weights = 0
    sum_zero_weights = 0
    
    for layer_str in layer_str_list:
        layer = getattr(model, layer_str)

        num_weights = layer.quant_weight().tensor.numel()
        num_zero_weights = torch.sum(layer.quant_weight().tensor == 0).item()
        
        #print(layer.quant_weight())
        
        # Calculate percentage of zero weights
        zero_weight_percentage = num_zero_weights / num_weights * 100
        
        sparsity_overview[layer_str] = {"num_weights": num_weights, "num_zero_weights": num_zero_weights, "zero_weight_percentage": zero_weight_percentage}
        
        #print(f"Layer: {layer_str} {num_weights=} {num_zero_weights=} {zero_weight_percentage} %")
        
        sum_weights += num_weights
        sum_zero_weights += num_zero_weights

        # Get the number of zero weights
        # Print the number of zero weights
        # Print the number of weights
        # Print the percentage of zero weights
        # Print the percentage of zero weights in the layer
        
    sparsity_overview["total"] = {"num_weights": sum_weights, "num_zero_weights": sum_zero_weights, "zero_weight_percentage": sum_zero_weights / sum_weights * 100}

    return sparsity_overview

def get_model_sparsity(model, layer_str_list):
    sparsity_overview = get_sparsity_overview(model, layer_str_list)
    
    return sparsity_overview["total"]["zero_weight_percentage"]

def get_model_param_num(model, layers=["dense1", "dense2", "dense3", "dense4", "dense5"]):

    params_count_per_layer = list()
    
    for layer in layers:
        params_count_per_layer.append(get_weights_per_layer(model, layer))
        
    return layers, params_count_per_layer

def get_model_zero_param_num(model, layers=["dense1", "dense2", "dense3", "dense4", "dense5"]):
    
        zero_params_per_layer = list()
        
        for layer in layers:
            zero_params_per_layer.append(get_weights_zero_value_per_layer(model, layer))
            
        return layers, zero_params_per_layer

def sort_layers_by_param_num(model, layer_str_list):
    
    layers = []
    params_count = []
    params = dict()
    
    for layer_str in layer_str_list:
        layer = getattr(model, layer_str)
        params_count.append(layer.quant_weight().tensor.numel())

    # Sort the layers by the number of parameters
    sorted_layers = [x for _, x in sorted(zip(params_count, layer_str_list), key=lambda pair: pair[0], reverse=True)]
    
    # Get the number of parameters for each layer to list
    sorted_params_count = [x for x in sorted(params_count, reverse=True)]
    
    return sorted_layers, sorted_params_count