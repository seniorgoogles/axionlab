import sys
import torch
import os
import yaml
import logging 
import random
import traceback
import matplotlib.pyplot as plt
from torchviz import make_dot

from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Add the parent directory to the path so project modules can be imported
from pathlib import Path
current_script_path = Path(__file__).parent
parent_directory = current_script_path.parent
sys.path.append(str(parent_directory))

from src.datasets.dataset_builder import DatasetBuilder
from src.models.model_builder import ModelBuilder
from src.core.inject.enum import ModelTypes, DatasetTypes
from src.engine.validator import Validator
from src.engine.trainer import Trainer
from src.engine.tuner import Tuner
from src.quantizer.learned_bitwidth_quantizer import LearnedBitWidthQuantizer

configs = [
    #"configs/jsc/jsc_2l.yaml",
    #"configs/jsc/jsc_5l.yaml",
    #"configs/jsc/jsc_lite.yaml",
    #"configs/jsc/jsc_m_lite_floating_point.yaml",
    "configs/jsc/jsc_xl.yaml",
    #"configs/jsc/jsc_xl_floating_point.yaml"
]

class CustomFormatter(logging.Formatter):
    # Define format for the log messages
    FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: Fore.YELLOW + FORMAT + Style.RESET_ALL,
        logging.INFO: Fore.LIGHTCYAN_EX + FORMAT + Style.RESET_ALL,
        logging.WARNING: Fore.MAGENTA + FORMAT + Style.RESET_ALL,
        logging.ERROR: Fore.RED + FORMAT + Style.RESET_ALL,
        logging.CRITICAL: Fore.RED + Style.BRIGHT + FORMAT + Style.RESET_ALL
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    
def _plot_sensitivity(data, title, output_path):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot each dataset
    for label, values in data.items():
        ax.plot(values.keys(), values.values(), marker='o', label=label)

    # Add a legend
    ax.legend()

    # Add titles and labels
    ax.set_title(title)
    ax.set_xlabel('Bitwidth')
    ax.set_ylabel('Accuracy')

    # Add grid
    ax.grid(True)
    
    # Save to file
    plt.savefig(output_path)
    
    # Show the plot
    plt.show()


def _load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def _set_config_value(model_config, config, layer_str, key, value):
    layers = None 
    
    # If no model configuration is provided, load the configuration from the file
    if model_config is None:
        model_config = _load_config(config)
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
                
                logging.debug(f"Setting {key} to {value} for {layer_name}")
                    
                layer[-1][key] = value
                
    return model_config
                
def determine_prune_rate(model, allowed_acc_drop=0.01):

    # while the model is still within the allowed accuracy drop
    # prune the model by 1% and retrain
    # if the model is still within the allowed accuracy drop
    pass

def get_params_count_per_layer(model, logger):
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
        
    logger.debug(f"Layers: {layers}")
    logger.debug(f"Params: {params_count}")

    return layers, params_count

def determine_weight_bitwidth(configuration, allowed_acc_drop=100.0, reset_sensitivity=True, weights=None, logger=None):
    layers = [f'dense{i}' for i in range(1, 6)]
    bitwidths = [8 for i in range(1, 6)] 
    
    model_builder = ModelBuilder()
    validator = Validator()
    
    model_sensitivity = dict()
    
    # Baseline Model
    baseline_model = model_builder.build(ModelTypes.JSC, "configs/jsc/jsc_xl.yaml")
    weight_path = "/home/mmecik/repositories/synapselab/train/jsc_xl/run_4/best_weights.pth"
    baseline_model.load_state_dict(torch.load(weight_path,  weights_only=False), strict=False)
    
    # Build dataset 
    dataset = DatasetBuilder.build(DatasetTypes.JSC, config_path="configs/jsc/jsc_xl.yaml")
    
    # Run the baseline model
    baseline_accuracy, loss = validator.validate(baseline_model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

    default_weight_bitwidth = 8
    default_activation_bitwidth = 8
    default_sparse_eps = 0.0
    
    logger.debug(f"Setting the weight bitwidth to {default_weight_bitwidth}")
    logger.debug(f"Setting the activation bitwidth to {default_activation_bitwidth}")
    logger.debug(f"Disabling sparsing")
    
    # Setting the weight bitwidth to 8
    model_config = _set_config_value(None, configuration, 'dense', 'weight_bitwidth', default_weight_bitwidth)
    
    # Setting the activation bitwidth to 8
    model_config = _set_config_value(model_config, None, 'dense', 'output_bit_width', default_activation_bitwidth)
    
    # Disabling sparsing
    model_config = _set_config_value(model_config, None, 'dense', 'weight_sparse_eps', default_sparse_eps)

    #model = model_builder.build(ModelTypes.JSC, config=model_config)
    model = model_builder.build(ModelTypes.JSC, "configs/jsc/quant_jsc_xl.yaml")
    
    weight_path = "/home/mmecik/repositories/synapselab/train/jsc_xl/run_4/best_weights.pth"
    model.load_state_dict(torch.load(weight_path,  weights_only=False), strict=False)
    
    # Build dataset 
    dataset = DatasetBuilder.build(DatasetTypes.JSC, config=model_config)
    accuracy, loss = validator.validate(model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

    # Show the accuracy in the format xx.xx%
    baseline_accuracy = round(baseline_accuracy, 2)
    accuracy = round(accuracy, 2)
    
    logger.info(f"{baseline_accuracy=} {accuracy=}")
    
    # Get sensitivity of the model layerwise
    for index, layer in enumerate(layers):
        
        for bitwidth in range(bitwidths[index], 1, -1):
            
            logger.info(f"Testing {layer} with {bitwidth} bit width")
            # Set the weight bitwidth to the new value
            model_config = _set_config_value(model_config, None, layer, 'weight_bit_width', bitwidth)
            
            # Build the model
            model = model_builder.build(ModelTypes.JSC, config=model_config)
            
            # Load the weights
            model.load_state_dict(torch.load(weight_path,  weights_only=False), strict=False)
            
            # Validate the model
            accuracy, loss = validator.validate(model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())
            
            # Show the accuracy in the format xx.xx%
            accuracy = round(accuracy, 2)
            
            logger.info(f"{layer} {bitwidth} {accuracy}")

            # Add accuaracy and bit_width to sensitivity list
            if layer not in model_sensitivity:
                model_sensitivity[layer] = dict()
                            
            if accuracy < baseline_accuracy - allowed_acc_drop:
                logger.info(f"Accuracy drop of {allowed_acc_drop} exceeded for {layer} with {bitwidth} bitwidth")
                model_config = _set_config_value(None, configuration, layer, 'weight_bit_width', bitwidth+1)
                break
            else: 
                model_sensitivity[layer][bitwidth] = accuracy
            
        # Set back to default
        if reset_sensitivity:
            logger.info("Setting back to default")
            model_config = _set_config_value(None, configuration, 'dense', 'weight_bitwidth', default_weight_bitwidth)

    _plot_sensitivity(model_sensitivity, f"JSC_XL: Weight Bitwidth Sensitivity, Allowed Accuracy-Drop {allowed_acc_drop}%", "weight_bitwidth_sensitivity_with_sparsity.png")    
    
    # Save optimal quant configuration
    with open(f"configs/jsc/optimal_quant_jsc_xl_acc_drop_{allowed_acc_drop}_sparsity_{default_sparse_eps}.yaml", 'w') as f:
        yaml.dump(model_config, f)
    
    
def determine_activation_bitwidth(model, configuration, allowed_acc_drop=0.01):
    pass

def train_model(dataset, config, logger, epochs=150, learning_rate=0.001, model_weights_path=None, result_file="result_lr_acc.txt"):
    trainer = Trainer()
    validator = Validator()
    
    lr_list = []
    
    # If learning rate is not a list, convert it to a list
    # Clear file
    with open(result_file, "w") as f:
        f.write("")
    
    if not isinstance(learning_rate, list):
        lr_list.append(learning_rate)
    else:
        lr_list = learning_rate
    
    for lr in lr_list:
        
        # Build model
        model = ModelBuilder().build(ModelTypes.JSC, config_path=config, preload_weights=True)
        
        # Load the weights if provided
        if model_weights_path is not None:
            model.load_state_dict(torch.load(model_weights_path,  weights_only=False), strict=False)
        
        logger.info(f"Training model with learning rate {lr}")
        trainer.train(model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr), lr, epochs)
        # Train the model
        trainer.train(model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), learning_rate), learning_rate, epochs)
        
        # Validate the model
        accuracy, loss = validator.validate(model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())
        
        logger.info(f"Accuracy: {accuracy}")
        
        # Get the last run the folder
        run = max([int(f.split('_')[-1]) for f in os.listdir("train/jsc_xl") if os.path.isdir(os.path.join("train/jsc_xl", f))])
        
        with open(result_file, "a") as f:
            f.write(f"{run}\t{lr}= {accuracy=}\n")


# run the script
if __name__ == '__main__':
    
    # Create a logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)

    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create formatter and add it to the handler
    ch.setFormatter(CustomFormatter())
        
    # Add the handler to the logger
    logger.addHandler(ch)

    # Set the project root directory
    os.environ['PROJECT_ROOT'] = str(Path.cwd().parent)
    
    # Setup the model, dataset and trainer
    
    '''
    for layer, count in zip(layers, params_count):
        print(f"{layer} {count}")
    '''
    
    logger.info("Determining the best weight bitwidth")
    #determine_weight_bitwidth("configs/jsc/quant_jsc_xl.yaml", allowed_acc_drop=3.0, reset_sensitivity=False, logger=logger)
    
    model_builder = ModelBuilder()
    trainer = Trainer()
    validator = Validator()
    
    dataset = DatasetBuilder.build(DatasetTypes.JSC, config_path="configs/jsc/jsc_xl.yaml")
    
    epochs = 150
    min_rate = 0.0001
    max_rate = 0.01
    num_rates = 300
    
    # List with different 100 learning rates randomly generated from the range [0.0001, 0.01]
    learning_rates = [random.uniform(min_rate, max_rate) for _ in range(num_rates)]

    # Train the model with different learning rates
    '''
    result_lr_acc = dict()
    
    for lr in learning_rates:
        model = model_builder.build(ModelTypes.JSC, config_path="configs/jsc/jsc_xl.yaml", preload_weights=True)
        trainer.train(model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr), lr,epochs)
        acc, loss = validator.validate(model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())
        
        result_lr_acc[lr] = acc
        
    # Save result_lr_acc to file as textfile, but sort the result by acc
    result_lr_acc = dict(sorted(result_lr_acc.items(), key=lambda item: item[1], reverse=True))
    with open("result_lr_acc.txt", "w") as f:
        for key, value in result_lr_acc.items():
            f.write(f"{key} {value}\n")
    '''
    
    # Go through all runs in train/jsc_xl/run_* and get the best weights
    
    # Get the all folders from train/jsc_xl/run_*
    '''
    runs = [f for f in os.listdir("/home/mmecik/repositories/synapselab/train/jsc_xl/") if os.path.isdir(os.path.join("/home/mmecik/repositories/synapselab/train/jsc_xl/", f))]
    sorted_runs = sorted(runs, key=lambda x: int(x.split('_')[-1]))
    
    print(sorted_runs)
    
    results = dict()
    
    for run in sorted_runs:
        model = model_builder.build(ModelTypes.JSC, config_path="configs/jsc/jsc_xl.yaml", preload_weights=True)
        model.load_state_dict(torch.load(f"train/jsc_xl/{run}/best_weights.pth",  weights_only=False), strict=False)
        acc, loss = validator.validate(model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())     

        print(f"{run} {acc}")
        results[run] = acc
        
    
    # Sort results by acc
    results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

    # Write it to a file
    with open("results.txt", "w") as f:
        for key in results.keys():
            f.write(f"{key} {results[key]}\n")
            print(f"{key} {results[key]}")        
    '''
    run = 209
    best_weight_path = f"/home/mmecik/repositories/synapselab/train/jsc_xl/run_{run}/best_weights.pth"
    
    model = model_builder.build(ModelTypes.JSC, config_path="configs/jsc/jsc_xl.yaml", preload_weights=False)
    model.load_state_dict(torch.load(best_weight_path,  weights_only=False), strict=False)
    acc, _ = validator.validate(model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

    model = model_builder.build(ModelTypes.JSC, config_path="configs/jsc/quant_jsc_xl.yaml", preload_weights=False)
    model.load_state_dict(torch.load(best_weight_path,  weights_only=False), strict=False)
    quant_acc, _ = validator.validate(model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())


    # Plot the graph
    #make_dot(model(torch.randn(1, 1, 5, 5)), params=dict(model.named_parameters())).render("model_graph", format="pdf")

    # Get the format of the input data from dataset
    dummy_input = dataset.get_test_loader().dataset[0][0]
    dummy_input = torch.tensor(dummy_input).unsqueeze(0)    
    print(f"{dummy_input.shape=}")  
    
    #print(f"{acc=} {quant_acc=}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dummy_input = torch.randn(1, 16).to(device)  # Beispiel für ein Bild mit 3 Kanälen und 224x224 Pixeln

    # Pass the input through the model
    output = model(dummy_input)

    # Check if output is detached
    #print("Is output detached?", not output.requires_grad)

    # Visualize the model
    make_dot(output, params=dict(model.named_parameters())).render("model_visualization", format="png")
    print(f"{acc=} {quant_acc=}")
    
    #print(model)
    '''
    #model_quant.load_state_dict(torch.load(best_weight_path,  weights_only=True), strict=False)
    #acc_quant, _ = validator.validate(model_quant, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

    #print(f"Results {acc=} {acc_quant=}")
    '''
    lr = 0.000233
    
    try:
        trainer.train(model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr), lr,epochs)
    except Exception as e:
        traceback.print_exc()
        print(e)

    #print(f"Results {acc=} {acc_quant=}")