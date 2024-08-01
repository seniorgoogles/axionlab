import sys
from pathlib import Path

# Ermitteln des Pfades des aktuellen Skripts (jsc_example.py)
current_script_path = Path(__file__).parent

# Ermitteln des übergeordneten Verzeichnisses von `src`
parent_directory = current_script_path.parent

# Hinzufügen des übergeordneten Verzeichnisses zu sys.path
sys.path.append(str(parent_directory))

# Jetzt können Sie Ihre Importe ausführen
from src.datasets.dataset_builder import DatasetBuilder
from src.models.model_builder import ModelBuilder
from src.core.inject.enum import ModelTypes, DatasetTypes
from src.engine.validator import Validator
from src.engine.trainer import Trainer
from src.engine.tuner import Tuner
from src.quantizer.learned_bitwidth_quantizer import LearnedBitWidthQuantizer

import torch
import os

def prune_weights(weights, eps):
    weights[abs(weights) < eps] = 0
    return weights

def sparse_weights(model, validator, test_loader, eps_steps=0.01, criterion=torch.nn.CrossEntropyLoss(), device=torch.device("cuda"), target_acc=10.0, eps_step=0.000001):
    # Get all layers
    layers = model.named_children()
    eps = eps_step

    target_zero_percentage = 30.0
    current_zeros_percentage = 0.0

    for l in layers:
        # Get the layer name and the layer itself
        layer_name = l[0]
        layer = l[1]
        min_weight = 0.0
        max_weight = 0.0
        iteration = 0

        # Prune only dense layers
        if "dense" in layer_name:
            # Create a deep copy of the weights
            while current_zeros_percentage < target_zero_percentage:
                weights_bak = deepcopy(layer.weight.data.cpu().numpy())
                weights = deepcopy(weights_bak)

                # Get Min and Max values of the weights
                min_weight = weights_bak.min()
                max_weight = weights_bak.max()

                # Print the percentage of weight that are zero
                current_zeros_percentage = 100 * (weights_bak == 0).sum() / weights_bak.size
                #print(f"{layer_name} Zeros: {current_zeros_percentage}% {min_weight=} {max_weight=}")

                # Prune every weight below the epsilon
                weights[abs(weights) < eps] = 0

                # Write weights back to layer1
                layer.weight.data = torch.tensor(weights)
                eps += eps_steps
                iteration += 1

            print(f"{layer_name} {iteration=} Zeros: {current_zeros_percentage}% {min_weight=} {max_weight=}")
            eps = eps_step
            current_zeros_percentage = 0.0


if __name__ == "__main__":
    # Set the project root directory
    os.environ['PROJECT_ROOT'] = str(Path.cwd().parent)

    modelbuilder = ModelBuilder()
    validator = Validator()
    trainer = Trainer()
    lr = 0.0005332
    epochs = 3

    dataset = DatasetBuilder.build(DatasetTypes.JSC, "configs/jsc/jsc_xl.yaml")

    configs = [
        #"configs/jsc/jsc_2l.yaml",
        #"configs/jsc/jsc_5l.yaml",
        #"configs/jsc/jsc_lite.yaml",
        #"configs/jsc/jsc_m_lite_floating_point.yaml",
        "configs/jsc/jsc_xl.yaml",
       #"configs/jsc/jsc_xl_floating_point.yaml"
    ]

    accuracies = dict()
    losses = dict()

    '''
    for config in configs:
        try:
            model = modelbuilder.build(ModelTypes.JSC, config, preload_weights=True)
            trainer.train(model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr), lr,epochs)

            accuracy, loss = trainer.eval(model, dataset.get_test_loader(), torch.device("cuda"), 0, epochs)
            model_name = model.name if hasattr(model, 'name') else 'default_model'

            accuracies[model_name] = accuracy
            losses[model_name] = loss


        except Exception as e:
            print(f"Error: {e}")
            continue

    # Write accuracies as latex table
    # Escape underscores in model names, add label for the table at the end, center table
    with open("accuracies.tex", "w") as f:
        f.write("\\begin{table}[]\n")
        f.write("\\begin{tabular}{|c|c|}\n")
        f.write("\\hline\n")
        f.write("Model & Accuracy \\\\ \\hline\n")
        for model, accuracy in accuracies.items():
            f.write(f"{model.replace('_', '\\_')} & {accuracy:.2f} \\\\ \\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Accuracy of models on the JSC dataset }" + f"{lr=} " + f"{epochs=}" + "\n")
        f.write("\\centering\n")
        f.write("\\end{table}\n")


    # Sparse weights per layer
    model = modelbuilder.build(ModelTypes.JSC, "configs/jsc/jsc_xl.yaml", preload_weights=True)
    #trainer.train(model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr), lr, epochs)

    try:
        # Load checkpoint .pth file
        # Get the path to the train folder
        train_folder = os.path.join("train")
        print(f"{train_folder=}")

        # Check if the jsc_xl folder exists in the train folder
        if os.path.exists(os.path.join(train_folder, "jsc_xl")):
            # Get all subdirectories in the jsc_xl folder
            subdirectories = [d for d in os.listdir(os.path.join(train_folder, "jsc_xl")) if os.path.isdir(os.path.join(train_folder, "jsc_xl", d))]

            # Sort the subdirectories in descending order based on the integer value in the directory name
            sorted_subdirectories = sorted(subdirectories, key=lambda x: int(x.split("_")[-1]), reverse=True)

            # Get the path to the run with the highest integer
            highest_run_path = os.path.join(train_folder, "jsc_xl", sorted_subdirectories[0])

            # Get the path to the best_weights.pth file
            best_weights_path = os.path.join(highest_run_path, "best_weights.pth")

            # Load the state dict from the best_weights.pth file
            model.load_state_dict(torch.load(best_weights_path,  weights_only=True,), strict=False)

            # Validate the model
            accuracy, loss = validator.validate(model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

            quant_model = modelbuilder.build(ModelTypes.JSC, "configs/jsc/quant_jsc_xl.yaml", preload_weights=False)

            # Load state dict, if keys missing, load only weights
            quant_model.load_state_dict(torch.load(best_weights_path,  weights_only=False), strict=False)
            quant_accuracy, quant_loss = validator.validate(quant_model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())


            print(f"Accuracy Model: {accuracy} Accuracy Quant-Model: {quant_accuracy}")


            # Get all layers from quant_model
            print(model)
            def update_bitwidth(module, new_value):
                for name, submodule in module.named_modules():
                    if isinstance(submodule, BitWidthConst):
                        submodule.bit_width.value = new_value

            update_bitwidth(quant_model.dense1.weight_quant, torch.tensor(2.))
            # Retrain quantized model
            #trainer.train(quant_model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(quant_model.parameters(), lr), lr, epochs)

            #trainer.train(quant_model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(quant_model.parameters(), lr), lr, epochs)
            #quant_accuracy, quant_loss = validator.validate(quant_model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

            #print(f"Accuracy Model: {accuracy} Accuracy Quant-Model: {quant_accuracy}")

            # Get the best accuracy and loss from the best_weights.pth file
            #best_accuracy = torch.load(best_weights_path, weights_only=True)["best_accuracy"]
            #print(best_accuracy)

        else:
            print("The 'jsc_xl' folder does not exist in the 'train' folder.")
    except Exception as ex:
        print(ex)
    '''

    #sparse_weights(model, validator, dataset.get_test_loader())
    #trainer.eval(model, dataset.get_test_loader(), torch.device("cuda"), None, None)
    base_acc, _ = validator.validate(model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

    # Get weights from first layer of the model
    weights = model.dense1.weight.data.cpu().numpy()
    weights_bak = deepcopy(weights)

    # Get Min and Max values of the weights
    min_weight = weights.min()
    max_weight = weights.max()

    print(f"Min weight: {min_weight}")
    print(f"Max weight: {max_weight}")

    eps_step = 0.001
    eps = eps_step

    results = dict()

    while eps < 0.3:

        # Print the percentage of weight that are zero
        print(f"Percentage of weights that are zero: {100 * (weights == 0).sum() / weights.size}")

        # Set all weights to Zero if they are less than absolute value of 0.1
        weights[abs(weights) < eps] = 0
        percentage_zeros = 100 * (weights == 0).sum() / weights.size
        print(f"Percentage of weights that are zero: {percentage_zeros}")

        # Write weights back to layer1
        model.dense1.weight.data = torch.tensor(weights)

        # Validate
        acc, loss = validator.validate(model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

        weights = deepcopy(weights_bak)
        model.dense1.weight.data = torch.tensor(weights)
        #validator.validate(model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss(), debug=False)

        results[f"Eps_{eps}"] = (acc, loss, percentage_zeros)

        print(f"Accuracy: {acc} Loss: {loss} Eps: {eps}")
        #if acc < base_acc - 10.0:
        #    #print(f"Accuracy: {acc} Loss: {loss} Eps: {eps}")
        #    break

        if acc < base_acc - 20.0:
            break

        eps += eps_step


    for key, value in results.items():
        print(f"{key}: {value}")
