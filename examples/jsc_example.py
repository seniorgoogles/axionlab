from copy import deepcopy

from src.datasets.dataset_builder import DatasetBuilder
from src.models.model_builder import ModelBuilder
from src.core.inject.enum import ModelTypes, DatasetTypes
from src.engine.validator import Validator
from src.engine.trainer import Trainer
from src.engine.tuner import Tuner
from src.quantizer.learned_bitwidth_quantizer import LearnedBitWidthQuantizer

import os
import torch
from pathlib import Path
import yaml

def sparse_weights(model, test_loader, eps_steps=0.01, criterion=torch.nn.CrossEntropyLoss(), device=torch.device("cuda"), allowed_acc_drop=10.0):
    # Get all layers
    layers = model.named_children()

    for l in layers:
        layer_name = l[0]
        layer = l[1]

        if "dense" in layer_name:
            print("Yo")


if __name__ == "__main__":
    # Set the project root directory
    os.environ['PROJECT_ROOT'] = str(Path.cwd().parent)

    modelbuilder = ModelBuilder()
    validator = Validator()
    trainer = Trainer()
    lr = 0.0005332
    epochs = 200

    dataset = DatasetBuilder.build(DatasetTypes.JSC, "../configs/jsc/jsc_xl.yaml")

    configs = [
        #"../configs/jsc/jsc_2l.yaml",
        #"../configs/jsc/jsc_5l.yaml",
        #"../configs/jsc/jsc_lite.yaml",
       #"../configs/jsc/jsc_m_lite_floating_point.yaml",
        "../configs/jsc/jsc_xl.yaml",
       #"../configs/jsc/jsc_xl_floating_point.yaml"
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
    '''

    # Sparse weights per layer
    model = modelbuilder.build(ModelTypes.JSC, "../configs/jsc/jsc_xl.yaml", preload_weights=True)

    # Load checkpoint .pth file
    model.load_state_dict(torch.load("train/jsc_xl/run_13/best_weights.pth"))

    sparse_weights(model, dataset.get_test_loader())
    '''
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
    '''