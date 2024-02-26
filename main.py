from src.models.model_builder import ModelBuilder
from src.datasets.dataset_builder import DatasetBuilder
from src.core.inject.enum import ModelTypes, DatasetTypes
import torch
import torchvision.models as models
import time
from tqdm import tqdm

def validate_model(model, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    device = torch.device("cpu")

    # Check that MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")

    else:
        device = torch.device("mps")

    num_batches = len(val_loader)
    index = 1

    start_time = time.time()  # Record the start time
    with torch.no_grad():  # Disable gradient calculation during validation
        with tqdm(total=num_batches, desc="Progress", unit="iteration") as pbar:
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                model = model.to(device)
                # Forward pass

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                accuracy = (100.0 * correct / total)
                val_loss = (val_loss / total)

                pbar.update(1)
                # Optionally, update the progress bar description with additional values
                pbar.set_description(
                    "Progress: {:d}/{:d}, accuracy: {:.4f}%, validation loss: {:.4f}".format(index, num_batches, accuracy, val_loss))

                index += 1
            avg_val_loss = val_loss / total
            accuracy = 100.0 * correct / total

            print(f'Validation Loss: {avg_val_loss:.4f} {accuracy:.4f}')
            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time  # Calculate the elapsed time

            print(f'Validation Time: {elapsed_time:.2f} seconds')

def compare_models(model1, model2, val_loader, criterion):
    model1.eval()  # Set the model to evaluation mode
    model2.eval()
    val_loss1 = 0.0
    val_loss2 = 0.0
    correct1 = 0
    correct2 = 0
    total = 0

    device = torch.device("cpu")

    # Check that MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")

    else:
        device = torch.device("mps")

    start_time = time.time()  # Record the start time
    with torch.no_grad():  # Disable gradient calculation during validation
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            model1 = model1.to(device)
            model2 = model2.to(device)
            # Forward pass
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)

            are_equal = torch.equal(outputs1, outputs2)

            print(are_equal)  # Outputs: True
            loss1 = criterion(outputs1, targets)
            loss2 = criterion(outputs2, targets)

            are_equal = torch.equal(loss1, loss2)
            print(f"{are_equal=} {loss1=} {loss2=}")

            # Compute validation loss
            val_loss1 += loss1.item() * inputs.size(0)
            val_loss2 += loss2.item() * inputs.size(0)
            # Compute accuracy
            _, predicted1 = torch.max(outputs1, 1)
            _, predicted2 = torch.max(outputs2, 1)

            total += targets.size(0)
            correct1 += (predicted1 == targets).sum().item()
            correct2 += (predicted2 == targets).sum().item()

            print(f"{correct1=} {correct2=}")

    # Calculate average loss and accuracy
    avg_val_loss1 = val_loss1 / total
    accuracy1 = 100.0 * correct1 / total
    avg_val_loss2 = val_loss2 / total
    accuracy2 = 100.0 * correct2 / total

    print(f'Validation Loss: {avg_val_loss1:.4f} {avg_val_loss2:.4f}, Accuracy: {accuracy1:.2f} {accuracy2:.2f}%')
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time

    print(f'Validation Time: {elapsed_time:.2f} seconds')

    #return avg_val_loss, accuracy

# Function to check if two models have the same weights
'''
def compare_models(model1, model2):
    index = 0
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        print()
        if not torch.equal(param1, param2):
            return False
    return True


def are_models_identical(model1, model2):
    # Get the state_dicts of both models
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    # Check if keys are the same
    if state_dict1.keys() != state_dict2.keys():
        return False

    # Check if values are the same for each key
    for key in state_dict1.keys():
        if not torch.all(torch.eq(state_dict1[key], state_dict2[key])):
            return False

    return True


'''

if __name__ == "__main__":
    modelbuilder = ModelBuilder()
    #model = modelbuilder.build(ModelTypes.RESNET, "quant_config.yaml", preload_weights=True)
    #resnet18 = models.resnet18(pretrained=True)
    #dataset = DatasetBuilder.build(DatasetTypes.IMAGENET, "quant_config.yaml")

    model = modelbuilder.build(ModelTypes.LENET, "configs/lenet5/config.yaml", preload_weights=True)
    print(model)

    inp = torch.rand(1, 1, 32, 32)
    model(inp)
    '''
    # Check if the models have the same weights
    if compare_models(model, resnet18):
        print("Models have the same weights.")
    else:
        print("Models have different weights.")

    if are_models_identical(model, resnet18):
        print("Models are identical.")
    else:
        print("Models are not identical.")

    inp = torch.rand(1, 3, 224, 224)

    # Trace the models
    trace1 = torch.jit.trace(resnet18, inp)
    trace2 = torch.jit.trace(model, inp)

    # Compare the traces
    # Compare the string representations of the traces
    # Get the computational graphs
    graph1 = trace1.graph
    graph2 = trace2.graph

    out1 = resnet18(inp)
    out2 = model(inp)


    print(out1)
    print(out2)

    # Compare the computational graphs
    if str(graph1) == str(graph2):
        print("The traces are the same.")
    else:
        print("The traces are different.")
        print("ORG.............")
        print(graph1)
        print("MINE.............")
        print(graph2)
        #raise Exception()

    #dummy_input = torch.randn(1, 3, 224, 224)

    #torch.onnx.export(model, dummy_input, "own_resnet.onnx", verbose=True)
    #torch.onnx.export(model, dummy_input, "resnet.onnx", verbose=True)

    #resnet18.load_state_dict(model.state_dict())

    #model.load_state_dict(resnet18.state_dict())
    '''
    #compare_models(model, resnet18, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())
    #validate_model(model, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

