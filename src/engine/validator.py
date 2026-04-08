import torch

from src.utils.device_selector import DeviceSelector
from src.utils.timer import timer
from tqdm import tqdm

class Validator(object):

    def __init__(self, criterion=None, dataset_loader=None, device=None):
        self.criterion = criterion
        self.dataset_loader = dataset_loader

        if device is None:
            self.device = DeviceSelector.get_device()
        else:
            self.device = torch.device(device)

    def validate(self, model, num_batches=-1, debug=False):

        val_loss = 0.0
        correct = 0
        total = 0
        index = 0

        model.eval()  # Set the model to evaluation mode
        model.to(self.device)

        num_batches = len(self.dataset_loader)

        with torch.no_grad():  # Disable gradient calculation during validation
            with tqdm(total=num_batches, desc="Progress", unit="iteration") as pbar:
                for inputs, targets in self.dataset_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    model = model.to(self.device)
                    # Forward pass

                    outputs = model(inputs)
                    loss = self.criterion(outputs, targets)

                    val_loss += loss.item() * inputs.size(0)

                    _, predicted = torch.max(outputs, 1)
                    total += targets.size(0)

                    correct += (predicted == targets).sum().item()

                    accuracy = (100.0 * correct / total)
                    val_loss = (val_loss / total)

                    pbar.update(1)
                    # Optionally, update the progress bar description with additional values
                    pbar.set_description(
                        "Progress: {:d}/{:d}, accuracy: {:.4f}%, validation loss: {:.4f}".format(index, num_batches,
                                                                                                 accuracy, val_loss))
                    index += 1
            accuracy = 100.0 * correct / total
            if debug:
                print(f"Accuracy: {accuracy:.2f}% Validation Loss: {val_loss:.4f}")
            return accuracy, val_loss
