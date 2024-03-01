import torch

from src.utils.device_selector import DeviceSelector
from src.utils.timer import timer
from tqdm import tqdm
class Validator(object):

    def __init__(self):
        self.lr = 0.001
        self.optimizer = None

        self.device = DeviceSelector.get_device()

    @timer
    def validate(self, model, config, dataset_loader, criterion):
        #self.lr = config["lr"]
        #self.optimizer = config["optimizer"]
        #self.criterion = config["criterion"]
        self.criterion = criterion

        val_loss = 0.0
        correct = 0
        total = 0
        index = 0


        model.eval()  # Set the model to evaluation mode
        model.to(self.device)

        num_batches = len(dataset_loader)

        with torch.no_grad():  # Disable gradient calculation during validation
            with tqdm(total=num_batches, desc="Progress", unit="iteration") as pbar:
                for inputs, targets in dataset_loader:
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

            avg_val_loss = val_loss / total
            accuracy = 100.0 * correct / total

            print(f'Validation Loss: {avg_val_loss:.4f} {accuracy:.4f}')
    @timer
    def compare_models(self, model_a, model_b, config, dataset):
        pass
