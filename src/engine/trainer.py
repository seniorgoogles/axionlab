import os
import torch
from colorama import Fore
from tqdm import tqdm
from src.utils.device_selector import DeviceSelector


class Trainer:
    def __init__(self, lr=0.0, epochs=0):
        """
        Initialize the Trainer with default values for learning rate and epochs.
        """
        self.lr = lr                  # Learning rate for training.
        self.epochs = epochs          # Total number of training epochs.
        self.optimizer = None         # Placeholder for the optimizer.
        self.criterion = None         # Placeholder for the loss function.
        self.save_dir = None          # Directory to save model weights.

    def train(self, model, config, dataset, criterion, optimizer, lr, epochs, update_step_count=200, scheduler=None, save_dir=None, save_model=False):
        """
        Main training loop.

        Parameters:
            model: The neural network model to train.
            config: Additional configuration settings (not used in this snippet).
            dataset: An object that provides training and testing data loaders.
            criterion: The loss function.
            optimizer: The optimizer for model parameter updates.
            lr: The learning rate.
            epochs: The number of training epochs.
            update_step_count: Number of steps between certain updates (not used explicitly).
            scheduler: A learning rate scheduler (optional).
            save_dir: Directory to save model checkpoints (optional).
            save_model: Boolean flag to enable model checkpoint saving.

        The function performs training over a specified number of epochs, evaluates the model
        on the test set, and optionally saves both the latest and the best model weights.
        """
        # Update training parameters.
        self.lr = lr
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion

        # Get the device (CPU or GPU) to perform training.
        device = DeviceSelector.get_device()
        model.to(device)  # Move model to the selected device.

        # Retrieve the training and testing data loaders from the dataset.
        train_loader = dataset.get_train_loader()
        test_loader = dataset.get_test_loader()
        
        # If saving is enabled, set up a directory for saving model weights.
        if save_model and save_dir == None:
            self._setup_save_directory(model)

        # Set the model to training mode
        model.train()
        print(f"{Fore.GREEN}------------------------------------")
        print("> Training starts")
        print(f"------------------------------------{Fore.RESET}")

        best_accuracy = 0.0  # Initialize the best accuracy for checkpointing.

        # Loop over each epoch.
        for epoch in range(self.epochs):
            running_loss = 0.0

            # Train the model for one epoch.
            self._train_one_epoch(model, train_loader, epoch, device, update_step_count)
            # Evaluate the model on the test set.
            accuracy, val_loss = self.eval(model, test_loader, device, epoch, self.epochs, scheduler=scheduler)
            
            # Save the most recent model weights (if saving is enabled).
            if save_model:
                self._save_model_weights(model, 'last_weights.pth', save_dir)

            # If the current epoch's accuracy is the best so far, save the model weights.
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                if save_model:
                    self._save_model_weights(model, 'best_weights.pth')
            
            avg_val_loss = val_loss / len(test_loader)
            
            if scheduler is not None:
                print(f"Average validation loss: {accuracy}")
                scheduler.step(accuracy)
                
                print(f"{accuracy=} {scheduler.get_last_lr()=}")

    def _setup_save_directory(self, model):
        """
        Set up a directory structure to save model weights.

        The structure is:
            train/<model_name>/run_<n>/
        where <n> increments to avoid overwriting previous runs.
        """
        # Create the base directory 'train' if it doesn't exist.
        if not os.path.exists('train'):
            os.makedirs('train')

        # Determine the model's name; use 'default_model' if not specified.
        model_name = model.name if hasattr(model, 'name') else 'default_model'
        print("Model name: ", model_name)
        model_dir = os.path.join('train', model_name)
        # Create the model directory if it doesn't exist.
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Increment run index until a new run directory is found.
        run_index = 1
        while os.path.exists(os.path.join(model_dir, f'run_{run_index}')):
            run_index += 1

        # Set and create the save directory for this run.
        self.save_dir = os.path.join(model_dir, f'run_{run_index}')
        os.makedirs(self.save_dir)

    def _train_one_epoch(self, model, train_loader, epoch, device, update_step_count):
        """
        Train the model for one epoch.

        Parameters:
            model: The model to train.
            train_loader: DataLoader for the training dataset.
            epoch: Current epoch number.
            device: Device (CPU/GPU) to run training on.
            update_step_count: The number of steps between any potential updates (not used here).

        This function iterates over the training data, performs a forward pass,
        computes the loss, backpropagates, and updates the model parameters.
        It also displays a progress bar with current loss and accuracy.
        """
        total = 0         # Total number of samples processed.
        correct = 0       # Number of correct predictions.
        val_loss = 0      # Accumulated loss over the epoch.
        
        # Create a tqdm progress bar for visual feedback.
        progress_bar = tqdm(total=len(train_loader), bar_format="{l_bar}{bar}{r_bar}", dynamic_ncols=True)

        # Iterate through batches in the training data.
        for index, (inputs, targets) in enumerate(train_loader, start=1):
            # Move inputs and targets to the specified device.
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)  # Perform a forward pass.
            loss = self.criterion(outputs, targets)  # Compute the loss.
            
            # Zero out gradients from previous iterations.
            self.optimizer.zero_grad()
            loss.backward()  # Backpropagation.
            # Optionally, perform gradient clipping by uncommenting the following line:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()  # Update the model parameters.

            # Update cumulative loss and accuracy metrics.
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # Update the progress bar with the current loss and accuracy.
            progress_bar.set_description(f"[{Fore.BLUE}{epoch + 1}/{self.epochs}{Fore.RESET} (Train)]")
            progress_bar.set_postfix_str(f"{Fore.GREEN}loss:{Fore.RESET} {val_loss / total:.2f} "
                                         f"{Fore.GREEN}accuracy:{Fore.RESET} {100.0 * correct / total:.2f}")
            progress_bar.update(1)  # Move the progress bar forward.
               
        progress_bar.close()  # Close the progress bar at the end of the epoch.

    def eval(self, model, test_loader, device, epoch=None, epochs=None, scheduler=None, save_dir=None):
        """
        Evaluate the model on the test dataset.

        Parameters:
            model: The model to evaluate.
            test_loader: DataLoader for the test dataset.
            device: Device (CPU/GPU) to run evaluation on.
            epoch: Current epoch number (optional, for logging).
            epochs: Total number of epochs (optional, for logging).
            scheduler: Learning rate scheduler (if provided, updated based on accuracy).
            save_dir: Directory for saving models (not used in this function).

        Returns:
            accuracy: The overall accuracy on the test dataset.
            loss: The average loss on the test dataset.
        """
        model = model.to(device)
        model.eval()  # Set the model to evaluation mode.
        val_loss = 0    # Initialize cumulative loss.
        correct = 0     # Initialize correct prediction count.
        total = 0       # Total number of samples.

        # Disable gradient computation for evaluation.
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)  # Forward pass.
                loss = self.criterion(outputs, targets)  # Compute loss.

                # Update cumulative loss and accuracy metrics.
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        # Compute overall accuracy.
        accuracy = 100.0 * correct / total

        # Optionally print epoch information if provided.
        if epoch is not None and epochs is not None:
            print(f"[{Fore.BLUE}{epoch + 1}/{epochs}{Fore.RESET} (Test)]\t", end='')

        # Print the loss and accuracy for the test set.
        print(
            f"{Fore.GREEN}loss:{Fore.RESET} {val_loss / total:.2f} "
            f"{Fore.GREEN}accuracy:{Fore.RESET} {accuracy:.2f}")

        loss = (val_loss / total)
        
        # If a scheduler is provided, step it using the current accuracy.
        if scheduler is not None:
            scheduler.step(accuracy)

        return accuracy, loss

    def _save_model_weights(self, model, filename, save_dir=None):
        """
        Save the model's state dictionary (weights) to a file.

        Parameters:
            model: The model whose weights will be saved.
            filename: The name of the file in which to store the weights.
            save_dir: Optionally update the save directory.
        """
        # Update the save directory if a new one is provided.
        if save_dir is not None:
            self.save_dir = save_dir
            
        # Save the model's weights if the save directory is defined.
        if self.save_dir is not None:
            save_path = os.path.join(self.save_dir, filename)
            torch.save(model.state_dict(), save_path)
