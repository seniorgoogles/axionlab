import torch
from colorama import Fore
from src.utils.device_selector import DeviceSelector
class Trainer(object):

    def __init__(self):
        self.lr = 0.0
        self.epochs = 0
        self.optimizer = None
        self.criterion = None

    def train(self, model, config, dataset, criterion, optimizer, update_step_count=200):
        """
        Basic training function

        :param model:
        :param config:
        :param dataset:
        :param criterion:
        :param optimizer:
        :param update_step_count:
        :return:
        """
        self.lr = 0.01
        self.epochs = 10
        self.optimizer = optimizer
        self.criterion = criterion

        device = DeviceSelector.get_device()

        model.to(device)

        train_loader = dataset.get_train_loader()
        test_loader = dataset.get_test_loader()

        model.train()
        index = 1

        print(f"{Fore.GREEN}")
        print("------------------------------------")
        print("\> Training starts")
        print("------------------------------------")
        print(f"{Fore.RESET}")

        for epoch in range(self.epochs):

            total = 0
            correct = 0
            val_loss = 0

            for inputs, targets, in train_loader:

                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                if index % update_step_count == 0:
                    print(f"[{Fore.BLUE}{epoch + 1}/{self.epochs}{Fore.RESET} (Train)]\t{Fore.GREEN}loss:{Fore.RESET} "
                          f"{val_loss / total:.2f} {Fore.GREEN}accuracy:{Fore.RESET} {100.0 * correct / total:.2f}")

                    self.__eval__(model, test_loader, criterion, device, epoch, self.epochs)
                index += 1

            # Do validation
            self.__eval__(model, test_loader, criterion, device, None, None)
            model.train()

    def train_by_strategy(self, model, config, dataset, criterion, optimizer, strategy):
        """
        Train the model by using the strategy, e.g. conquer and divide, by params, etc.
        :param model:
        :param config:
        :param dataset:
        :param criterion:
        :param optimizer:
        :param strategy:
        :return:
        """
        pass


    def train_teacher_student(self, teacher, student, config, dataset_loader, criterion, optimizer):
        """
        Training model by using teacher-student learning strategy
        :param teacher:
        :param student:
        :param config:
        :param dataset_loader:
        :param criterion:
        :param optimizer:
        :return:
        """
        pass

    def __eval__(self, model, test_loader, criterion, device, epoch, epochs):
        """
        Evaluate the model
        :param model:
        :param test_loader:
        :param criterion:
        :param device:
        :param epoch:
        :param epochs:
        :return:
        """
        val_loss = 0
        correct = 0
        total = 0

        model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                if epoch is not None:
                    print(f"[{Fore.BLUE}{epoch + 1}/{epochs}{Fore.RESET} (Valid)]\t{Fore.GREEN}loss:{Fore.RESET} "
                          f"{val_loss / total:.2f} {Fore.GREEN}accuracy:{Fore.RESET} {100.0 * correct / total:.2f}")
                else:
                    print(f"[{Fore.BLUE}Final{Fore.RESET} (Valid)]\t{Fore.GREEN}loss:{Fore.RESET} "
                          f"{val_loss / total:.2f} {Fore.GREEN}accuracy:{Fore.RESET} {100.0 * correct / total:.2f}")