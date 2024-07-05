import os
import torch
from colorama import Fore
from src.utils.device_selector import DeviceSelector


class Trainer:
    def __init__(self, lr=0.0, epochs=0):
        self.lr = lr
        self.epochs = epochs
        self.optimizer = None
        self.criterion = None
        self.save_dir = None

    def train(self, model, config, dataset, criterion, optimizer, lr, epochs, update_step_count=200):
        self.lr = lr
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion

        device = DeviceSelector.get_device()
        model.to(device)

        self._setup_save_directory(model)

        train_loader = dataset.get_train_loader()
        test_loader = dataset.get_test_loader()

        model.train()
        print(f"{Fore.GREEN}------------------------------------")
        print("\> Training starts")
        print(f"------------------------------------{Fore.RESET}")

        best_accuracy = 0.0

        for epoch in range(self.epochs):
            self._train_one_epoch(model, train_loader, epoch, device, update_step_count)
            accuracy,_ = self.eval(model, test_loader, device, epoch, self.epochs)

            # Save the last weights
            self._save_model_weights(model, 'last_weights.pth')

            # Save the best weights
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self._save_model_weights(model, 'best_weights.pth')

    def train_by_strategy(self, model, dataset, criterion, optimizer, strategy):
        pass

    def train_teacher_student(self, teacher, student, dataset_loader, criterion, optimizer, epochs, T,
                              teacher_is_pretrained=False):
        if not teacher_is_pretrained:
            print(f"{Fore.GREEN}------------------------------------")
            print("\> Training Teacher")
            print(f"------------------------------------{Fore.RESET}")
            self.train(teacher, dataset_loader, criterion, optimizer, self.lr, epochs)
        else:
            print(f"{Fore.GREEN}------------------------------------")
            print("\> Teacher Already Trained")
            print(f"------------------------------------{Fore.RESET}")

        print(f"{Fore.GREEN}------------------------------------")
        print("\> Training Student")
        print(f"------------------------------------{Fore.RESET}")

        device = DeviceSelector.get_device()
        student.to(device)

        teacher.eval()
        student.train()

        train_loader = dataset_loader.get_train_loader()
        test_loader = dataset_loader.get_test_loader()

        for epoch in range(epochs):
            self._train_student_one_epoch(teacher, student, train_loader, criterion, optimizer, T, epoch, device)
            accuracy,_ = self.eval(student, test_loader, device, epoch, epochs)

            # Save the last weights
            self._save_model_weights(student, 'last_weights.pth')

            # Save the best weights
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self._save_model_weights(student, 'best_weights.pth')

    def train_dcq(self, teacher, student, dataset_loader, criterion, optimizer, epochs, sections,
                  teacher_is_pretrained=False):
        device = DeviceSelector.get_device()
        student.to(device)

        layer_count = sum(1 for _ in student.children())
        sections = min(sections, layer_count)
        layers_per_section = layer_count // sections

        teacher.eval()
        student.train()

        for section in range(sections):
            self._freeze_layers(student, section, layers_per_section)
            optimizer = self._create_optimizer_for_unfrozen_layers(student, self.lr)
            if optimizer is None:
                print("Optimizer is empty. Nothing to do.")
                return

            self.train_teacher_student(teacher, student, dataset_loader, criterion, optimizer, epochs, T=2,
                                       teacher_is_pretrained=teacher_is_pretrained)

    def _setup_save_directory(self, model):
        if not os.path.exists('train'):
            os.makedirs('train')

        model_name = model.name if hasattr(model, 'name') else 'default_model'
        print("Model name: ", model_name)
        model_dir = os.path.join('train', model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        run_index = 1
        while os.path.exists(os.path.join(model_dir, f'run_{run_index}')):
            run_index += 1

        self.save_dir = os.path.join(model_dir, f'run_{run_index}')
        os.makedirs(self.save_dir)

    def _train_one_epoch(self, model, train_loader, epoch, device, update_step_count):
        total = 0
        correct = 0
        val_loss = 0

        for index, (inputs, targets) in enumerate(train_loader, start=1):

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            if index % int(update_step_count / 100) == 0 or index % update_step_count == 0:
                print(f"[{Fore.BLUE}{epoch + 1}/{self.epochs}{Fore.RESET} (Train)]\t"
                      f"{Fore.GREEN}loss:{Fore.RESET} {val_loss / total:.2f} "
                      f"{Fore.GREEN}accuracy:{Fore.RESET} {100.0 * correct / total:.2f}")

    def _train_student_one_epoch(self, teacher, student, train_loader, criterion, optimizer, T, epoch, device, update_step_count=200):
        total = 0
        correct = 0
        val_loss = 0

        for index, (inputs, targets) in enumerate(train_loader, start=1):

            inputs = inputs.to(torch.float32)
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.no_grad():
                teacher_logits = teacher(inputs)

            outputs = student(inputs)
            soft_targets = torch.nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = torch.nn.functional.log_softmax(outputs / T, dim=-1)
            soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T ** 2)
            label_loss = criterion(outputs, targets)
            loss = 0.25 * soft_targets_loss + 0.75 * label_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            if index % int(update_step_count / 100) == 0 or index % update_step_count == 0:
                print(f"[{Fore.BLUE}{epoch + 1}/{self.epochs}{Fore.RESET} (Train)]\t"
                      f"{Fore.GREEN}loss:{Fore.RESET} {val_loss / total:.2f} "
                      f"{Fore.GREEN}accuracy:{Fore.RESET} {100.0 * correct / total:.2f}")

    def _freeze_layers(self, model, section, layers_per_section):
        start_layer = section * layers_per_section
        end_layer = start_layer + layers_per_section

        for i, layer in enumerate(model.children()):
            requires_grad = start_layer <= i < end_layer
            for param in layer.parameters():
                param.requires_grad = requires_grad

    def _create_optimizer_for_unfrozen_layers(self, model, lr):
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        return optimizer if optimizer.param_groups else None

    def eval(self, model, test_loader, device, epoch=None, epochs=None):

        model = model.to(device)
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = 100.0 * correct / total

        if epoch is not None and epochs is not None:
            print(f"[{Fore.BLUE}{epoch + 1}/{epochs}{Fore.RESET} (Test)]\t", end='')

        print(
            f"{Fore.GREEN}loss:{Fore.RESET} {val_loss / total:.2f} "
            f"{Fore.GREEN}accuracy:{Fore.RESET} {accuracy:.2f}")

        loss = (val_loss/total)

        return accuracy, loss

    def _save_model_weights(self, model, filename):
        if self.save_dir is not None:
            save_path = os.path.join(self.save_dir, filename)
            torch.save(model.state_dict(), save_path)
