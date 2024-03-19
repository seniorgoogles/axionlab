import torch
from colorama import Fore
from src.utils.device_selector import DeviceSelector
class Trainer(object):

    def __init__(self):
        self.lr = 0.0
        self.epochs = 0
        self.optimizer = None
        self.criterion = None

    def train(self, model, config, dataset, criterion, optimizer, lr, epochs, update_step_count=200):
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
        self.lr = lr
        self.epochs = epochs
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

            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                if index % int(update_step_count/100) == 0:
                    print(f"[{Fore.BLUE}{epoch + 1}/{self.epochs}{Fore.RESET} (Train)]\t{Fore.GREEN}loss:{Fore.RESET} "
                          f"{val_loss / total:.2f} {Fore.GREEN}accuracy:{Fore.RESET} {100.0 * correct / total:.2f}")
                elif index % update_step_count == 0:
                    self.__eval__(model, test_loader, criterion, device, epoch, self.epochs)
                index += 1

            # Do validation
        #self.__eval__(model, test_loader, criterion, device, None, None)
        #model.train()

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

    def train_teacher_student(self, teacher, student, config, dataset_loader, criterion, optimizer, epochs, T, teacherIsPreTrained=False):
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

        if(teacherIsPreTrained == False):
            print(f"{Fore.GREEN}")
            print("------------------------------------")
            print("\> Training Teacher")
            print("------------------------------------")
            print(f"{Fore.RESET}")
            self.train(teacher, config, dataset_loader, criterion, optimizer, 0.001, epochs)
        else: 
            print(f"{Fore.GREEN}")
            print("------------------------------------")
            print("\> Teacher Already Trained")
            print("------------------------------------")
            print(f"{Fore.RESET}")

        print(f"{Fore.GREEN}")
        print("------------------------------------")
        print("\> Training Student")
        print("------------------------------------")
        print(f"{Fore.RESET}")
        
        device = DeviceSelector.get_device()
        student.to(device)

        teacher.eval()
        student.train()
        index = 1

        train_loader = dataset_loader.get_train_loader()
        test_loader = dataset_loader.get_test_loader()
        soft_target_loss_weight = 0.25
        ce_loss_weight = 0.75
        update_step_count=200

        for epoch in range(epochs):
            total = 0
            correct = 0
            val_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                with torch.no_grad():
                    teacher_logits = teacher(inputs)

                outputs = student(inputs)

                #Soften student logits
                soft_targets = torch.nn.functional.softmax(teacher_logits / T, dim=-1)
                soft_prob = torch.nn.functional.log_softmax(outputs / T, dim=-1)

                soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T**2)

                label_loss = criterion(outputs, targets)
                loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                if index % int(update_step_count/100) == 0:
                    print(f"[{Fore.BLUE}{epoch + 1}/{self.epochs}{Fore.RESET} (Train)]\t{Fore.GREEN}loss:{Fore.RESET} "
                          f"{val_loss / total:.2f} {Fore.GREEN}accuracy:{Fore.RESET} {100.0 * correct / total:.2f}")
                elif index % update_step_count == 0:
                    self.__eval__(student, test_loader, criterion, device, epoch, self.epochs)
                index += 1

    def train_dcq(self, teacher, student, config, dataset_loader, criterion, optimizer, epochs, teacherIsPreTrained=False):

        if(teacherIsPreTrained == False):
            print(f"{Fore.GREEN}")
            print("------------------------------------")
            print("\> Training Teacher")
            print("------------------------------------")
            print(f"{Fore.RESET}")
            self.train(teacher, config, dataset_loader, criterion, optimizer, 0.001, epochs)
        else: 
            print(f"{Fore.GREEN}")
            print("------------------------------------")
            print("\> Teacher Already Trained")
            print("------------------------------------")
            print(f"{Fore.RESET}")

        # freeze_layer_index = 3

        # for index, layer in enumerate(student.layer):
        #     if index == freeze_layer_index:
        #         continue
        #     for param in layer.parameters():
        #         param.requires_grad = False

        # optimizer_frozen = torch.optim.Adam(filter(lambda p: not p.requires_grad, student.parameters()), lr=0.01)
        # optimizer_trainable = torch.optim.Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=0.001)
        
        device = DeviceSelector.get_device()

        student.to(device)

        train_loader = dataset_loader.get_train_loader()
        test_loader = dataset_loader.get_test_loader()
        index = 1
        update_step_count = 200

        all_layers = list(student.children())
        for layer in all_layers:
            if isinstance(layer, torch.nn.Sequential):
                for sub_layer in layer:
                    #print(sub_layer)
                    sub_layer.requires_grad = False
            else:
                layer.requires_grad = False
                #print(layer)
        
        optimizer_trainable = torch.optim.Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=0.001)
        if not optimizer_trainable.param_groups:
            print("optimizer is empty")
            return
        
        optimizer = optimizer_trainable
        
        
        for epoch in range(epochs):

                total = 0
                correct = 0
                val_loss = 0

                for inputs, targets in train_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = student(inputs)
                    loss = criterion(outputs, targets)
                
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                    if index % int(update_step_count/100) == 0:
                        print(f"[{Fore.BLUE}{epoch + 1}/{self.epochs}{Fore.RESET} (Train)]\t{Fore.GREEN}loss:{Fore.RESET} "
                            f"{val_loss / total:.2f} {Fore.GREEN}accuracy:{Fore.RESET} {100.0 * correct / total:.2f}")
                    elif index % update_step_count == 0:
                        self.__eval__(student, test_loader, criterion, device, epoch, self.epochs)
                    index += 1

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