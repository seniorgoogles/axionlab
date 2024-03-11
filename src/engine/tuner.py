import torch 
import copy
from torch import optim
import random
from tqdm import tqdm
import numpy as np
from itertools import islice


class Tuner(object):
    
    @staticmethod
    def tune(model, optimizer_cls, criterion, dataset, epochs, num_batches, iterations):
        
        history = []
    
        # Backup the model
        model_backup = copy.deepcopy(model)
        
        # Number of iterations to determine the best hyperparameters
        for i in range(iterations):
            train_dataset = dataset.get_train_loader()
            
            # If number of batches is > 0, then we will only use that many batches for training
            num_batches = num_batches if num_batches > 0 else len(train_dataset)
            train_dataset = islice(train_dataset, num_batches) if num_batches > 0 else train_dataset

            init_optim_vals = Tuner.__generate_random_optimizer_values(optimizer_cls)
            print(init_optim_vals)
            optimizer = optimizer_cls(model.parameters(), **init_optim_vals)

            correct = 0
            total = 0
            epoch_loss = 0
            
            for epoch in range(epochs):
                progress = tqdm(enumerate(train_dataset), total=num_batches, desc=f"Epoch {epoch+1}/{epochs}")

                for i, data in progress:
                    inputs, labels = data
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                    progress.set_postfix({'loss': epoch_loss/(i+1), 'accuracy': 100. * correct / total})

                accuracy = (100.0 * correct) / total
                print(f'Accuracy of the network on the train images: {accuracy} %')

            history.append({
                'accuracy': accuracy,
                'loss': epoch_loss / total,
                'params': init_optim_vals,
            })
                    
            # Reset the model to the original state after each iteration
            model = copy.deepcopy(model_backup)
        print(history)
     
    @staticmethod
    def __validate(model, criterion, dataset):
        
        test_loader = dataset.get_test_loader()
        
        model.eval()
        
        correct = 0
        total = 0
        val_loss = 0
        
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = model(images)
                
                va_loss += criterion(outputs, labels)


                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        val_loss = val_loss / total 
                   
        model.train()
        return accuracy, val_loss
            
        
    @staticmethod
    def __generate_random_optimizer_values(optimizer_cls):
        
        ret_val = None
        
        if issubclass(optimizer_cls, optim.Adam):
            
            lr_range = [0.0001,  0.01]
            betas_range = [(0.7, 0.9), (0.777, 0.999)]
            eps_range = [1e-08, 1e-06]
            weight_decay_range = [0, 0.0001]
            amsgrad_range = [False, True]
            
            # Generate random values within the specified ranges
            lr = random.uniform(*lr_range)
            betas = (random.uniform(*betas_range), random.uniform(*betas_range))
            eps = random.uniform(*eps_range)
            weight_decay = random.uniform(*weight_decay_range)
            amsgrad = random.choice(amsgrad_range)

            ret_val = {
                'lr': lr,
                'betas': betas,
                'eps': eps,
                'weight_decay': weight_decay,
                'amsgrad': amsgrad
            }
            
        elif issubclass(optimizer_cls, optim.SGD):
            
            lr_range = [0.0001,  0.01]
            momentum_range = [0.7, 0.9]
            dampening_range = [0, 0.1]
            weight_decay_range = [0, 0.0001]
            nesterov_range = [False, True]
            
            lr = random.uniform(*lr_range)
            nesterov = random.choice(nesterov_range)
            momentum = random.uniform(*momentum_range)
            dampening = random.uniform(*dampening_range) if not nesterov else 0.0
            weight_decay = random.uniform(*weight_decay_range)
            
            ret_val = {
                'lr': lr,
                'momentum': momentum,
                'dampening': dampening,
                'weight_decay': weight_decay,
                'nesterov': nesterov
            }
            
        return ret_val
