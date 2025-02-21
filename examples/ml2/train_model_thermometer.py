from matplotlib import pyplot as plt
import datetime
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from thermometer import DistributiveThermometer
from includes_ml2 import *
import openml

# Import your custom modules as needed.
# from includes_ml2 import *
# from helpers import train_model, print_colored_window
# from thermometer import DistributiveThermometer
# from model_builder import ModelBuilder, ModelTypes  # Example, adjust as needed.



class ThermometerDataset:
    """Container for new dataset loaders."""
    def __init__(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

def convert_to_thermometer(old_dataset, bit_count):
    # Transformation: ensure tensor and flatten.
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x if isinstance(x, torch.Tensor) 
                            else transforms.functional.to_tensor(x)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    
    # Extract one batch from the original loaders.
    x_train, y_train = next(iter(old_dataset.get_train_loader()))
    x_test, y_test = next(iter(old_dataset.get_test_loader()))
    
    # Transform the input samples.
    x_train_transformed = torch.stack([transform(sample) for sample in x_train])
    x_test_transformed  = torch.stack([transform(sample) for sample in x_test])
    
    # Fit the thermometer on training data and binarize both sets.
    thermometer = DistributiveThermometer(bit_count).fit(x_train_transformed)
    x_train_therm = thermometer.binarize(x_train_transformed).flatten(start_dim=1)
    x_test_therm  = thermometer.binarize(x_test_transformed).flatten(start_dim=1)
    
    # Label conversion: ensure labels are Pandas Series.
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test)
    
    combined_labels = pd.concat([y_train, y_test])
    label_names = list(combined_labels.unique())
    
    y_train_indices = y_train.map(lambda x: label_names.index(x)).values
    y_test_indices  = y_test.map(lambda x: label_names.index(x)).values
    
    # Convert indices to tensors.
    y_train_indices = torch.tensor(y_train_indices, dtype=torch.int64)
    y_test_indices  = torch.tensor(y_test_indices, dtype=torch.int64)
    
    # One-hot encode the labels.
    num_classes = len(label_names)
    y_train_one_hot = F.one_hot(y_train_indices, num_classes=num_classes)
    y_test_one_hot  = F.one_hot(y_test_indices, num_classes=num_classes)
    
    # Convert one-hot encoded labels to float tensors.
    y_train_tensor = y_train_one_hot.float()
    y_test_tensor  = y_test_one_hot.float()
    
    batch_size_train = x_train_therm.shape[0]
    batch_size_test  = x_test_therm.shape[0]
    
    # Create new TensorDatasets and DataLoaders.
    train_dataset = TensorDataset(x_train_therm, y_train_tensor)
    test_dataset  = TensorDataset(x_test_therm, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    
    return ThermometerDataset(train_loader, test_loader)

def get_parameters(model):
    num_weights = 0
    num_biases = 0
    layer_weights = {}
    
    for name, param in model.named_parameters():
        # Get the number of params in the layer
        num_params = param.numel()

        if "weight" in name:
            num_weights += num_params
            layer_weights[name] = num_params
        elif "bias" in name:
            num_biases += num_params
        
        print(f"Layer: {name}, Weights: {num_params}")
    
    print(f"Total weights: {num_weights}")
    print(f"Total biases: {num_biases}")
    
    layer_weights["total_weights"] = num_weights
    layer_weights["total_biases"] = num_biases
    
    return layer_weights

def evaluate(model, x_test, y_test, device):
    model.eval()
    with torch.no_grad():
        outputs = model(x_test.to(device))
        # For one-hot targets, get predicted indices.
        preds = outputs.argmax(dim=1)
        acc = (preds.cpu() == y_test.argmax(dim=1)).sum().item() / y_test.shape[0]
    return acc

def train_and_evaluate(model, optimizer, scheduler, x_train, y_train, x_test, y_test, epochs, batch_size, device, save_weights_path):
    
    accuracy = 0.0
    best_accuracy = 0.0
    
    n_samples = x_train.shape[0]
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n_samples)
        correct_train = 0
        total_train = 0
        
        for i in range(0, n_samples, batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i+batch_size]
            batch_x = x_train[indices].to(device)
            batch_y = y_train[indices].to(device)
            
            outputs = model(batch_x)
            loss = F.cross_entropy(outputs, batch_y.argmax(dim=1))
            loss.backward()
            optimizer.step()
            
            preds = outputs.argmax(dim=1)
            correct_train += (preds == batch_y.argmax(dim=1)).sum().item()
            total_train += batch_y.size(0)
        
        train_acc = correct_train / total_train

        
        test_acc = evaluate(model, x_test, y_test, device)
        scheduler.step(test_acc)
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), save_weights_path)
            
            print("Model weights saved. Best accuracy: {:.4f}".format(best_accuracy))
            
        
        print(f"Epoch {epoch+1}/{epochs}: Loss={loss.item():.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

if __name__ == "__main__":
    THERMOMETER_STEPS = 30
    # Load dataset from OpenML.
    dataset = openml.datasets.get_dataset(42468)
    df_features, df_labels, _, _ = dataset.get_data(dataset_format='dataframe', 
                                                     target=dataset.default_target_attribute)
    features = df_features.values.astype(np.float32)
    label_names = list(df_labels.unique())
    labels = np.array(df_labels.map(lambda x: label_names.index(x)).values)
    
    # Split into training and testing sets.
    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.8, random_state=42)
    
    # Convert features using the thermometer.
    thermometer = DistributiveThermometer(THERMOMETER_STEPS).fit(x_train)
    x_train = thermometer.binarize(x_train).flatten(start_dim=1)
    x_test = thermometer.binarize(x_test).flatten(start_dim=1)
    
    # Convert labels to tensors.
    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test = torch.tensor(y_test, dtype=torch.int64)
    
    # One-hot encode the labels.
    num_classes = len(label_names)
    y_train = F.one_hot(y_train, num_classes=num_classes).float()
    y_test = F.one_hot(y_test, num_classes=num_classes).float()
    
    device = 'cuda:0'
    epochs = 70
    batch_size = 100
    
    config = "/home/fry/Documents/repositories/synapselab/examples/ml2/thermometer_float_model/config.yaml"
    save_weights_path = "/home/fry/Documents/repositories/synapselab/examples/ml2/thermometer_float_model/best_weights.pth"
    
    config_float = "/home/fry/Documents/repositories/synapselab/examples/ml2/float_model/config.yaml"
    save_weights_path_float = "/home/fry/Documents/repositories/synapselab/examples/ml2/float_model/best_weights.pth"
    
    current_date = datetime.datetime.now()
    # Define paths
    train_folder = f"{parent_directory}/examples/ml2/train_thermometer"
    model_config = f"{parent_directory}/examples/ml2/thermometer_float_model/config.yaml"
    model_output_folder = f"{train_folder}/jsc_xl"
    
    current_year = current_date.year
    current_month = f"0{current_date.month}" if current_date.month < 10 else f"{current_date.month}"
    current_day = f"0{current_date.day}" if current_date.day < 10 else f"{current_date.day}"
    current_hour = f"0{current_date.hour}" if current_date.hour < 10 else f"{current_date.hour}"
    current_minute = f"0{current_date.minute}" if current_date.minute < 10 else f"{current_date.minute}"
    current_second = f"0{current_date.second}" if current_date.second < 10 else f"{current_date.second}"
    
    model_output_folder_train = f"{model_output_folder}/{current_year}{current_month}{current_day}_{current_hour}{current_minute}{current_second}"
    save_weights_path = f"{model_output_folder_train}/best_weights.pth"
    
    # Build and move the model to the device.
    model = ModelBuilder().build(ModelTypes.JSC, config=config)
    model = model.to(device)
    
    lr = 0.00009238
    lr = 0.000999
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=14)
    
    os.makedirs(model_output_folder_train, exist_ok=True)

    
    train_and_evaluate(model, optimizer, scheduler, x_train, y_train, x_test, y_test, epochs, batch_size, device, save_weights_path)

    # Copy the config file to the output folder.
    os.makedirs(model_output_folder_train, exist_ok=True)
    os.system(f"cp {model_config} {model_output_folder_train}")
    
    model = ModelBuilder().build(ModelTypes.JSC, config=config, weights_path=save_weights_path)
    model = model.to(device)
    
    accuracy = evaluate(model, x_test, y_test, device)
    accuracy = accuracy * 100.0
    
    thermometer_params = get_parameters(model)
    
    float_model = ModelBuilder().build(ModelTypes.JSC, config=config_float, weights_path=save_weights_path_float)
    float_params = get_parameters(float_model)

    # Write model accuracy, the thermometer params and as reference the float model params to a file.
    with open(f"{model_output_folder_train}/results.txt", "w") as f:
        f.write(f"Model accuracy: {accuracy:.2f} | 75.33% \n")

        # Write thermometer and float model parameters next to each other
        for key in thermometer_params.keys():
            f.write(f"{key}: {thermometer_params[key]} | {float_params[key]}\n")
            
            if key == "dense5.weight":
                f.write("-------------------------\n")