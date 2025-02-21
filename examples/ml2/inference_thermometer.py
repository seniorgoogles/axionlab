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

def evaluate(model, x_test, y_test, device):
    model.eval()
    with torch.no_grad():
        outputs = model(x_test.to(device))
        y_test = y_test.to(device)
        # For one-hot targets, get predicted indices.
        preds = outputs.argmax(dim=1)
        acc = (preds == y_test.argmax(dim=1)).sum().item() / y_test.shape[0]
    
    return acc
if __name__ == "__main__":
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
    thermometer = DistributiveThermometer(200).fit(x_train)
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
    
    config = "/home/fry/Documents/repositories/synapselab/examples/ml2/thermometer_float_model/config.yaml"
    save_weights_path = "/home/fry/Documents/repositories/synapselab/examples/ml2/thermometer_float_model/best_weights.pth"
    
    model = ModelBuilder().build(ModelTypes.JSC, config=config, weights_path=save_weights_path)
    model = model.to(device)

    print(evaluate(model, x_test, y_test, device))