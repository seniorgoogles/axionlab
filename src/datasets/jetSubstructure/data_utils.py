import math

import numpy as np
from torch.utils.data import DataLoader

from scr.jetSubstructure.dataloader import CustomDataset

def calculate_data_bitwidth(dataset: CustomDataset):
    features = dataset.features

    msb_in = calculate_data_msb(features)
    lsb_in = calculate_data_lsb(features)

    return msb_in, lsb_in


def calculate_data_msb(features: np.ndarray):
    max_feature = features.max()
    min_feature = features.max()

    if abs(min_feature) > max_feature:
        max_feature = abs(min_feature)

    msb = math.log2(max_feature)
    msb_in = int(msb)
    if msb > msb_in:
        msb_in += 2
    else:
        msb_in += 1

    return msb_in


def calculate_data_lsb(features: np.ndarray):
    features = np.abs(features)
    frac, _ = np.modf(features)

    lsb = 0
    while len(frac[frac > 0]) > 0:
        lsb -= 1
        features *= 2
        frac, _ = np.modf(features)

    return lsb


def get_dataloader(dataset: CustomDataset):
    train_set, test_set = dataset.split()

    training_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)

    return training_loader, test_loader
