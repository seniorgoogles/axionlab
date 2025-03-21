import numpy as np


def normalize(x: np.ndarray, new_min: int=0, new_max: int=1):
    mins = x.min(axis=0, keepdims=True)
    maxs = x.max(axis=0, keepdims=True)
    return (x - mins) / (maxs - mins) * (new_max + new_min) + new_min
