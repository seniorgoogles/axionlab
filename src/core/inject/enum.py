from enum import auto


class ModelTypes:
    """Enum for model types."""
    VGG = auto()
    RESNET = auto()

class DatasetTypes:
    MNIST = auto()
    FASHION_MNIST = auto()
    CIFAR10 = auto()
    IMAGENET = auto()
    COCO = auto()