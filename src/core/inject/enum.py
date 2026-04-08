from enum import Enum, auto


class ModelTypes(Enum):
    """Enum for model types."""
    LENET = auto()
    VGG = auto()
    RESNET = auto()
    JSC = auto()
    HDR = auto()
    CONTINUOUSFLOW_MNIST = auto()


class DatasetTypes(Enum):
    """Enum for dataset types."""
    MNIST = auto()
    FASHION_MNIST = auto()
    CIFAR10 = auto()
    IMAGENET = auto()
    COCO = auto()
    JSC = auto()