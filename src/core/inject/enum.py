from enum import auto


class ModelTypes:
    """Enum for model types."""
    LENET = auto()
    VGG = auto()
    RESNET = auto()
    JSC = auto()
    HDR = auto()
    CONTINUOUSFLOW_MNIST = auto()

class DatasetTypes:
    MNIST = auto()
    FASHION_MNIST = auto()
    CIFAR10 = auto()
    IMAGENET = auto()
    COCO = auto()
    JSC = auto()