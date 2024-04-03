import torch


class DeviceSelector(object):
    """
    This class is responsible for selecting the device to be used for training or inference.
    """
    @staticmethod
    def get_device(enable_mps=False):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA.")
        elif torch.backends.mps.is_available() and enable_mps:
            device = torch.device("mps")
            print("Using MPS.")
        else:
            device = torch.device("cpu")
            print("Using CPU.")
        return device
