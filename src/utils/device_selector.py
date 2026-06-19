import torch


class DeviceSelector(object):
    """Select the compute device: NVIDIA CUDA, AMD ROCm, Apple MPS, or CPU.

    Note: PyTorch's ROCm build reports through the CUDA API (HIP masquerades as
    cuda), so `torch.cuda.is_available()` is True on ROCm too -- both share the
    "cuda" device path.
    """

    @staticmethod
    def get_device(prefer: str = None, enable_mps: bool = True) -> torch.device:
        """Pick a device.

        Args:
            prefer: force a device string ("cuda", "mps", "cpu") if given.
            enable_mps: allow Apple MPS when no CUDA/ROCm GPU is present.
        """
        if prefer:
            return torch.device(prefer)

        if torch.cuda.is_available():
            backend = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
            print(f"Using GPU ({backend}): {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")

        mps = getattr(torch.backends, "mps", None)
        if enable_mps and mps is not None and mps.is_available():
            print("Using Apple MPS.")
            return torch.device("mps")

        print("Using CPU.")
        return torch.device("cpu")
