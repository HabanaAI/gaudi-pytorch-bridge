import torch
from habana_frameworks.torch import _hpex_C


def random_seed(x: torch.tensor) -> torch.tensor:
    # Error checking
    dtype = x.dtype
    if dtype != torch.int32:
        raise TypeError(f"Only int32 seed is accepted, got: {dtype}")
    device = x.device

    if device == torch.device("cpu"):
        raise ValueError(f"HPU RandomSeed is only supported on hpu device")
    else:
        try:
            from habana_frameworks.torch import _hpex_C

            return _hpex_C.random_seed(x)
        except ImportError:
            raise ImportError("Please install habana_torch.")
