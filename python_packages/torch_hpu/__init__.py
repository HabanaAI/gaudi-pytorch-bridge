import warnings
from typing import Optional, Union

import torch

_device_t = Union[torch.device, str, int, None]


def is_available() -> bool:
    warnings.warn("torch_hpu.is_available is deprecated. " "Please use habana_frameworks.torch.hpu.is_available")
    import habana_frameworks.torch.hpu as hpu

    return hpu.is_available()


def device_count() -> int:
    warnings.warn("torch_hpu.device_count is deprecated. " "Please use habana_frameworks.torch.hpu.device_count")
    import habana_frameworks.torch.hpu as hpu

    return hpu.device_count()


def get_device_name(device: Optional[_device_t] = None) -> str:
    warnings.warn("torch_hpu.get_device_name is deprecated. " "Please use habana_frameworks.torch.hpu.get_device_name")
    import habana_frameworks.torch.hpu as hpu

    return hpu.get_device_name(device)
