import enum
import habana_frameworks.torch.hpu as hpu
from habana_frameworks.torch.utils._experimental_C import synDeviceType
from habana_frameworks.torch.utils import _experimental_C

def _data_ptr(t) -> int:
    if hpu.is_available():
        hpu.init()
        return _experimental_C.data_ptr(t)
    else:
        return 0

def _get_device_type() -> int:
    if hpu.is_available():
        hpu.init()
        return _experimental_C.get_device_type()
    else:
        return -1

def _compute_stream() -> int:
    if hpu.is_available():
        hpu.init()
        return _experimental_C.compute_stream()
    else:
        return 0
