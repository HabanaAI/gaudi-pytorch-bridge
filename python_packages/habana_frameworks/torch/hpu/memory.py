import collections
import torch
import warnings
from typing import Any, Dict, Union, Optional
import habana_frameworks.torch.hpu as hpu
from habana_frameworks.torch import _hpu_C
from ._utils import _get_device_index
_device_t = Union[torch.device, str, int, None]

def max_memory_allocated(device: Optional[_device_t] = None) -> int:
    r"""This API (TORCH.HPU.MAX_MEMORY_ALLOCATED) returns peak HPU memory
    allocated by tensors( in bytes). reset_peak_memory_stats() can be used
    to reset the starting point in tracing stats.
    """
    hpu.init()
    device = _get_device_index(device)
    if device < 0 or device >= hpu.device_count():
        raise AssertionError("Invalid device id")
    return memory_stats(device=device).get("MaxInUse")

def memory_allocated(device: Optional[_device_t] = None) -> int:
    r"""This API (TORCH.HPU.MEMORY_ALLOCATED) returns the current
    HPU memory occupied by tensors.
    """
    hpu.init()
    device = _get_device_index(device)
    if device < 0 or device >= hpu.device_count():
        raise AssertionError("Invalid device id")
    return memory_stats(device=device).get("InUse")

def reset_peak_memory_stats(device: Optional[_device_t] = None) -> None:
    r"""This API (TORCH.HPU.RESET_PEAK_MEMORY_STATS) resets starting point
    of memory occupied by tensors.
    """
    hpu.init()
    device = _get_device_index(device)
    if device < 0 or device >= hpu.device_count():
        raise AssertionError("Invalid device id")
    _hpu_C.reset_peak_memory_stats(device)

def reset_accumulated_memory_stats(device: Optional[_device_t] = None) -> None:
    r"""This API (TORCH.HPU.RESET_ACCUMULATED_MEMORY_STATS) to clear
    number of allocs and number of frees.
    """
    hpu.init()
    device = _get_device_index(device)
    if device < 0 or device >= hpu.device_count():
        raise AssertionError("Invalid device id")
    _hpu_C.clear_memory_stats(device)

def memory_stats(device: Optional[_device_t] = None) -> Dict[str, Any]:
    r"""This API (TORCH.HPU.MEMORY_STATS) returns dict of HPU memory statics.
    Below sample memory stats printout and details
    ('Limit', 3050939105) : amount of total memory on HPU device
    ('InUse', 20073088) : amount of allocated memory at any instance. ( starting point after reset_peak_memroy_stats() )
    ('MaxInUse', 20073088) : amount of total active memory allocated
    ('NumAllocs', 0) : number of allocations
    ('NumFrees', 0) : number of freed chunks
    ('ActiveAllocs', 0) : number of active allocations
    ('MaxAllocSize', 0) : maximum allocated size
    ('TotalSystemAllocs', 34) : total number of system allocations
    ('TotalSystemFrees', 2) : total number of system frees
    ('TotalActiveAllocs', 32)] : total number of active allocations
    """
    hpu.init()
    device = _get_device_index(device)
    if device < 0 or device >= hpu.device_count():
        raise AssertionError("Invalid device id")
    return _hpu_C.get_mem_stats(device)

def memory_summary(device: Optional[_device_t] = None) -> str:
    r"""This API (TORCH.HPU.RESET_ACCUMULATED_MEMORY_STATS) returns
    human readable printout of current memory stats.
    """
    hpu.init()
    device = _get_device_index(device)
    if device < 0 or device >= hpu.device_count():
        raise AssertionError("Invalid device id")
    tbl = []
    tbl.append("=" * 52)
    tbl.append(" {_:5} PyTorch HPU memory summary, device ID {device:<6d} ")
    tbl.append("-" * 52)
    fmt_tbl = {"_": "", "device": device}
    str = "|" + "|\n|".join(tbl).format(**fmt_tbl) + "|\n"
    str1 = _hpu_C.get_memory_summary(device)
    char1 = str1.split("\n")
    return(str +str1)