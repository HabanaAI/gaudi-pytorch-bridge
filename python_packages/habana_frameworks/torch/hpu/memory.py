###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################


import warnings
from typing import Any

from habana_frameworks.torch import _hpu_C, hpu

import torch

from ._utils import _get_device_index

_device_t = torch.device | str | int | None


def max_memory_allocated(device: _device_t | None = None) -> int:
    r"""This API (TORCH.HPU.MAX_MEMORY_ALLOCATED) returns peak HPU memory
    allocated by tensors( in bytes). reset_peak_memory_stats() can be used
    to reset the starting point in tracing stats.
    """
    hpu.init()
    device = _get_device_index(device, optional=True)
    if device < 0 or device >= hpu.device_count():
        raise AssertionError("Invalid device id")
    return memory_stats(device=device).get("MaxInUse")


def memory_allocated(device: _device_t | None = None) -> int:
    r"""This API (TORCH.HPU.MEMORY_ALLOCATED) returns the current
    HPU memory occupied by tensors.
    """
    hpu.init()
    device = _get_device_index(device, optional=True)
    if device < 0 or device >= hpu.device_count():
        raise AssertionError("Invalid device id")
    return memory_stats(device=device).get("InUse")


def reset_peak_memory_stats(device: _device_t | None = None) -> None:
    r"""This API (TORCH.HPU.RESET_PEAK_MEMORY_STATS) resets starting point
    of memory occupied by tensors.
    """
    hpu.init()
    device = _get_device_index(device, optional=True)
    if device < 0 or device >= hpu.device_count():
        raise AssertionError("Invalid device id")
    _hpu_C.reset_peak_memory_stats(device)


def reset_accumulated_memory_stats(device: _device_t | None = None) -> None:
    r"""This API (TORCH.HPU.RESET_ACCUMULATED_MEMORY_STATS) to clear
    number of allocs and number of frees.
    """
    hpu.init()
    device = _get_device_index(device, optional=True)
    if device < 0 or device >= hpu.device_count():
        raise AssertionError("Invalid device id")
    _hpu_C.clear_memory_stats(device)


def memory_stats(device: _device_t | None = None) -> dict[str, Any]:
    r"""This API (TORCH.HPU.MEMORY_STATS) returns dict of HPU memory statics.
    Below sample memory stats printout and details
    ('Limit', 3050939105) : amount of total reserved memory on HPU device
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
    device = _get_device_index(device, optional=True)
    if device < 0 or device >= hpu.device_count():
        raise AssertionError("Invalid device id")
    return _hpu_C.get_mem_stats(device)


def _format_memory_summary(summary: dict) -> str:
    NON_MEMORY_KEYS = ["num_allocs", "num_free", "active_allocs"]
    GB = 1024 * 1024 * 1024
    LINE_LENGTH = 50
    tbl = []
    tbl.append("=" * 52)
    tbl.append(" {_:5} PyTorch HPU memory summary, device ID {device:<6d} ")
    tbl.append("-" * 52)
    fmt_tbl = {"_": "", "device": 0}
    header = "|" + "|\n|".join(tbl).format(**fmt_tbl) + "|\n"
    formatted_summary = ""
    for k, v in summary.items():
        line = "  "
        label = str(k) + ":"
        value = str(v) + f" ({v / GB:.2f}) GB" if k not in NON_MEMORY_KEYS else ""
        line += label + " " * (LINE_LENGTH - (len(label) + len(value))) + value + "  \n"
        formatted_summary += line

    return header + formatted_summary


def memory_summary(device: _device_t | None = None) -> str:
    r"""This API (TORCH.HPU.RESET_ACCUMULATED_MEMORY_STATS) returns
    human readable printout of current memory stats.
    """
    hpu.init()
    device = _get_device_index(device, optional=True)
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
    return str + str1


def _extended_memory_summary_dict(device: _device_t | None = None) -> dict:
    hpu.init()
    return _hpu_C.get_extended_memory_summary()


def _extended_memory_summary(device: _device_t | None = None) -> str:
    return _format_memory_summary(_extended_memory_summary_dict())


def _get_hlml_shared_object_name(device: _device_t | None = None) -> str:
    if device is None:
        device = 0
    return _hpu_C.get_hlml_shared_object_name(device)


def formatted_memory_stats(device: _device_t | None = None) -> str:
    r"""This API returns string of HPU memory statics.
    Below sample memory stats printout and details
    Limit:- 135960424448 : amount of total reserved memory on HPU device
    InUse:- 7340288 : amount of allocated memory at any instance. ( starting point after reset_peak_memroy_stats() )
    MaxInUse:- 7340288 : amount of total active memory allocated
    NumAllocs:- 11 : number of allocations
    NumFrees:- 2 : number of freed chunks
    ActiveAllocs:- 9 : number of active allocations
    MaxAllocSize:- 4194304 : maximum allocated size
    TotalSystemAllocs:- 12 : total number of system allocations
    TotalSystemFrees:- 2 : total number of system frees
    TotalActiveAllocs:- 10 : total number of active allocations"""

    def format_memory(key, value):
        if key in ["Limit", "InUse", "MaxInUse", "MaxAllocSize"]:
            formatted_values = [f"{value}"]

            if value >= 1024:  # 1 KB
                formatted_values.append(f"{value / 1024:.2f}KB")

            if value >= 1_048_576:  # 1 MB
                formatted_values.append(f"{value / (1024 * 1024):.2f}MB")

            if value >= 1_073_741_824:  # 1 GB
                formatted_values.append(f"{value / (1024 * 1024 * 1024):.2f}GB")

            return " / ".join(formatted_values)
        else:
            return str(value)

    stats = memory_stats(device)
    device_index = _get_device_index(device, optional=True)
    header = f"    Memory Statistics (Device ID) {' ' * 9} :- {device_index:<3}"
    stats_text = "\n".join(f"    {k:<39} :- {format_memory(k, v)}" for k, v in stats.items())
    return f"{header}\n{stats_text}"


def memory_reserved(device: _device_t | None = None) -> int:
    r"""Returns the current HPU memory managed by caching allocator in bytes for a given device."""
    stats = memory_stats(device)
    return stats["Limit"]


def max_memory_reserved(device: _device_t | None = None) -> int:
    r"""Returns the maximum HPU memory managed by caching allocator in bytes for a given device."""
    stats = memory_stats(device)
    return stats["Limit"]


def memory_cached(device: _device_t | None = None) -> int:
    r"""Deprecated same as memory_reserved"""
    warnings.warn("torch.hpu.memory_cached has been renamed to torch.hpu.memory_reserved", FutureWarning)
    return memory_reserved(device)


def max_memory_cached(device: _device_t | None = None) -> int:
    r"""Deprecated: same as max_memory_reserved"""
    warnings.warn("torch.hpu.max_memory_cached has been renamed to torch.hpu.max_memory_reserved", FutureWarning)
    return max_memory_reserved(device)


def mem_get_info(device: _device_t | None = None) -> tuple:
    r"""Returns the free and total memory occupied by a HPU device"""
    stats = memory_stats(device)
    return (stats["Limit"] - stats["InUse"], stats["Limit"])
