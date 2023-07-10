###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import collections
import os
from torch.types import Device
import threading
import warnings
from typing import Any, List, Optional, Union

import torch
from habana_frameworks.torch import _hpu_C
from habana_frameworks.torch.utils.internal import is_lazy

from ._utils import (
    HABANA_VISIBLE_MODULES_VAR,
    HLS_MODULE_ID_VAR,
    _get_available_modules_from_environ,
    _get_device_index,
    _get_module_id_from_environ,
)
from .events import *
from .memory import *
from .metrics import *
from .streams import *

if is_lazy():
    from .graphs import *

_device_t = Union[torch.device, str, int, None]
_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()


def init() -> None:
    r"""Initialize PyTorch's HPU state.  You may need to call
    this explicitly if you are interacting with PyTorch via
    its C API, as Python bindings for HPU functionality will not
    be available until this initialization takes place.  Ordinary users
    should not need this, as all of PyTorch's HPU methods
    automatically initialize HPU state on-demand.

    Does nothing if the HPU state is already initialized.
    """
    global _initialized
    if is_initialized() or hasattr(_tls, "is_initializing"):
        return
    with _initialization_lock:
        # We be double-checked locking. This is OK because
        # the above test was GIL protected anyway.  The inner test
        # is for when a thread blocked on some other thread which was
        # doing the initialization; when they get the lock, they will
        # find there is nothing left to do.
        if is_initialized():
            return
        # This function throws if there's a driver initialization error, no HPUs
        # are found or any other error occurs
        _hpu_C.init()
        # hpu does not support queued calls currenlty, so no
        # _tls.is_initializing = True
        # process all the queud calls and then set the _tls.is_initializing = false
        _initialized = True


def is_initialized() -> bool:
    r"""Returns whether PyTorch's HPU state has been initialized."""
    return _initialized


def is_available() -> bool:
    r"""Returns a bool indicating if HPU is currently available."""
    if not hasattr(_hpu_C, "device_count"):
        return False
    # This function never throws and returns 0 if driver is missing or can't
    # be initialized
    return _hpu_C.device_count() > 0


def device_count() -> int:
    r"""Returns the number of HPUs available."""
    if is_available():
        return _hpu_C.device_count()
    else:
        return 0


def get_device_name(device: Optional[_device_t] = None) -> str:
    r"""Gets the name of a device.

    Args:
        device (torch.device or int, optional): device for which to return the
            name. This function is a no-op if this argument is a negative
            integer. It uses the current device,
            if :attr:`device` is ``None`` (default).

    Returns:
        str: the name of the device
    """

    if not is_available():
        warnings.warn("Device not available")
        return ""

    init()
    device = _get_device_index(device)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    return _hpu_C.get_device_name(device)


def current_device() -> int:
    r"""Returns the index of a currently selected device."""
    init()
    return _hpu_C.current_device()


def synchronize() -> None:
    r"""Waits for all kernels in all streams on a HPU device to complete."""
    init()
    return _hpu_C.synchronize_device()


def set_sync_debug_mode(debug_mode) -> None:
    r"""Enable/Disable Asynchronous Streams for debug.
     Args:
        debug_mode: True/False
    ."""
    os.environ["PT_ENABLE_HABANA_STREAMASYNC"] = str(debug_mode)


def get_sync_debug_mode() -> int:
    r"""Returns current value of debug mode for Asynchronous Streams."""

    import os

    return int(os.environ["PT_ENABLE_HABANA_STREAMASYNC"])


def setDeterministic(val: bool) -> None:
    _hpu_C.setDeterministic(val)


def set_autocast_hpu_enabled(enabled) -> None:
    _hpu_C.set_autocast_hpu_enabled(enabled)


def is_autocast_hpu_enabled() -> bool:
    return _hpu_C.is_autocast_hpu_enabled()


def set_autocast_hpu_dtype(dtype) -> None:
    _hpu_C.set_autocast_hpu_dtype(dtype)


def get_autocast_hpu_dtype() -> Any:
    return _hpu_C.get_autocast_hpu_dtype()


def enable_dynamic_shape():
    _hpu_C.enable_dynamic_shape()


def disable_dynamic_shape():
    _hpu_C.disable_dynamic_shape()


def is_bf16_supported():
    r"""Check if bf16 is supported."""
    if is_available():
        return True
    else:
        return False


def get_device_capability(device: Optional[_device_t] = None) -> str:
    if not is_available():
        warnings.warn("Device not available")
        return ""

    init()
    device = _get_device_index(device)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    return _hpu_C.get_device_capability()


def get_device_properties(device: Optional[_device_t] = None) -> str:
    if not is_available():
        warnings.warn("Device not available")
        return ""

    init()
    device = _get_device_index(device)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    return _hpu_C.get_device_properties(device)


def can_device_access_peer(device: _device_t, peer_device: _device_t) -> bool:
    if not is_available():
        warnings.warn("Device not available")
        return ""
    init()
    device = _get_device_index(device)
    peer_device = _get_device_index(peer_device)
    count = device_count()
    if device < 0 or device >= count:
        raise AssertionError("Invalid device id : {}".format(device))
    if peer_device < 0 or peer_device >= count:
        raise AssertionError("Invalid device id : {}".format(peer_device))
    if device == peer_device:
        raise AssertionError("Both the ids are same.")
    if device <= count and peer_device <= count:
        return True
    else:
        return False


def get_gencode_flags() -> str:
    r"""Returns the gencode flags the library is compiled with."""
    return ""


def get_arch_list() -> List[str]:
    r"""Returns the architecture the library is compiled with"""
    arch_list = []
    device = current_device()
    device_name = get_device_name(device)
    arch_list.append(device_name)
    return arch_list


def set_device(device: _device_t) -> None:
    r"""Sets the current device"""
    device_idx = _get_device_index(device)
    # hack to match torch.cuda API
    available_modules = _get_available_modules_from_environ()
    if device_idx > len(available_modules):
        raise AssertionError(
            f"Trying to open device with idx={device_idx} when only {len(available_modules)} are avaliable)"
        )

    requested_module_id = available_modules[device_idx]
    current_module_id = _get_module_id_from_environ()

    if current_module_id >= 0:
        if int(current_module_id) != int(requested_module_id):
            raise AssertionError(
                f"Requested module_id={requested_module_id} is different from current_module_id={current_module_id}"
                f" which was previously set."
            )

    os.environ[HLS_MODULE_ID_VAR] = available_modules[device_idx]
    set_device.current_device_idx = device_idx


set_device.current_device_idx = -1


class device(object):
    r"""Context manager that changes the selected device."""

    def __init__(self, device: Any):
        self.idx = _get_device_index(device)
        self.prev_idx = -1

    def __enter__(self):
        # hack to match the behavior of torch.cuda APIs
        self.prev_idx = set_device.current_device_idx
        if self.idx == -1:
            return
        if self.idx != self.prev_idx:
            set_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        if self.prev_idx != self.idx and self.prev_idx != -1:
            set_device(self.idx)
        return False


class device_of(device):
    r"""Context manager that changes the current device of the given object"""

    def __init__(self, obj):
        idx = obj.get_device() if obj.is_hpu else -1
        super(device_of, self).__init__(idx)


def memory_usage(device: Optional[Union[Device, int]] = None) -> int:
    r"""Returns the memory used. as given by `hl-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    init()
    device_idx = _get_device_index(device)
    if device_idx < 0 or device_idx >= device_count():
        raise AssertionError("Invalid device id")
    return _hpu_C.get_mem_stats(device_idx)["InUse"]


def utilization(device: Optional[Union[Device, int]] = None) -> int:
    r"""Returns the usage as given by `hl-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    init()
    device_idx = _get_device_index(device)
    if device_idx < 0 or device_idx >= torch.hpu.device_count():
        raise AssertionError("Invalid device id")
    try:
        import pyhlml  # type: ignore[import]
    except ModuleNotFoundError:
        raise ModuleNotFoundError("pyhlml module not found, please install pyhlml")
    pyhlml.hlmlInit()
    pyhlml_device = pyhlml.hlmlDeviceGetHandleByIndex(device_idx)
    usage = pyhlml.hlmlDeviceGetUtilizationRates(pyhlml_device)
    pyhlml.hlmlShutdown()
    return usage


def _create_tensor_alias(name, dtype):
    def tensor_alias(*args, **kwargs):
        if "device" in kwargs:
            raise TypeError(f"hpu.{name}() got an unexpected keyword argument 'device'")
        if "dtype" in kwargs:
            raise TypeError(f"hpu.{name}() got an unexpected keyword argument 'dtype'")
        kwargs["device"] = "hpu"
        kwargs["dtype"] = dtype
        return torch.tensor(*args, **kwargs)

    return tensor_alias


BFloat16Tensor = _create_tensor_alias("BFloat16Tensor", torch.bfloat16)
BoolTensor = _create_tensor_alias("BoolTensor", torch.bool)
ByteTensor = _create_tensor_alias("ByteTensor", torch.uint8)
CharTensor = _create_tensor_alias("CharTensor", torch.int8)
DoubleTensor = _create_tensor_alias("DoubleTensor", torch.float64)
FloatTensor = _create_tensor_alias("FloatTensor", torch.float32)
HalfTensor = _create_tensor_alias("HalfTensor", torch.float16)
IntTensor = _create_tensor_alias("IntTensor", torch.int32)
LongTensor = _create_tensor_alias("LongTensor", torch.int64)
ShortTensor = _create_tensor_alias("ShortTensor", torch.int16)
