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

import contextlib
import os
import threading
import warnings
from typing import Any, List, Optional, Union

import torch
from habana_frameworks.torch import _hpu_C
from habana_frameworks.torch.utils.internal import is_lazy
from torch.types import Device
from torch.utils.checkpoint import DefaultDeviceType

from ._proxy_module import *
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
from .random import *
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

    # Upstream PT commit https://github.com/pytorch/pytorch/commit/6aeb85a
    # introduced checkpoint support for all non-cpu devices.
    # However, default checkpointing device is 'cuda', so to avoid changing models,
    # we should set default device type to 'hpu'.
    DefaultDeviceType.set_device_type("hpu")


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
    device = _get_device_index(device, optional=True)
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
    warnings.warn(
        "torch.hpu.setDeterministic is deprecated and will be removed in next release. Please use torch.use_deterministic_algorithms instead."
    )
    _hpu_C.setDeterministic(val)


def getDeterministic() -> bool:
    warnings.warn(
        "torch.hpu.getDeterministic is deprecated and will be removed in next release. Please use torch.are_deterministic_algorithms_enabled instead."
    )
    return _hpu_C.getDeterministic()


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


def get_dynamic_shape_status() -> bool:
    return _hpu_C.get_dynamic_shape_status()


def enable_optim_output_sif():
    _hpu_C.enable_optim_output_sif()


def disable_optim_output_sif():
    _hpu_C.disable_optim_output_sif()


def enable_inference_mode():
    _hpu_C.enable_inference_mode()


def disable_inference_mode():
    _hpu_C.disable_inference_mode()


def enable_quantization():
    _hpu_C.enable_quantization()


def disable_quantization():
    _hpu_C.disable_quantization()


def enable_const_section_serialization(path, clear_path, use_compression):
    _hpu_C.enable_const_section_serialization(str(path), clear_path, use_compression)


def disable_const_section_serialization():
    _hpu_C.enable_const_section_serialization("", False, False)


def enable_matmul3d_2d_reshape():
    _hpu_C.enable_matmul3d_2d_reshape()


def disable_matmul3d_2d_reshape():
    _hpu_C.disable_matmul3d_2d_reshape()


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
    device = _get_device_index(device, optional=True)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    return _hpu_C.get_device_capability()


def get_device_properties(device: Optional[_device_t] = None) -> str:
    if not is_available():
        warnings.warn("Device not available")
        return ""

    init()
    device = _get_device_index(device, optional=True)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    return _hpu_C.get_device_properties(device)


def can_device_access_peer(device: _device_t, peer_device: _device_t) -> bool:
    if not is_available():
        warnings.warn("Device not available")
        return ""
    init()
    device = _get_device_index(device, optional=True)
    peer_device = _get_device_index(peer_device, optional=True)
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
    device_idx = _get_device_index(device, optional=True)
    # hack to match torch.cuda API
    available_modules = _get_available_modules_from_environ()
    if device_idx not in available_modules and device_idx > len(available_modules):
        raise AssertionError(
            f"Trying to open device with idx={device_idx} when only {available_modules} are avaliable)"
        )

    requested_module_id = device_idx
    current_module_id = _get_module_id_from_environ()

    if current_module_id >= 0:
        if int(current_module_id) != int(requested_module_id):
            requested_module_id = current_module_id

        for index in range(device_count()):
            if available_modules[index] == current_module_id:
                device_idx = index
                break

        assert (
            available_modules[device_idx] == current_module_id
        ), f"Requested module_id={available_modules[device_idx]} is different from current_module_id={current_module_id} which was previously set."

    if current_module_id == -1 and HABANA_VISIBLE_MODULES_VAR not in os.environ and device_count() < 8:
        # As HLS_MODULE_ID is not set and HABANA_VISIBLE_MODULES is not provided
        # by user:
        # - the only supported device idx is 0
        # - Module ID (HLS_MODULE_ID) can't be set as we don't know what Module
        #   IDs are available in system. In a result device will be allocated by
        #   type
        assert device_idx == 0, f"As {HABANA_VISIBLE_MODULES_VAR} is not provided, the only supported device idx is 0."
    else:
        os.environ[HLS_MODULE_ID_VAR] = str(available_modules[device_idx])

    set_device.current_device_idx = device_idx


set_device.current_device_idx = -1


class device(object):
    r"""Context manager that changes the selected device."""

    def __init__(self, device: Any):
        # After 2.1 upgrade, device coming from fork might be 0
        device_idx = _get_device_index(device, optional=True)
        env_device_idx = _get_module_id_from_environ()
        if device_idx != 0 and device_idx != env_device_idx:
            raise AssertionError(f"Requested device_id={device_idx} is different from env_device_id={env_device_idx}")
        self.idx = env_device_idx
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
    device_idx = _get_device_index(device, optional=True)
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
    device_idx = _get_device_index(device, optional=True)
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
    target_device = "hpu"

    class TypeFabric(torch.Tensor):
        @staticmethod
        def __new__(cls, *args, **kwargs):  # no __init__ due torch.Tensor is C object
            input_device = kwargs.get("device", None)
            if input_device is not None and input_device != target_device:
                raise RuntimeError(
                    f"legacy constructor expects device type: {target_device} but device type: {input_device} was passed"
                )

            input_dtype = kwargs.get("dtype", None)
            if input_dtype is not None and input_dtype != dtype:
                raise RuntimeError(f"legacy constructor expects dtype: {dtype} but dtype: {input_dtype} was passed")

            # Object of this type has fixed "device" and "dtype"
            kwargs["device"] = target_device
            kwargs["dtype"] = dtype

            # this always copy data
            data = torch.tensor(*args, **kwargs)
            result = torch.Tensor._make_subclass(cls, data)

            return result

    TypeFabric.__name__ = name
    TypeFabric.__qualname__ = name  # python 3 compatibility

    return TypeFabric


def enable_recompute_sdp(enabled: bool):
    r"""User control to enable or disable recompute based fused SDPA
    enabled = True -> Fused SDPA with recompute
    enabled = False -> Fused SDPA without recompute
    """
    _hpu_C.enable_recompute_FSDPA(enabled)


def recompute_sdp_enabled():
    r"""User control to check if recompute based fused SDPA is enabled.
    return = True -> Fused SDPA with recompute enabled
    return = False -> Fused SDPA without recompute enabled
    """
    return _hpu_C.is_recompute_FSDPA_enabled()


@contextlib.contextmanager
def sdp_kernel(
    enable_recompute: bool = True,
):
    r"""Context manager to enable or disable recompute based fused SDPA
    enable_recompute = True -> Fused SDPA with recompute
    enable_recompute = False -> Fused SDPA without recompute
    """
    recompute_backup: bool = recompute_sdp_enabled()

    try:
        enable_recompute_sdp(enable_recompute)
        yield {}
    finally:
        enable_recompute_sdp(recompute_backup)


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
