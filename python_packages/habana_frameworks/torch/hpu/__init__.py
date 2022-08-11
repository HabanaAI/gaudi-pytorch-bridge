import collections
import torch
import warnings
import threading
from habana_frameworks.torch import _hpu_C
from typing import Optional, Union
from ._utils import _get_device_index
from .memory import *
from .streams import *
from .events import *
from .graphs import *
_device_t = Union[torch.device, str, int, None]
_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
import os

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
    if is_initialized() or hasattr(_tls, 'is_initializing'):
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
    if not hasattr(_hpu_C, 'device_count'):
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
    os.environ['PT_ENABLE_HABANA_STREAMASYNC'] = str(debug_mode)
def setDeterministic(val: bool) -> None:
    if not is_initialized():
       init()
    _hpu_C.setDeterministic(val)

def set_autocast_hpu_enabled(enabled) -> None:
    _hpu_C.set_autocast_hpu_enabled(enabled)

def is_autocast_hpu_enabled() -> bool:
    return _hpu_C.is_autocast_hpu_enabled()

def set_autocast_hpu_dtype(dtype) -> None:
    _hpu_C.set_autocast_hpu_dtype(dtype)

def get_autocast_hpu_dtype() -> Any:
    return _hpu_C.get_autocast_hpu_dtype()

