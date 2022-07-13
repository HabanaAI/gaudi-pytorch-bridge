import collections
import torch
import warnings
from typing import Any, Dict, Union, Optional
import habana_frameworks.torch as htorch
from habana_frameworks.torch import _hpu_C
from ._utils import _get_device_index
import ctypes

class Stream(object):
    r"""Wrapper around a HPU stream.

    A HPU stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams.  See :ref:`HPU-semantics` for
    details.

    Args:
        device: Unused parameter as HPU as only 1 device per process is supported
        priority: Unused parameter as only low priority streams are supported

    """

    def __init__(self, device=None, priority=0, provided_stream=None):
        if provided_stream is not None:
            self.stream = provided_stream
        else:
            self.device = -1
            device = _get_device_index(device)

            self.device = device
            self.isHighPriorityStream = priority < 0
            self.stream = _hpu_C.get_stream(self.isHighPriorityStream,self.device)

    def query(self):
        r"""Checks if all the work submitted  on the stream has been completed.

        Returns:
            A boolean indicating if all kernels in this stream are completed."""
        return _hpu_C.query(self.stream)


    def synchronize(self):
        r"""Wait for all the kernels in this stream to complete.

        """
        _hpu_C.synchronize(self.stream)


class StreamContext(object):
    r"""Context-manager that selects a given stream.
    All hpu kernels queued within its context will be enqueued on a selected
    stream.
    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """

    def __init__(self, stream):
        self.stream = stream
        self.prev_stream = None

    def __enter__(self):

        cur_stream = self.stream
        # Return if stream is None 
        if cur_stream is None:
            return
        self.prev_stream =_hpu_C.get_current_stream()
        htorch.hpu.set_stream(cur_stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        cur_stream = self.stream
        # If stream is None  return
        if cur_stream is None:
            return
        htorch.hpu.set_stream(self.prev_stream)  # type: ignore[arg-type]

def stream(stream) -> StreamContext:
    r"""Wrapper around the Context-manager StreamContext that
    selects a given stream.
    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    """
    if stream is None:
        return StreamContext(None)

    return StreamContext(stream.stream)

def set_stream(in_stream):
    r"""Sets the current stream.This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.
    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    if in_stream is None:
        return
    if isinstance(in_stream,Stream):
        stream = in_stream.stream
    else:
        stream = in_stream
    _hpu_C.set_current_stream(stream)

def current_stream():
    r"""Gets the current stream.
    Args:
        None.
    """
    #print('STREAMS: current_stream')
    return Stream(provided_stream = _hpu_C.get_current_stream())

def default_stream():
    r"""Gets the default stream on HPU device.This is a wrapper API to get the stream.
    Args:
        None.
    """
    return Stream(provided_stream = _hpu_C.get_default_stream())
