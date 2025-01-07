###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################


import builtins
import collections
import ctypes
import inspect
import os
import warnings
from typing import Any, Dict, List, Optional, Union

import habana_frameworks.torch as htorch
import torch
from habana_frameworks.torch import _hpu_C
from torch._streambase import _EventBase, _StreamBase

from ._utils import _get_device_index

_int = builtins.int

is_lazy_mode = os.getenv("PT_HPU_LAZY_MODE", "1") != "0"


class _device:
    type: str  # THPDevice_type
    index: _int  # THPDevice_index

    def __init__(self, type: str, index: _int):
        self.type = type
        self.index = index

    def __eq__(self, other):
        if not isinstance(other, _device):
            return False
        return self.type == other.type and self.index == other.index


_device_t = Union[_device, str, int, None]


class Stream(_hpu_C._HpuStreamBase, _StreamBase):
    r"""Wrapper around a HPU stream.

    A HPU stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams.  See :ref:`HPU-semantics` for
    details.

    Args:
        device: Unused parameter as HPU as only 1 device per process is supported
        priority: Unused parameter as only low priority streams are supported

    """

    def __new__(cls, device=None, priority=0, **kwargs):
        if not htorch.hpu.is_initialized():
            htorch.hpu.init()
        if "device_index" in kwargs and not isinstance(kwargs["device_index"], int):
            kwargs["device_index"] = _get_device_index(kwargs["device_index"])
        return super(Stream, cls).__new__(cls, priority=priority, **kwargs)

    def wait_event(self, event):
        r"""Makes all future work submitted to the stream wait for an event.

        Args:
            event (htorch.hpu.Event): an event to wait for.

           This function returns without waiting for :attr:`event`: only future
           operations are affected.
        """
        event.wait(self)

    def wait_stream(self, stream):
        r"""Synchronizes with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Args:
            stream (Stream): a stream to synchronize.

        .. note:: This function returns without waiting for currently enqueued
           kernels in :attr:`stream`: only future operations are affected.
        """
        self.wait_event(stream.record_event())

    def record_event(self, event=None):
        r"""Records an event.

        Args:
            event (htorch.hpu.Event, optional): event to record. If not given, a new one
                will be allocated.

        Returns:
            Recorded event.
        """
        if event is None:
            event = htorch.hpu.Event()
        event.record(self)
        return event

    def query(self):
        r"""Checks if all the work submitted  on the stream has been completed.

        Returns:
            A boolean indicating if all kernels in this stream are completed."""

        return super().query()

    def synchronize(self):
        r"""Wait for all the kernels in this stream to complete."""
        super().synchronize()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.hpu_stream)

    def __eq__(self, o):
        if isinstance(o, Stream):
            return super().__eq__(o)
        return False

    def __hash__(self):
        return hash((self.hpu_stream, self.device))

    def __repr__(self):
        return f"<torch.hpu.Stream device={self.device} hpu_stream={self.hpu_stream:#x}>"

    @property
    def device_index(self):
        return self.device

    def id(self):
        return self.stream_id


class StreamContext:
    r"""Context-manager that selects a given stream.

    All hpu kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """

    cur_stream: Optional["torch.hpu.Stream"]

    def __init__(self, stream: Optional["torch.hpu.Stream"]):
        self.stream = stream
        self.idx = _get_device_index(None, True)
        if not torch.jit.is_scripting():
            if self.idx is None:
                self.idx = -1

        self.src_prev_stream = None if not torch.jit.is_scripting() else torch.hpu.default_stream(None)
        self.dst_prev_stream = None if not torch.jit.is_scripting() else torch.hpu.default_stream(None)

    def __enter__(self):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # Return if stream is None
        if cur_stream is None:
            return
        self.src_prev_stream = current_stream()
        if self.src_prev_stream.stream_id == self.stream.stream_id:
            return
        htorch.hpu.set_stream(cur_stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # If stream is None or no hpu device available, return
        if cur_stream is None or self.idx == -1:
            return

        if self.src_prev_stream.stream_id == self.stream.stream_id:
            return

        # Reset the stream on the original device
        # and destination device
        # if self.src_prev_stream.device != cur_stream.device:  # type: ignore[union-attr]
        #    torch.hpu.set_stream(self.dst_prev_stream)  # type: ignore[arg-type]
        torch.hpu.set_stream(self.src_prev_stream)  # type: ignore[arg-type]


def stream(stream) -> StreamContext:
    r"""Wrapper around the Context-manager StreamContext that
    selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    ..Note:: In eager mode stream is of type Stream class while in JIT it is
    an object of the custom class ``torch.classes.hpu.Stream``.
    """
    return StreamContext(stream)


def _set_stream_by_id(stream_id, device_index, device_type):
    r"""set stream specified by the stream id, device index and
        device type

    Args: stream_id (int): stream id in stream pool
          device_index (int): device index in topo
          device_type (int): enum device type
    """
    _hpu_C._hpu_setStream(
        stream_id=stream_id,
        device_index=device_index,
        device_type=device_type,
    )


# Global variable to cache the current stream
_cached_stream = None


def set_stream(stream):
    r"""Sets the current stream.This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.

    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    global _cached_stream

    if stream is None:
        return

    if is_lazy_mode:
        if _cached_stream and _cached_stream.stream_id == stream.stream_id:
            return
        _cached_stream = stream

    device_idx = stream.device_index
    if not isinstance(device_idx, int):
        device_idx = _get_device_index(stream.device_index)

    _set_stream_by_id(
        stream_id=stream.stream_id,
        device_index=device_idx,
        device_type=stream.device_type,
    )


def current_stream(device: Optional[_device_t] = None) -> Stream:
    r"""Gets the current stream.
    Args:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.hpu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    global _cached_stream

    # If a stream is cached, return it
    if is_lazy_mode:
        if _cached_stream is not None:
            return _cached_stream

    streamdata = _hpu_C._hpu_getCurrentStream(_get_device_index(device, optional=True))
    stream = Stream(
        stream_id=streamdata[0],
        device_index=streamdata[1],
        device_type=streamdata[2],
        is_default_stream=(streamdata[0] == 0),
    )

    if is_lazy_mode:
        _cached_stream = stream

    return stream


def default_stream(device: Optional[_device_t] = None) -> Stream:
    r"""Gets the default stream on HPU device.This is a wrapper API to get the stream.
    Args:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch.hpu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    streamdata = _hpu_C._hpu_getDefaultStream(_get_device_index(device, optional=True))
    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2], is_default_stream=True
    )


def get_stream_info(stream: Stream):
    r"""Gets the info for HPU stream.
    Args:
        stream.
    """
    device_idx = stream.device_index
    if not isinstance(device_idx, int):
        device_idx = _get_device_index(stream.device_index)
    return _hpu_C._hpu_getStreamInfo(
        stream_id=stream.stream_id, device_index=device_idx, device_type=stream.device_type
    )


def record_stream(self, stream):
    device_idx = stream.device_index
    if not isinstance(device_idx, int):
        device_idx = _get_device_index(stream.device_index)
    return _hpu_C.record_stream(self, stream.stream_id, device_idx, stream.device_type)
