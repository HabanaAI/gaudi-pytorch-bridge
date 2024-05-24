import collections
import ctypes
import warnings
from typing import Any, Dict, Optional, Union

import habana_frameworks.torch as htorch
import torch
from habana_frameworks.torch import _hpu_C
from torch._streambase import _EventBase


class Event(_hpu_C._HpuEventBase, _EventBase):
    r"""Wrapper around a HPU event.

    HPU events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize HPU
    streams.

    The underlying HPU events are lazily initialized when the event is first
    recorded or exported to another process. After creation, only streams on the
    same device may record the event. However, streams on any device can wait on
    the event.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
    """

    def __new__(cls, enable_timing=False):
        return super().__new__(cls, enable_timing=enable_timing)

    def record(self, stream=None):
        r"""Record the event in a given stream.

        Uses ``torch.hpu.current_stream()`` if no stream is specified. The
        stream's device must match the event's device.
        """
        if stream is None:
            stream = htorch.hpu.current_stream()
        super().record(stream)

    def wait(self, stream=None):
        r"""Make all future work submitted to the given stream wait for this event.

        Use ``torch.hpu.current_stream()`` if no stream is specified.

        .. note:: This is a wrapper around ``hpuStreamWaitEvent()``: see
            `HPU Event documentation`_ for more info.
        """
        if stream is None:
            stream = htorch.hpu.current_stream()
        super().wait(stream)

    def query(self):
        r"""Check if all work currently captured by event has completed.

        Returns:
            A boolean indicating if all work currently captured by event has
            completed.
        """
        return super().query()

    def elapsed_time(self, end_event):
        r"""Return the time elapsed.

        Time reported in milliseconds after the event was recorded and
        before the end_event was recorded.
        """
        return super().elapsed_time(end_event) / 1e6

    def synchronize(self):
        r"""Wait for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.

         .. note:: This is a wrapper around ``hpuEventSynchronize()``: see
            `HPU Event documentation`_ for more info.
        """
        super().synchronize()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.hpu_event)

    def __repr__(self):
        if self.hpu_event:
            return f"<htorch.hpu.Event {self._as_parameter_.value:#x}>"
        else:
            return "<htorch.hpu.Event uninitialized>"
