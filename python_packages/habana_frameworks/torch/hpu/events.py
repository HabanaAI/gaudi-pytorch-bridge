import collections
import torch
import warnings
from typing import Any, Dict, Union, Optional
import habana_frameworks.torch as htorch
from habana_frameworks.torch import _hpu_C


class Event:
    r"""Wrapper around a HPU event.
    events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize HPU
    streams.
    After creation, only streams on the same device may record the event.
    Args:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
    """

    def __init__(self, enable_timing=False):
        if not htorch.hpu.is_initialized():
            htorch.hpu.init()

        self.event = _hpu_C.get_event(enable_timing)

    def record(self, stream=None):
        r"""Records the event in a given stream.
        Uses ``htorch.hpu.current_stream()`` if no stream is specified.
        """
        if stream is None:
            stream = htorch.hpu.current_stream()
        _hpu_C.event_record(self.event, stream.stream)

    def wait(self, stream=None):
        r"""Makes all future work submitted to the given stream wait for this
        event.
        Use ``htorch.hpu.current_stream()`` if no stream is specified.
        """
        if stream is None:
            stream = htorch.hpu.current_stream()
        _hpu_C.event_wait(self.event, stream.stream)

    def query(self):
        r"""Checks if all work currently captured by event has completed.
        Returns:
            A boolean indicating if all work currently captured by event has
            completed.
        """
        return _hpu_C.event_query(self.event)

    def elapsed_time(self, other):
        r"""Returns the time elapsed in milliseconds after the event was
        recorded and before the end_event was recorded.
        """
        assert isinstance(other, Event), "other is not of type Event"
        return _hpu_C.elapsed_time(self.event, other.event) / 1e6

    def synchronize(self):
        r"""Waits for the event to complete.
        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.
        """
        _hpu_C.event_synchronize(self.event)

    def __repr__(self):
        info = _hpu_C.get_event_info(self.event)
        return "<htorch.hpu.Event device={0} is_recorded={1:#x}>".format(info[0], info[1])
