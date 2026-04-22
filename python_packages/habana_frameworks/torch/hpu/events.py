###############################################################################
# Copyright (c) 2021-2026 Intel Corporation
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


import ctypes

import habana_frameworks.torch as htorch
from habana_frameworks.torch import _hpu_C


class Event(_hpu_C._HpuEventBase):
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


def _patch_dynamo_event_wait_for_hpu() -> None:
    from torch._dynamo.utils import proxy_args_kwargs
    from torch._dynamo.variables import streams as dynamo_streams
    from torch._dynamo.variables.constant import ConstantVariable

    if getattr(dynamo_streams.EventVariable, "_hpu_wait_event_patched", False):
        return

    orig_call_method = dynamo_streams.EventVariable.call_method

    def _call_method(self, tx, name, args, kwargs):
        if name == "wait":
            stream = dynamo_streams.EventVariable._get_stream_arg(tx, args, kwargs)
            tx.output.create_proxy(
                "call_method",
                "wait_event",
                *proxy_args_kwargs([stream, self], {}),
            )
            return ConstantVariable(None)
        return orig_call_method(self, tx, name, args, kwargs)

    dynamo_streams.EventVariable.call_method = _call_method
    dynamo_streams.EventVariable._hpu_wait_event_patched = True


_patch_dynamo_event_wait_for_hpu()
