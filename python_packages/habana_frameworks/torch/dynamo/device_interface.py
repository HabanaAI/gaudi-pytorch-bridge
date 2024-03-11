###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import inspect
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union

import torch
from torch._streambase import _EventBase, _StreamBase

get_hpu_stream: Optional[Callable[[int], int]]

import habana_frameworks.torch as htorch
from habana_frameworks.torch import _hpu_C
from habana_frameworks.torch._hpu_C import _hpu_getCurrentRawStream as get_hpu_stream

_device_t = Union[torch.device, str, int, None]

# Recording the device properties in the main process but used in worker process.
caching_worker_device_properties: Dict[str, Any] = {}
caching_worker_current_devices: Dict[str, int] = {}


class DeviceInterfaceMeta(type):
    def __new__(metacls, *args, **kwargs):
        class_member = args[2]
        if "Event" in class_member:
            assert inspect.isclass(class_member["Event"]) and issubclass(
                class_member["Event"], _EventBase
            ), "DeviceInterface member Event should be inherit from _EventBase"
        if "Stream" in class_member:
            assert inspect.isclass(class_member["Stream"]) and issubclass(
                class_member["Stream"], _StreamBase
            ), "DeviceInterface member Stream should be inherit from _StreamBase"
        return super().__new__(metacls, *args, **kwargs)


class DeviceInterface(metaclass=DeviceInterfaceMeta):
    """
    This is a device runtime interface for registering to pytorch.
    """

    class device:
        def __new__(cls, device: _device_t):
            raise NotImplementedError()

    class Worker:
        """
        Worker API to query device properties that will work in multi processing
        workers that cannot use the GPU APIs (due to processing fork() and
        initialization time issues). Properties are recorded in the main process
        before we fork the workers.
        """

        @staticmethod
        def set_device(device: int):
            raise NotImplementedError()

        @staticmethod
        def current_device() -> int:
            raise NotImplementedError()

        @staticmethod
        def get_device_properties(device: _device_t = None):
            raise NotImplementedError()

    @staticmethod
    def current_device():
        raise NotImplementedError()

    @staticmethod
    def set_device(device: _device_t):
        raise NotImplementedError()

    @staticmethod
    def device_count():
        raise NotImplementedError()

    @staticmethod
    def is_available() -> bool:
        raise NotImplementedError()

    @staticmethod
    def stream(stream: torch.Stream):
        raise NotImplementedError()

    @staticmethod
    def current_stream():
        raise NotImplementedError()

    @staticmethod
    def set_stream(stream: torch.Stream):
        raise NotImplementedError()

    @staticmethod
    def _set_stream_by_id(stream_id: int, device_index: int, device_type: int):
        raise NotImplementedError()

    @staticmethod
    def get_raw_stream():
        raise NotImplementedError()

    @staticmethod
    def synchronize(device: _device_t = None):
        raise NotImplementedError()

    @staticmethod
    def get_device_properties(device: _device_t = None):
        raise NotImplementedError()

    @staticmethod
    def get_compute_capability(device: _device_t = None):
        raise NotImplementedError()


class HpuInterface(DeviceInterface):
    from torch.hpu import device

    device = torch.hpu.device

    from habana_frameworks.torch.hpu.events import Event
    from habana_frameworks.torch.hpu.streams import Stream

    # register Event and Stream class into the backend interface
    # make sure Event and Stream are implemented and inherited from the _EventBase and _StreamBase
    Event = torch.hpu.Event
    Stream = torch.hpu.Stream

    class Worker:
        @staticmethod
        def set_device(device: int):
            caching_worker_current_devices["hpu"] = device

        @staticmethod
        def current_device() -> int:
            if not htorch.hpu.is_initialized():
                return
            if "hpu" in caching_worker_current_devices:
                return caching_worker_current_devices["hpu"]
            return torch.hpu.current_device()

        @staticmethod
        def get_device_properties(device: _device_t = None):
            if not htorch.hpu.is_initialized():
                return

            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                    assert device.type == "hpu"
                if isinstance(device, torch.device):
                    device = device.index
            if device is None:
                device = HpuInterface.Worker.current_device()

            if "hpu" not in caching_worker_device_properties:
                device_prop = [torch.hpu.get_device_properties(i) for i in range(torch.hpu.device_count())]
                caching_worker_device_properties["hpu"] = device_prop

            return caching_worker_device_properties["hpu"][device]

    current_device = staticmethod(torch.hpu.current_device)
    set_device = staticmethod(torch.hpu.set_device)
    device_count = staticmethod(torch.hpu.device_count)
    stream = staticmethod(torch.hpu.stream)  # type: ignore[assignment]
    current_stream = staticmethod(torch.hpu.current_stream)
    set_stream = staticmethod(torch.hpu.set_stream)  # type: ignore[assignment]
    # _set_stream_by_id = staticmethod(torch.hpu._set_stream_by_id)  # type: ignore[assignment]
    synchronize = staticmethod(torch.hpu.synchronize)
    get_device_properties = staticmethod(torch.hpu.get_device_properties)  # type: ignore[assignment]
    get_raw_stream = staticmethod(get_hpu_stream)  # type: ignore[arg-type]

    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        return torch.hpu.is_available()

    @staticmethod
    def get_compute_capability(device: _device_t = None):
        return torch.hpu.get_device_capability(device)
