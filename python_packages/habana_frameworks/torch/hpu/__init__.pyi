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
from typing import List, Optional, Union

import torch

_device_t = Union[torch.device, str, int, None]

def init() -> None: ...
def is_available() -> bool: ...
def device_count() -> int: ...
def get_device_type() -> int: ...
def is_initialized() -> bool: ...
def get_device_name(device: Optional[_device_t] = None) -> str: ...
def current_device() -> int: ...
def synchronize() -> None: ...
def is_bf16_available() -> bool: ...
def get_device_capability() -> str: ...
def get_device_properties(device: Optional[_device_t] = None) -> str: ...
def can_device_access_peer(device: _device_t, peer_device: _device_t) -> bool: ...
def get_arch_list() -> List[str]: ...
def get_gencode_flags() -> str: ...

from torch.types import _bool, _device, _dtype, _float, _int

class device:
    type: str  # THPDevice_type
    index: _int  # THPDevice_index

class Stream:
    stream_id: _int  # Stream id
    device_index: _int
    device_type: _int

    device: device  # The device of the stream

# Defined in habana_frameworks/torch/hpu/csrc/Stream.cpp
class _HpuStreamBase(Stream):
    stream_id: _int
    device_index: _int
    device_type: _int

    device: _device
    hpu_stream: _int
    priority: _int

    def __new__(
        self,
        priority: _int = 0,
        stream_id: _int = 0,
        device_index: _int = 0,
        stream_ptr: _int = 0,
    ) -> _HpuStreamBase: ...
    def query(self) -> _bool: ...
    def synchronize(self) -> None: ...

# Defined in habana_frameworks/torch/hpu/csrc/Event.cpp
class _HpuEventBase:
    device: _device
    hpu_event: _int

    def __new__(cls, enable_timing: _bool = False) -> _HpuEventBase: ...
    @classmethod
    def record(self, stream: _HpuStreamBase) -> None: ...
    def wait(self, stream: _HpuStreamBase) -> None: ...
    def query(self) -> _bool: ...
    def elapsed_time(self, other: _HpuEventBase) -> _float: ...
    def synchronize(self) -> None: ...
