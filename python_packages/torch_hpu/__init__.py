###############################################################################
# Copyright (c) 2021-2024 Intel Corporation
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

import warnings

from torch.types import Device


def is_available() -> bool:
    warnings.warn("torch_hpu.is_available is deprecated. Please use habana_frameworks.torch.hpu.is_available")
    from habana_frameworks.torch import hpu

    return hpu.is_available()


def device_count() -> int:
    warnings.warn("torch_hpu.device_count is deprecated. Please use habana_frameworks.torch.hpu.device_count")
    from habana_frameworks.torch import hpu

    return hpu.device_count()


def get_device_name(device: Device = None) -> str:
    warnings.warn("torch_hpu.get_device_name is deprecated. Please use habana_frameworks.torch.hpu.get_device_name")
    from habana_frameworks.torch import hpu

    return hpu.get_device_name(device)
