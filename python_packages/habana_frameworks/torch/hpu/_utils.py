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


import os
from typing import Any

import habana_frameworks.torch.utils.experimental as htexp
from habana_frameworks.torch import hpu

import torch

HABANA_VISIBLE_MODULES_VAR = "HABANA_VISIBLE_MODULES"
HLS_MODULE_ID_VAR = "HLS_MODULE_ID"


def _get_device_index(device: Any, optional: bool = False, allow_cpu: bool = False) -> int:
    r"""gets the device index from :attr:`device`, which can be a torch.device
    object, a python integer, or ``none``.

    if :attr:`device` is a torch.device object, returns the device index if it
    is a hpu device. note that for a hpu device without a specified index,
    i.e., ``torch.device('hpu')``, this will return the current default hpu
    device if :attr:`optional` is ``true``. if :attr:`allow_cpu` is ``true``,
    cpu devices will be accepted and ``-1`` will be returned in this case.

    if :attr:`device` is a python integer, it is returned as is.

    if :attr:`device` is ``none``, this will return the current default hpu
    device if :attr:`optional` is ``true``.
    """
    if isinstance(device, int):
        device_idx = device
    if isinstance(device, str):
        device = torch.device(device)
    device_idx: int | None = None
    if isinstance(device, torch.device):
        if allow_cpu:
            if device.type not in ["hpu", "cpu"]:
                raise ValueError(f"Expected a hpu or cpu device, but got: {device}")
        elif device.type != "hpu":
            raise ValueError(f"Expected a hpu device, but got: {device}")
        device_idx = -1 if device.type == "cpu" else device.index
    if isinstance(device, int):
        device_idx = device
    if device_idx is None:
        if optional:
            device_idx = hpu.current_device()
        else:
            raise ValueError(f"Expected a torch.device with a specified index or an integer, but got:{device}")
    return device_idx


def _get_module_id_from_environ():
    device_index = int(os.getenv(HLS_MODULE_ID_VAR, "-1"))
    if not device_index:
        device_index = -1

    return device_index


def _get_available_modules_from_environ():
    visible_modules_str = os.getenv(HABANA_VISIBLE_MODULES_VAR, default="0,1,2,3,4,5,6,7")
    visible_modules = [int(x) for x in visible_modules_str.split(",")]
    if not visible_modules:
        # For handling situation when {HABANA_VISIBLE_MODULES_VAR}
        # is set, but empty
        return [0, 1, 2, 3, 4, 5, 6, 7]
    module_size = len(visible_modules)
    is_gaudi3 = htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi3
    # GAUDI3 support rack-scale project, so visible device length may more than 1 and 2^n modules
    # practically 4 or 16 rack scale up configuration
    if not (module_size > 0 and module_size <= 8 and not (is_gaudi3 and module_size > 1 and (module_size % 2) != 0)):
        raise AssertionError(f"{HABANA_VISIBLE_MODULES_VAR} does not have valid value.")
    return visible_modules
