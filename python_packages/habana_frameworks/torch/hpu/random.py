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

from typing import Iterable, List, Union

import habana_frameworks.torch._core_C as htcore
import torch
from torch import Tensor

from ._utils import _get_device_index

# Keeping default_generators as a list with single instance to keep aligned to cuda
default_generators: List[torch._C.Generator] = [htcore._get_default_generator()]


def _default_generator():
    return default_generators[0]


__all__ = [
    "get_rng_state",
    "get_rng_state_all",
    "set_rng_state",
    "set_rng_state_all",
    "manual_seed",
    "manual_seed_all",
    "seed",
    "seed_all",
    "initial_seed",
]


def get_rng_state(device: Union[int, str, torch.device] = "hpu") -> Tensor:
    device_index = _get_device_index(device, optional=True)
    if device_index != 0:
        raise RuntimeError("hpu get_rng_state supports only device 0")
    return _default_generator().get_state()


def set_rng_state(new_state: torch.Tensor, device: Union[int, str, torch.device] = "hpu") -> None:
    device_index = _get_device_index(device, optional=True)
    if device_index != 0:
        raise RuntimeError("hpu set_rng_state supports only device 0")
    _default_generator().set_state(new_state)


def manual_seed(seed) -> torch._C.Generator:
    seed = int(seed)
    return _default_generator().manual_seed(seed)


def seed() -> int:
    return _default_generator().seed()


def initial_seed() -> int:
    return _default_generator().initial_seed()


def get_rng_state_all() -> List[Tensor]:
    return [get_rng_state(0)]


def set_rng_state_all(new_states: Iterable[Tensor]) -> None:
    if len(new_states) != 1:
        raise RuntimeError("hpu set_rng_state_all supports only states len of 1")
    for state in new_states:
        set_rng_state(state, 0)


def manual_seed_all(seed: int) -> None:
    manual_seed(seed)


def seed_all() -> None:
    _default_generator().seed()
    _default_generator().initial_seed()
