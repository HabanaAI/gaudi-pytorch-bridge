###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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

import threading
from collections.abc import Iterable

import habana_frameworks.torch._core_C as htcore

import torch
from torch import Tensor

from ._utils import _get_device_index

# Keeping default_generators as a list with single instance to keep aligned to cuda
default_generators: list[torch._C.Generator] = [htcore._get_default_generator()]


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
    "set_rng_ctx",
    "unset_rng_ctx",
]

parallel_rng_state = None
device_rank_list = []
global_seed = 0
global_offset = 0
use_philox_based_rng = False
cv = (
    threading.Condition()
)  # Condition variable for thread synchronization(https://docs.python.org/3/library/threading.html#condition-objects)


class Offset_LFSR64:
    def __init__(self, n, seed, taps, offset=0):
        self.n = n
        self.seed = 0x3FF6A09E667F3BCD if (seed == 0) else seed
        self.taps = taps

        # Apply offset by advancing the LFSR state
        self._advance_by_offset(offset)

    def _advance_by_offset(self, offset):
        for _ in range(offset):
            self.next_bit()

    def next_bit(self):
        seed = self.seed
        output_bit = self.seed & 1
        for tap in self.taps:
            seed ^= self.seed >> tap

        msb = seed & 1
        self.seed = (self.seed >> 1) | (msb << (self.n - 1))
        return output_bit

    def next_number(self, k=64):
        n = 0
        for _ in range(k):
            n = (n << 1) | self.next_bit()
        return n

    def value(self):
        return format(self.seed, f"0{self.n}b")


def get_rng_state(device: int | str | torch.device = "hpu") -> Tensor:
    global global_seed
    global global_offset
    global parallel_rng_state
    device_index = _get_device_index(device, optional=True)
    is_parallel_rng = False
    if use_philox_based_rng:
        is_parallel_rng = True
    rank_inited = False
    if device_index in device_rank_list:
        rank_inited = True

    if not is_parallel_rng:
        return _default_generator().get_state()
    if not rank_inited:
        global_seed = _default_generator().initial_seed()
        seed_tensor = torch.tensor([global_seed], dtype=torch.uint64).view(torch.uint8).to("hpu")
        offset_tensor = torch.tensor([global_offset], dtype=torch.int64, device="hpu").view(torch.uint8)
        parallel_rng_state = torch.cat([seed_tensor, offset_tensor])
        device_rank_list.append(device_index)

    return parallel_rng_state


def combine_seed_offset(seed_tensor, offset_tensor, gen):
    """
    This function combines the seed and offset tensors and create a new seed
    out of it and set the seed with current generator.
    """
    seed = int.from_bytes(seed_tensor.tolist(), byteorder="little")
    offset = int.from_bytes(offset_tensor.tolist(), byteorder="little")

    # this is to create new seed_tensor based on offset based LFSR on 64 bits
    final_seed = Offset_LFSR64(64, seed, [2, 19, 37, 53], offset).next_number()

    # craete new state with the final_seed value
    gen.manual_seed(final_seed)


def set_rng_state(new_state: torch.Tensor, device: int | str | torch.device = "hpu") -> None:
    global parallel_rng_state

    is_parallel_rng = False
    if use_philox_based_rng:
        is_parallel_rng = True
    if not is_parallel_rng:
        _default_generator().set_state(new_state)
    else:
        parallel_rng_state.copy_(new_state)
        combine_seed_offset(new_state[0:8], new_state[8:16], _default_generator())


def manual_seed(seed) -> torch._C.Generator:
    seed = int(seed)
    return _default_generator().manual_seed(seed)


def seed() -> int:
    return _default_generator().seed()


def initial_seed() -> int:
    return _default_generator().initial_seed()


def get_rng_state_all() -> list[Tensor]:
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


def set_rng_ctx(rng_ctx: str = "philox") -> None:
    """Enable the use of Philox-based RNG context.
    This is useful for operations that require a consistent random number generation
    behavior across multiple devices.
    """
    if rng_ctx != "philox":
        raise ValueError(f"Unsupported RNG context: {rng_ctx}. Supported context is 'philox'.")
    global use_philox_based_rng
    with cv:
        while use_philox_based_rng:
            cv.wait()
        use_philox_based_rng = True


def unset_rng_ctx(rng_ctx: str = "philox") -> None:
    """Disable the use of Philox-based RNG context.
    This will revert to the default RNG behavior, which is based on mersenne twister.
    """
    if rng_ctx != "philox":
        raise ValueError(f"Unsupported RNG context: {rng_ctx}. Supported context is 'philox'.")
    global use_philox_based_rng
    with cv:
        use_philox_based_rng = False
        cv.notify()
