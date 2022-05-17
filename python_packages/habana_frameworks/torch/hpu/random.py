import torch
from typing import Iterable, List, Union
import habana_frameworks.torch.hpu as hpu
from ._utils import _get_device_index
from . import device_count, current_device
from torch._C import default_generator
from torch import Tensor

default_generators: List[torch._C.Generator] = []
for i in range(device_count()):
    default_generators.append(default_generator)

__all__ = ['get_rng_state', 'get_rng_state_all',
           'set_rng_state', 'set_rng_state_all',
           'manual_seed', 'manual_seed_all',
           'seed', 'seed_all', 'initial_seed']

def get_rng_state(device: Union[int, str, torch.device] = 'hpu') -> Tensor:
    device_index = _get_device_index(device)
    return default_generators[device_index].get_state()

def get_rng_state(device: Union[int, str, torch.device] = 'hpu') -> Tensor:
    device_index = _get_device_index(device)
    return default_generators[device_index].get_state()

def set_rng_state(new_state: torch.Tensor, device: Union[int, str, torch.device] = 'hpu') -> None:
    device_index = _get_device_index(device)
    default_generators[device_index].set_state(new_state)

def manual_seed(seed) -> torch._C.Generator:
    device_index = current_device()
    seed = int(seed)
    return default_generators[device_index].manual_seed(seed)

def seed() -> int:
    device_index = current_device()
    return default_generators[device_index].seed()

def initial_seed() -> int:
    device_index = current_device()
    return default_generators[device_index].initial_seed()

def get_rng_state_all() -> List[Tensor]:
    results = []
    for device_index in range(device_count()):
        results.append(get_rng_state(device_index))
    return results

def set_rng_state_all(new_states: Iterable[Tensor]) -> None:
    for device_index, state in enumerate(new_states):
        set_rng_state(state, device_index)

def manual_seed_all(seed: int) -> None:
    seed = int(seed)
    for device_index in range(device_count()):
        default_generators[device_index].manual_seed(seed)

def seed_all() -> None:
    random_seed = 0
    seeded = False
    for device_index in range(device_count()):
        if not seeded:
            default_generators[device_index].seed()
            random_seed = default_generators[device_index].initial_seed()
            seeded = True
        else:
            default_generators[device_index].manual_seed(random_seed)
