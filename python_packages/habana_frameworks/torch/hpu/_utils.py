import torch
from typing import Optional, Any
import os

def _get_device_index(device: Any) -> int:
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a HPU device. Note that for a HPU device without a specified index,
    i.e., ``torch.device('hpu')``, this will return the current default HPU
    device.

    If :attr:`device` is a Python integer, it is returned as is.
    """

    if isinstance(device, str):
        device = torch.device(device)
    device_idx: Optional[int] = None
    if isinstance(device, torch.device):
        if device.type != 'hpu':
            raise ValueError('Expected a hpu device, but got: {}'.format(device))
        device_idx = device.index
    if isinstance(device, int):
        device_idx = device
    if device_idx is None:
        device_idx = 0

    return device_idx

def _get_device_id_from_environ():
    device_id = os.getenv("ID")
    if not device_id:
        device_id = os.getenv("LOCAL_RANK")
    if not device_id:
        device_id = os.getenv("OMPI_COMM_WORLD_LOCAL_RANK")
    if device_id:
        device_index = int(device_id)
    else:
        device_index = -1
    return device_index
