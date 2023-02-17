import torch
from typing import Optional, Any
import os

HABANA_VISIBLE_MODULES_VAR = "HABANA_VISIBLE_MODULES"
HLS_MODULE_ID_VAR = "HLS_MODULE_ID"

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

def _get_module_id_from_environ():
    device_id = os.getenv(HLS_MODULE_ID_VAR, -1)
    if device_id:
        device_index = int(device_id)
    else:
        device_index = -1
    return device_index

def _get_available_modules_from_environ():
    visible_modules_str = os.getenv(HABANA_VISIBLE_MODULES_VAR, default="0,1,2,3,4,5,6,7")
    visible_modules = visible_modules_str.split(",")
    if not visible_modules:
        # For handling situation when {HABANA_VISIBLE_MODULES_VAR}
        # is set, but empty
        return [0,1,2,3,4,5,6,7]
    assert len(visible_modules) > 0 and len(visible_modules) <= 8, \
        f"{HABANA_VISIBLE_MODULES_VAR} does not have valid value."
    return visible_modules
