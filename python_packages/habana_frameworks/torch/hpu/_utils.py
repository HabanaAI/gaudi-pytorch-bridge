import os
from typing import Any, Optional

import habana_frameworks.torch.hpu as hpu
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
    device_idx: Optional[int] = None
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
    device_id = os.getenv(HLS_MODULE_ID_VAR, -1)
    if device_id:
        device_index = int(device_id)
    else:
        device_index = -1
    return device_index


def _get_available_modules_from_environ():
    visible_modules_str = os.getenv(HABANA_VISIBLE_MODULES_VAR, default="0,1,2,3,4,5,6,7")
    visible_modules = list(map(lambda x: int(x), visible_modules_str.split(",")))
    if not visible_modules:
        # For handling situation when {HABANA_VISIBLE_MODULES_VAR}
        # is set, but empty
        return [0, 1, 2, 3, 4, 5, 6, 7]
    assert (
        len(visible_modules) > 0 and len(visible_modules) <= 8
    ), f"{HABANA_VISIBLE_MODULES_VAR} does not have valid value."
    return visible_modules
