import torch
import warnings

from typing import Optional, Union
from ._utils import _get_device_index
_device_t = Union[torch.device, str, int, None]

def is_available() -> bool:
    try:
        from habana_frameworks.torch.utils.library_loader import load_habana_module # type: ignore[import]
        load_habana_module()
    except:
        return False
    import habana_frameworks.torch.core as htcore # type: ignore[import]
    return htcore.is_available()

def get_device_type() -> int:
    if is_available():
        import habana_frameworks.torch.core as htcore # type: ignore[import]
        return htcore.get_device_type()
    else:
        return -1

def device_count() -> int:
    r"""Returns the number of HPUs available."""
    if is_available():
        import habana_frameworks.torch.core as htcore # type: ignore[import]
        return htcore.get_device_count()
    else:
        return 0

def get_device_name(device: Optional[_device_t] = None) -> str:
    r"""Gets the name of a device.

    Args:
        device (torch.device or int, optional): device for which to return the
            name. This function is a no-op if this argument is a negative
            integer. It uses the current device,
            if :attr:`device` is ``None`` (default).

    Returns:
        str: the name of the device
    """

    if not is_available():
        warnings.warn("Device not available")
        return ""

    device = _get_device_index(device)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    import habana_frameworks.torch.core as htcore # type: ignore[import]
    return htcore.get_device_name(device)

