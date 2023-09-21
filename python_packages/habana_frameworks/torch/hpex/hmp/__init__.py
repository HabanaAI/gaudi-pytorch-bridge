from .hmp import convert
from .utils import disable_casts

import logging

logging.warning("Habana Mixed Precision (HMP) module is deprecated and will be removed in version 1.13.0.\nPlease use native PyTorch autocast for mixed precision training and inference")
