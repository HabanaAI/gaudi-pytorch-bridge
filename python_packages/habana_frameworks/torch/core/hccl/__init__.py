import warnings
warnings.warn("habana_frameworks.torch.core.hccl is deprecated. "
            "Please use habana_frameworks.torch.distributed.hccl")
from habana_frameworks.torch.distributed._hccl_C import *
