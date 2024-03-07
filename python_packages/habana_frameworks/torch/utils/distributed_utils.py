import os
import warnings

import habana_frameworks.torch.distributed.hccl as hccl


def initialize_distributed_hpu():
    warnings.warn(
        "habana_frameworks.torch.utils.distributed_utils.initialize_distributed_hpu is deprecated. "
        "Please use habana_frameworks.torch.distributed.hccl.initialize_distributed_hpu"
    )
    return hccl.initialize_distributed_hpu()
