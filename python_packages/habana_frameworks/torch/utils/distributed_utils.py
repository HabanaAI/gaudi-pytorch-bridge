###############################################################################
# Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import os
import warnings

import habana_frameworks.torch.distributed.hccl as hccl


def initialize_distributed_hpu():
    warnings.warn(
        "habana_frameworks.torch.utils.distributed_utils.initialize_distributed_hpu is deprecated. "
        "Please use habana_frameworks.torch.distributed.hccl.initialize_distributed_hpu"
    )
    return hccl.initialize_distributed_hpu()
