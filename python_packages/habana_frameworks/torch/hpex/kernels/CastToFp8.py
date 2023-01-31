###############################################################################
# Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import torch
import numpy as np
from habana_frameworks.torch import _hpex_C

def cast_to_fp8(x: torch.tensor, stochastic_rounding = False, seed: int = 0) -> torch.tensor:
    # Error checking
    dtype = x.dtype
    if dtype != torch.bfloat16 and dtype != torch.float32:
        raise TypeError(f"Only float32 and bfloat16 can be casted to fp8 in SR mode, got: {dtype}")
    device = x.device

    if device == torch.device("cpu"):
        if stochastic_rounding:
            raise ValueError(f"Stochastic rounding must be disabled for CPU quantization")
        return x.to(torch.fp8r152)
    else:
        try:
            from habana_frameworks.torch import _hpex_C
            return _hpex_C.cast_to_fp8(x, stochastic_rounding, seed)
        except ImportError:
            raise ImportError("Please install habana_torch.")
