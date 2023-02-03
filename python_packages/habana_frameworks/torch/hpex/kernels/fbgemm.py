# ******************************************************************************
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************
import torch
from typing import Optional, Tuple
from habana_frameworks.torch import _hpex_C

# The file implements operators included in the FBGEMM (Facebook GEneral Matrix Multiplication) library.

def permute_1D_sparse_data(permute: torch.Tensor, lengths: torch.Tensor, indices: torch.Tensor, weights: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    list_res = _hpex_C.permute_1D_sparse_data(permute, lengths, indices, weights)
    return tuple(list_res) if len(list_res) == 3 else (list_res[0], list_res[1], None)

def permute_2D_sparse_data(permute: torch.Tensor, lengths: torch.Tensor, indices: torch.Tensor, weights: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    list_res = _hpex_C.permute_2D_sparse_data(permute, lengths, indices, weights)
    return tuple(list_res) if len(list_res) == 3 else (list_res[0], list_res[1], None)
