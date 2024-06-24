# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.

"""Utilities for C++ extensions"""
from typing import Optional, Union

import torch

from ..utils import FP8BwdTensors, FP8FwdTensors, FP8TensorMeta


def _update_amax_history(
    new_amax: torch.Tensor,
    fp8_meta_tensor: FP8TensorMeta,
    fp8_tensor: Union[FP8FwdTensors, FP8BwdTensors],
):
    if fp8_meta_tensor.amax_history.shape[0] > 1:
        # amax_history length > 1
        # NOTE: This path is functional, but performance could be improved by removing the temporary tensor
        tmp = torch.index_select(fp8_meta_tensor.amax_history, dim=0, index=fp8_meta_tensor.amax_history_index)
        tmp[0][fp8_tensor].copy_(new_amax)
        fp8_meta_tensor.amax_history[fp8_meta_tensor.amax_history_index] = tmp
    else:
        # In case amax_history length = 1, we don't need to use amax_history_index - it simplifies the graph
        fp8_meta_tensor.amax_history[0][fp8_tensor].copy_(new_amax)


def select_amax_and_exec(
    operator,
    fp8_meta_tensor: FP8TensorMeta,
    fp8_tensor: Union[FP8FwdTensors, FP8BwdTensors],
    fp8_meta_tensor2: Optional[FP8TensorMeta] = None,
    measure_amax=True,
):
    outputs = operator()

    if measure_amax:
        _update_amax_history(outputs[-1], fp8_meta_tensor, fp8_tensor)
        if fp8_meta_tensor2 is not None:
            _update_amax_history(outputs[-1], fp8_meta_tensor2, fp8_tensor)

    return outputs[:-1]
