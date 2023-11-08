# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.

"""Utilities for C++ extensions"""
from typing import Union
import torch
from habana_frameworks.torch import _hpex_C as tex


def select_amax_and_exec(
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    operator,
    measure_amax=True
):
    outputs = operator()
    if measure_amax and fp8_meta_tensor.amax_history.shape[0] > 1:
        # amax_history length > 1
        # NOTE: This path is functional, but performance could be improved by removing the temporary tensor
        tmp = torch.index_select(fp8_meta_tensor.amax_history, dim=0, index=fp8_meta_tensor.amax_history_index)
        tmp[0][fp8_tensor].copy_(outputs[-1])
        fp8_meta_tensor.amax_history[fp8_meta_tensor.amax_history_index] = tmp
    elif measure_amax:
        # In case amax_history length = 1, we don't need to use amax_history_index - it simplifies the graph
        fp8_meta_tensor.amax_history[0][fp8_tensor].copy_(outputs[-1])
    return outputs[:-1]
