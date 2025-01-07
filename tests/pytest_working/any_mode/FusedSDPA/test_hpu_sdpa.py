###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

import csv
import math  # for ceil etc
import os
import sys

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as ht
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from habana_frameworks.torch.hpex.kernels import FusedSDPA
from sdpa_test_utils import check_dbg_env_var, get_dbg_env_var_num, vb_print
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    is_gaudi1,
    is_pytest_mode_compile,
)

DBG_FLAG_use_func_drpout = False
print_max_diff = False

# Large -ve value ; Using -inf can cause issues when softmax soft max is taken over a
# section that has all -inf on the row. This can happen in algs that operate on slices.
# So use a large -ve value other than -inf.

# LNEG = float('-inf')
LNEG = -1e9


def create_dropout_mask(input, shape, p, generator=None):
    assert generator is None
    t = torch.rand(shape, dtype=input.dtype, layout=input.layout, device=input.device)
    mask = (t < p).to(dtype=torch.uint8)
    return mask


def dropout_with_mask(input, p, mask):
    if p == 1.0:
        dropout_scaling = 0.0
    else:
        dropout_scaling = 1.0 / (1.0 - p)
    # RTC: should type_as be done ouside to avoid this being done within the loop in BWD
    # because this may have d2d copy. but then it can increase temp. mem if converted
    # upfront to float.
    res = mask.type_as(input) * input * dropout_scaling
    return res


# Debug Wrapper for dropout incase we want to call nn.functional.dropout
# instead of dropout with given mask
def dropout_wrapper(x, p, mask=None):

    if mask is not None:
        return dropout_with_mask(x, p, mask)
    else:
        return F.dropout(x, p=p)


# ****************************************************************************************
# ***********************************Test code Follows************************************
# ****************************************************************************************


# RTC : creates 4d attn mask. If Q/K/V shape is different, need to create mask in that shape for detailed testing
# reference code from : pytorch/test/test_transformers.py and modified
def create_attention_mask_for_test(
    batch_size, n_heads, seq_len_N_t, seq_len_N_s, dtype, shape, float_mask=True
):  # RTC: cross attention will need to use source and target seq lens
    attn_mask = torch.randint(0, 2, (seq_len_N_s,)).float()
    if float_mask:
        attn_mask = attn_mask.masked_fill(attn_mask == 0, LNEG).masked_fill(attn_mask == 1, float(0.0))
    attn_mask = attn_mask.to(dtype)

    if shape == "Bx1x1xN":
        if n_heads == 0:
            mask_shape = (batch_size, 1, seq_len_N_s)
        else:
            mask_shape = (batch_size, 1, 1, seq_len_N_s)
        attn_mask = attn_mask.expand(mask_shape)
    else:
        if n_heads == 0:
            mask_shape = (batch_size, seq_len_N_t, seq_len_N_s)
        else:
            mask_shape = (batch_size, n_heads, seq_len_N_t, seq_len_N_s)
        attn_mask = attn_mask.expand(mask_shape)
    return attn_mask


def vanilla_attention_impl_for_test(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, dbg_dropout_mask=None, return_attn_probs=False
):

    sqrt_dim_head = query.shape[-1] ** 0.5
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / sqrt_dim_head

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores.masked_fill_(attn_mask == False, -float("inf"))
        else:
            scores = scores + attn_mask
    elif is_causal:
        seq_len_N_t = query.shape[-2]
        seq_len_N_s = key.shape[-2]
        attn_mask = torch.ones(seq_len_N_t, seq_len_N_s, dtype=torch.bool).tril(diagonal=0)
        # scores.masked_fill_(attn_mask == False, -float('inf'))
        scores.masked_fill_(attn_mask == False, LNEG)

    softmax = F.softmax(scores, dim=-1)
    weight = softmax
    if dropout_p > 0.0:
        if dbg_dropout_mask is not None:
            weight = dropout_wrapper(weight, dropout_p, mask=dbg_dropout_mask)
        else:
            mask = None
            if not DBG_FLAG_use_func_drpout:
                mask = create_dropout_mask(weight, weight.shape, dropout_p)

            weight = dropout_wrapper(weight, dropout_p, mask=mask)

    if return_attn_probs:
        return torch.matmul(weight, value), softmax
    else:
        return torch.matmul(weight, value), None


def perf_cmp_fsdpa_vs_vanilla_attn(g_hpu, q_hpu, k_hpu, v_hpu, attn_mask=None, dropout_p=0.0, is_causal=False):
    is_perf_run = True
    if check_dbg_env_var("FSDPA_DBG_PERF_CMP_RUN_FSDPA"):
        vb_print("Perf cmp run: FSDPA")
        os.environ["FSDPA_DBG_USE_DROPOUT_STUB"] = "0"
        O_hpu = FusedSDPA.apply(q_hpu, k_hpu, v_hpu, attn_mask_hpu, dropout_p, is_causal)
    elif check_dbg_env_var("FSDPA_DBG_PERF_CMP_RUN_VANILLA"):
        vb_print("Perf cmp run: Vanilla")
        O_hpu = vanilla_attention_impl_for_test(
            q_hpu, k_hpu, v_hpu, attn_mask_hpu, dropout_p=dropout_p, is_causal=is_causal
        )
    else:
        is_perf_run = False
        return is_perf_run

    # htcore.mark_step() # if FWD and BWD in two graphs
    # O = O_hpu.to("cpu")
    O_hpu.backward(g_hpu)

    htcore.mark_step()
    # O = O_hpu.to("cpu") No need to take out FWD pass o/p
    q_grad = q_hpu.grad.to("cpu")
    k_grad = k_hpu.grad.to("cpu")
    v_grad = v_hpu.grad.to("cpu")
    return is_perf_run


tc_list = [
    # Cross attention with head_dim qk != head_dim v
    (
        2,  # batch_size,
        4,  # n_heads,
        32,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        16,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        16,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        True,  # use_attn_mask,
        True,  # use_float_mask,
        False,  # enable_autocast
        False,  # is_causal
        False,  # recompute
        False,  # rhslice
        False,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
    # Cross attention with head_dim qk != head_dim v, non-inference mode, returning dropout mask to user
    (
        2,  # batch_size,
        4,  # n_heads,
        32,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        16,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        16,  # head_dim_v,  i.e. head_dim of v
        0.1,  # dropout_p,
        True,  # use_attn_mask,
        True,  # use_float_mask,
        False,  # enable_autocast
        False,  # is_causal
        False,  # recompute
        False,  # rhslice
        False,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
    # Cross attention with head_dim qk != head_dim v, inference mode, returning dropout mask and attn_probs to user
    (
        2,  # batch_size,
        4,  # n_heads,
        32,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        16,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        16,  # head_dim_v,  i.e. head_dim of v
        0.1,  # dropout_p,
        True,  # use_attn_mask,
        True,  # use_float_mask,
        False,  # enable_autocast
        False,  # is_causal
        False,  # recompute
        False,  # rhslice
        True,  # inference
        "None",  # softmax_mode
        True,  # return_attn_probs
    ),
    # Cross attention with head_dim qk != head_dim v, inference mode, returning attn_probs to user
    (
        2,  # batch_size,
        4,  # n_heads,
        32,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        16,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        16,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        False,  # use_attn_mask,
        True,  # use_float_mask,
        False,  # enable_autocast
        True,  # is_causal
        False,  # recompute
        False,  # rhslice
        True,  # inference
        "None",  # softmax_mode
        True,  # return_attn_probs
    ),
    # Cross attention with head_dim qk != head_dim v ;enable auto cast, is_causal = True
    (
        2,  # batch_size,
        4,  # n_heads,
        16,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        32,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        16,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        False,  # use_attn_mask,
        True,  # use_float_mask,
        True,  # enable_autocast
        True,  # is_causal
        False,  # recompute
        False,  # rhslice
        False,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
    # Cross attention with head_dim qk != head_dim v without multi head, i.e 3D tensors
    (
        2,  # batch_size,
        0,  # n_heads, special meaning for test code ; n_heads = 0 means no multi head attn ; i.e 3D tensors
        32,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        16,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        16,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        True,  # use_attn_mask,
        True,  # use_float_mask,
        True,  # enable_autocast
        False,  # is_causal
        False,  # recompute
        False,  # rhslice
        False,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
]

tc_list_recompute = [
    # Self attention with head_dim qk == head_dim v, is_causal, recompute
    (
        2,  # batch_size,
        4,  # n_heads,
        32,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        32,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        8,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        False,  # use_attn_mask,
        True,  # use_float_mask,
        False,  # enable_autocast
        True,  # is_causal
        True,  # recompute
        False,  # rhslice
        False,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
]

# batchsize/numheads slice
tc_list_rhslice = [
    # Self attention with head_dim qk == head_dim v, is_causal, recompute, batchsize/numheads slice
    # 4D Training
    # 3D Training
    (
        3,  # batch_size,
        0,  # n_heads,
        16,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        32,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        4,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        False,  # use_attn_mask,
        True,  # use_float_mask,
        True,  # enable_autocast
        True,  # is_causal
        True,  # recompute
        True,  # rhslice
        False,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
    # 4D inference
    (
        3,  # batch_size,
        4,  # n_heads,
        16,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        32,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        8,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        False,  # use_attn_mask,
        True,  # use_float_mask,
        True,  # enable_autocast
        True,  # is_causal
        True,  # recompute
        True,  # rhslice
        True,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
    # 3D inference
    (
        3,  # batch_size,
        0,  # n_heads,
        16,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        32,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        8,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        False,  # use_attn_mask,
        True,  # use_float_mask,
        True,  # enable_autocast
        True,  # is_causal
        True,  # recompute
        True,  # rhslice
        True,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
]

tc_list_rhslice_inf_attn_mask = [
    # 4D inference, float attn_mask
    (
        3,  # batch_size,
        4,  # n_heads,
        16,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        32,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        8,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        True,  # use_attn_mask,
        True,  # use_float_mask,
        True,  # enable_autocast
        False,  # is_causal
        True,  # recompute
        True,  # rhslice
        True,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
    # 3D inference, float attn_mask
    (
        3,  # batch_size,
        0,  # n_heads,
        16,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        32,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        8,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        True,  # use_attn_mask,
        True,  # use_float_mask,
        True,  # enable_autocast
        False,  # is_causal
        True,  # recompute
        True,  # rhslice
        True,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
    # 4D inference, bool attn_mask
    (
        3,  # batch_size,
        4,  # n_heads,
        16,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        32,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        8,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        True,  # use_attn_mask,
        False,  # use_float_mask,
        True,  # enable_autocast
        False,  # is_causal
        True,  # recompute
        True,  # rhslice
        True,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
    # 3D inference, bool attn_mask
    (
        3,  # batch_size,
        0,  # n_heads,
        16,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        32,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        8,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        True,  # use_attn_mask,
        False,  # use_float_mask,
        True,  # enable_autocast
        False,  # is_causal
        True,  # recompute
        True,  # rhslice
        True,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
]
# For now disable additional tests
# total_tc_list = tc_list + tc_list_recompute + tc_list_rhslice + tc_list_rhslice_inf_attn_mask

# total_tc_list = tc_list

# batchsize/numheads slice
tc_list_new_rules = [
    # Self attention with head_dim qk == head_dim v, is_causal, recompute, batchsize/numheads slice
    # 4D Training
    (
        7,  # batch_size,
        3,  # n_heads,
        16,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        32,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        4,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        False,  # use_attn_mask,
        True,  # use_float_mask,
        True,  # enable_autocast
        True,  # is_causal
        True,  # recompute
        True,  # rhslice
        False,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
    # 4D Inference
    (
        7,  # batch_size,
        3,  # n_heads,
        16,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        32,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        4,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        False,  # use_attn_mask,
        True,  # use_float_mask,
        True,  # enable_autocast
        True,  # is_causal
        True,  # recompute
        True,  # rhslice
        True,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
    # 4D Inference dropout
    (
        7,  # batch_size,
        3,  # n_heads,
        16,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        32,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        4,  # head_dim_v,  i.e. head_dim of v
        0.1,  # dropout_p,
        False,  # use_attn_mask,
        True,  # use_float_mask,
        True,  # enable_autocast
        True,  # is_causal
        True,  # recompute
        True,  # rhslice
        True,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
    # 4D Training dropout
    (
        7,  # batch_size,
        3,  # n_heads,
        16,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        32,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        4,  # head_dim_v,  i.e. head_dim of v
        0.1,  # dropout_p,
        False,  # use_attn_mask,
        True,  # use_float_mask,
        True,  # enable_autocast
        True,  # is_causal
        True,  # recompute
        True,  # rhslice
        False,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
    # 3D Inference
    (
        21,  # batch_size,
        0,  # n_heads,
        16,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        32,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        4,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        False,  # use_attn_mask,
        True,  # use_float_mask,
        True,  # enable_autocast
        True,  # is_causal
        True,  # recompute
        True,  # rhslice
        True,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
    # 3D Training
    (
        21,  # batch_size,
        0,  # n_heads,
        16,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        32,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        4,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        False,  # use_attn_mask,
        True,  # use_float_mask,
        True,  # enable_autocast
        True,  # is_causal
        True,  # recompute
        True,  # rhslice
        False,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
    # 4D Inference bool attnmask
    (
        7,  # batch_size,
        3,  # n_heads,
        16,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        32,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        4,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        True,  # use_attn_mask,
        False,  # use_float_mask,
        True,  # enable_autocast
        False,  # is_causal
        True,  # recompute
        True,  # rhslice
        True,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
    # 4D Inference attnmask
    (
        7,  # batch_size,
        3,  # n_heads,
        16,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        32,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        4,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        True,  # use_attn_mask,
        True,  # use_float_mask,
        True,  # enable_autocast
        False,  # is_causal
        True,  # recompute
        True,  # rhslice
        True,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
    # 4D Training bool attnmask
    (
        7,  # batch_size,
        3,  # n_heads,
        16,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        32,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        4,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        True,  # use_attn_mask,
        False,  # use_float_mask,
        True,  # enable_autocast
        False,  # is_causal
        True,  # recompute
        True,  # rhslice
        False,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
    # 4D Training attnmask
    (
        7,  # batch_size,
        3,  # n_heads,
        16,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        32,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        4,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        True,  # use_attn_mask,
        True,  # use_float_mask,
        True,  # enable_autocast
        False,  # is_causal
        True,  # recompute
        True,  # rhslice
        False,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
]

test_llama_set = [
    (
        2,  # batch_size,
        32,  # n_heads,
        4096,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        4096,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        128,  # head_dim_qk, i.e. head_dim of q and k
        128,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        False,  # use_attn_mask,
        True,  # use_float_mask,
        True,  # enable_autocast
        True,  # is_causal
        False,  # recompute
        True,  # rhslice
        False,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
    (
        4,  # batch_size,
        32,  # n_heads,
        4096,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        4096,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        128,  # head_dim_qk, i.e. head_dim of q and k
        128,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        False,  # use_attn_mask,
        True,  # use_float_mask,
        True,  # enable_autocast
        True,  # is_causal
        False,  # recompute
        True,  # rhslice
        False,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
    (
        4,  # batch_size,
        32,  # n_heads,
        4096,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        4096,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        128,  # head_dim_qk, i.e. head_dim of q and k
        128,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        False,  # use_attn_mask,
        True,  # use_float_mask,
        True,  # enable_autocast
        True,  # is_causal
        True,  # recompute
        True,  # rhslice
        False,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
    (
        2,  # batch_size,
        32,  # n_heads,
        4096,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        4096,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        128,  # head_dim_qk, i.e. head_dim of q and k
        128,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        False,  # use_attn_mask,
        True,  # use_float_mask,
        True,  # enable_autocast
        True,  # is_causal
        True,  # recompute
        True,  # rhslice
        False,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
    (
        1,  # batch_size,
        32,  # n_heads,
        4096,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        4096,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        128,  # head_dim_qk, i.e. head_dim of q and k
        128,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        False,  # use_attn_mask,
        True,  # use_float_mask,
        True,  # enable_autocast
        True,  # is_causal
        False,  # recompute
        True,  # rhslice
        False,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
    (
        4,  # batch_size,
        12,  # n_heads,
        4096,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        4096,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        128,  # head_dim_qk, i.e. head_dim of q and k
        128,  # head_dim_v,  i.e. head_dim of v
        0.1,  # dropout_p,
        False,  # use_attn_mask,
        True,  # use_float_mask,
        True,  # enable_autocast
        True,  # is_causal
        True,  # recompute
        True,  # rhslice
        False,  # inference
        "None",  # softmax_mode
        False,  # return_attn_probs
    ),
]

fast_list = [
    (  # triangular training
        3,  # batch_size,
        5,  # n_heads,
        16,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
        32,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
        8,  # head_dim_qk, i.e. head_dim of q and k
        4,  # head_dim_v,  i.e. head_dim of v
        0.0,  # dropout_p,
        False,  # use_attn_mask,
        True,  # use_float_mask,
        True,  # enable_autocast
        True,  # is_causal
        True,  # recompute
        True,  # rhslice
        False,  # inference
        "fast",  # softmax_mode
        False,  # return_attn_probs
    ),
    #    (  #Non-triangular training
    #        3,  # batch_size,
    #        5,  # n_heads,
    #        16,  # seq_len_N_t, i.e. Target seq len (i.e, of q)
    #        32,  # seq_len_N_s, i.e. Source seq len (i.e, of k and v)
    #        8,  # head_dim_qk, i.e. head_dim of q and k
    #        4,  # head_dim_v,  i.e. head_dim of v
    #        0.0,  # dropout_p,
    #        True,  # use_attn_mask,
    #        True,  # use_float_mask,
    #        True,  # enable_autocast
    #        False,  # is_causal
    #        True,  # recompute
    #        True,  # rhslice
    #        False,  # inference
    #        "fast", # softmax_mode
    #    ),
]
# total_tc_list = test_llama_set[-1:]
# total_tc_list = tc_list_new_rules + tc_list_new_rules_non_recomp
# total_tc_list = tc_list_new_rules[0:1]
# For now disable additional tests
total_tc_list = tc_list + tc_list_recompute + tc_list_rhslice + tc_list_rhslice_inf_attn_mask + fast_list
# total_tc_list = fast_list
current_dir = os.path.dirname(__file__)
csv_file_path = os.path.join(current_dir, "sdpa_config.csv")
with open(csv_file_path, "r") as config_obj:
    config_reader = csv.reader(config_obj)
    # total_tc_list = list(config_reader)


def is_param_combo_valid(
    batch_size,
    n_heads,
    seq_len_N_t,
    seq_len_N_s,
    head_dim_qk,
    head_dim_v,
    dropout_p,
    use_attn_mask,
    use_float_mask,
    enable_autocast,
    is_causal,
    recompute,
    rhslice,
    inference,
    softmax_mode,
    return_attn_probs,
):
    if is_causal:
        if use_attn_mask:
            return False

    if n_heads == 0 and rhslice:  # 3D batch-heads slicing case
        # in 3D case, 3D tensors are expanded to 4D by adding a 1
        # at batch dim. So in slicing case, if BATCH FACTOR is provided,
        # it should should be 1.
        batch_slice_fac = get_dbg_env_var_num("PT_HPU_SDPA_BATCH_FACTOR")
        if batch_slice_fac != 0 and batch_slice_fac != 1:
            return False

    if not inference:
        # In training, fast softmax is supported only in Triangular mask case
        if softmax_mode == "fast" and is_causal == False:
            return False
        # return_attn_probs supported only for inference
        if return_attn_probs:
            return False

    return True


# DONOT remove next line:Disable black formatting for easier parameter update
# fmt: off

#@pytest.mark.xfail(reason="Temporarily disabled")
"""
@pytest.mark.parametrize(
    "batch_size",
    (
        5,
    ),
    ids=lambda batch_size: f"batch_size-{batch_size}"
    )
@pytest.mark.parametrize(
    "n_heads",
    (
        4,
    ),
    ids=lambda n_heads: f"n_heads-{n_heads}"
 )
@pytest.mark.parametrize(
    "seq_len_N_t",
    (
        16,
    ),
    ids=lambda seq_len_N_t: f"seq_len_N_t-{seq_len_N_t}"
    )
@pytest.mark.parametrize(
    "seq_len_N_s",
    (
        32,
    ),
    ids=lambda seq_len_N_s: f"seq_len_N_s-{seq_len_N_s}"
    )


@pytest.mark.parametrize(
    "head_dim_qk",
    (
        8,
    ),
    ids=lambda head_dim_qk: f"head_dim_qk-{head_dim_qk}"
    )
@pytest.mark.parametrize(
    "head_dim_v",
    (
        8,
    ),
    ids=lambda head_dim_v: f"head_dim_v-{head_dim_v}"
    )

@pytest.mark.parametrize(
    "dropout_p",
    (
        0.0,
        0.1,
    ),
    ids=lambda dropout_p: f"dropout_p-{dropout_p}"
    )

@pytest.mark.parametrize(
    "use_attn_mask",
    (
        True,
        False,
    ),
    ids=lambda use_attn_mask: f"use_attn_mask-{use_attn_mask}"
    )
@pytest.mark.parametrize(
    "use_float_mask",
    (
        True,
        False,  # enable for detailed test
    ),
    ids=lambda use_float_mask: f"use_float_mask-{use_float_mask}"
    )
@pytest.mark.parametrize(
    "enable_autocast",
    (
        True,
        False,
    ),
    ids=lambda enable_autocast: f"enable_autocast-{enable_autocast}"
    )

@pytest.mark.parametrize(
    "is_causal",
    (
        True,
        False,
    ),
    ids=lambda is_causal: f"is_causal-{is_causal}"
    )

@pytest.mark.parametrize(
    "recompute",
    (
        True,
        False,
    ),
    ids=lambda recompute: f"recompute-{recompute}"
    )
@pytest.mark.parametrize(
    "rhslice",
    (
        True,
        False,
    ),
    ids=lambda rhslice: f"rhslice-{rhslice}"
    )

@pytest.mark.parametrize(
    "inference",
    (
        True,
        False,
    ),
    ids=lambda inference: f"inference-{inference}"
    )
@pytest.mark.parametrize(
    "softmax_mode",
    (
        "None",
        "fast",
    ),
    ids=lambda softmax_mode: f"softmax_mode-{softmax_mode}"
    )
@pytest.mark.parametrize(
    "return_attn_probs",
    (
        "True",
        "False",
    ),
    ids=lambda return_attn_probs: f"return_attn_probs-{return_attn_probs}"
    )
"""
# DONOT remove following line: re-enable black formatting
# fmt: on


# @pytest.mark.xfail(reason="Results mismatch")
@pytest.mark.parametrize(
    "batch_size, n_heads, seq_len_N_t, seq_len_N_s, head_dim_qk, head_dim_v, dropout_p, use_attn_mask, use_float_mask, enable_autocast, is_causal, recompute, rhslice, inference, softmax_mode, return_attn_probs",
    total_tc_list,
)


# @pytest.mark.skip(reason="Too many tests; So Temporarily disabled")
def test_sdpa(
    batch_size,
    n_heads,
    seq_len_N_t,
    seq_len_N_s,
    head_dim_qk,
    head_dim_v,
    dropout_p,
    use_attn_mask,
    use_float_mask,
    enable_autocast,
    is_causal,
    recompute,
    rhslice,
    inference,
    softmax_mode,
    return_attn_probs,
):
    config_name = (
        "BatchSize = "
        + str(batch_size)
        + " num_heads = "
        + str(n_heads)
        + " Nt = "
        + str(seq_len_N_t)
        + " Ns = "
        + str(seq_len_N_s)
        + " head_dim_qk = "
        + str(head_dim_qk)
        + " head_dim_v = "
        + str(head_dim_v)
        + " dropout_p = "
        + str(dropout_p)
        + "use_attn_mask = "
        + str(use_attn_mask)
        + " use_float_mask = = "
        + str(use_float_mask)
        + " enable_autocast = "
        + str(enable_autocast)
        + " is_causal = "
        + str(is_causal)
        + " recompute = "
        + str(recompute)
        + " rhSlice = "
        + str(rhslice)
        + " infrecence = "
        + str(inference)
        + " softmax_mode = "
        + str(softmax_mode)
        + " return_attn_probs = "
        + str(return_attn_probs)
    )
    print(config_name)
    test_case_valid = is_param_combo_valid(
        batch_size,
        n_heads,
        seq_len_N_t,
        seq_len_N_s,
        head_dim_qk,
        head_dim_v,
        dropout_p,
        use_attn_mask,
        use_float_mask,
        enable_autocast,
        is_causal,
        recompute,
        rhslice,
        inference,
        softmax_mode,
        return_attn_probs,
    )

    if is_gaudi1():
        pytest.skip("SDPA tests not supported on G1")

    # print("test_case_valid = ", test_case_valid)
    if not test_case_valid:
        pytest.skip("This testcase is not valid")

    torch.manual_seed(1234567)
    # batch_size = 8
    # seq_len_N = 128
    # embed_dim = 768
    # n_heads = 12
    """
    batch_size = 2
    seq_len_N = 4
    embed_dim = 128
    #head_dim  = 64
    n_heads = 2
    """
    # dropout_p = 0.1

    # debug flags

    dtype = torch.float32
    grad_dtype = torch.float32
    rtol = 1e-3
    atol = 1e-3

    # use_float_mask = True
    # enable_autocast = False

    if enable_autocast:
        dtype = torch.bfloat16
        grad_dtype = torch.bfloat16
        rtol = 1e-3
        atol = 0.08

    if softmax_mode == "fast":
        atol = 0.13

    attn_mask_shape = "Bx1x1xN"
    if use_float_mask:
        mask_dtype = dtype
    else:
        mask_dtype = torch.bool

    vb_print("\nbatch_size = ", batch_size)
    vb_print("num_heads = ", n_heads)
    vb_print("seq_len_N_s = ", seq_len_N_s)
    vb_print("head dim q k = ", head_dim_qk)
    vb_print("head dim v = ", head_dim_v)

    vb_print("dropout probability = ", dropout_p)
    vb_print("Using float attention mask = ", use_float_mask)

    if n_heads == 0:  # special meaning ; no multi head attn . i.e, use 3d tensors
        q_shape = (batch_size, seq_len_N_t, head_dim_qk)
        k_shape = (batch_size, seq_len_N_s, head_dim_qk)
        v_shape = (batch_size, seq_len_N_s, head_dim_v)
        fwd_out_shape = (batch_size, seq_len_N_t, head_dim_v)
    else:  # Multi head attn with n_heads
        q_shape = (batch_size, n_heads, seq_len_N_t, head_dim_qk)
        k_shape = (batch_size, n_heads, seq_len_N_s, head_dim_qk)
        v_shape = (batch_size, n_heads, seq_len_N_s, head_dim_v)
        fwd_out_shape = (batch_size, n_heads, seq_len_N_t, head_dim_v)

    vb_print("q shape = ", q_shape)
    vb_print("k shape = ", k_shape)
    vb_print("v shape = ", v_shape)
    q = torch.randn(q_shape).to(dtype).detach()
    k = torch.randn(k_shape).to(dtype).detach()
    v = torch.randn(v_shape).to(dtype).detach()
    g = torch.ones(fwd_out_shape).to(grad_dtype)
    if not inference:
        q = q.requires_grad_()
        k = k.requires_grad_()
        v = v.requires_grad_()

    q_t = q.clone().detach()
    k_t = k.clone().detach()
    v_t = v.clone().detach()
    g_t = g.clone()
    if not inference:
        q_t = q_t.requires_grad_()
        k_t = k_t.requires_grad_()
        v_t = v_t.requires_grad_()
    q_hpu = q.to("hpu").detach()
    k_hpu = k.to("hpu").detach()
    v_hpu = v.to("hpu").detach()
    if not inference:
        q_hpu = q_hpu.requires_grad_()
        k_hpu = k_hpu.requires_grad_()
        v_hpu = v_hpu.requires_grad_()

    g_hpu = g.to("hpu")

    if use_attn_mask:
        attn_mask = create_attention_mask_for_test(
            batch_size, n_heads, seq_len_N_t, seq_len_N_s, mask_dtype, attn_mask_shape, float_mask=use_float_mask
        )
        attn_mask_hpu = attn_mask.to("hpu")
    else:
        attn_mask = None
        attn_mask_hpu = None

    if use_attn_mask:
        assert is_causal == False, " use_attn_mask and is_causal can not be True at the same time"

    DBG_ONLY_dropout_mask_g = None
    return_dropout_mask = dropout_p > 0.0 and not recompute

    # Set the env. var to enable batchsize/Num heads slicing if needed.
    if rhslice:
        os.environ["PT_HPU_SDPA_BATCH_NUMHEADS_SLICE"] = "1"
    else:
        os.environ["PT_HPU_SDPA_BATCH_NUMHEADS_SLICE"] = "0"

    # ------------------------------- if Perf Run, run it first and return-----------------------------
    perf_run_count = 1
    profile_step = -1
    profile_api = False

    if check_dbg_env_var("FSDPA_DBG_PERF_CMP_RUN_CYCLES"):
        perf_run_count = 6
        profile_step = 4

    if profile_step != -1:
        try:
            sys.path.append(os.environ["PYTORCH_MODULES_ROOT_PATH"])
            from topologies.tools import SynapseProfilerApi, TraceType
        except ImportError:
            print("Failed to import profiling tools")
            profile_step = -1
            pass

    if profile_step != -1:
        profile_api = SynapseProfilerApi()
        trace_type = TraceType.TraceDevice
        profile_dev_id = 0

    for i in range(perf_run_count):
        if profile_api and i == profile_step:
            profile_api.profiler_start(trace_type, profile_dev_id)
        perf_run = perf_cmp_fsdpa_vs_vanilla_attn(
            g_hpu, q_hpu, k_hpu, v_hpu, attn_mask=attn_mask_hpu, dropout_p=dropout_p
        )
        if profile_api and i == profile_step:
            profile_api.profiler_sync(profile_dev_id)
            profile_api.profiler_stop(trace_type, profile_dev_id)
            profile_api.profiler_get_trace_json(trace_type, profile_dev_id)
    if perf_run:
        exit(0)

    # ----------------------------------HPU Fused SDPA attention---------------------------------------------
    def sdpa_fn(
        q_hpu=None,
        k_hpu=None,
        v_hpu=None,
        attn_mask_hpu=None,
        dropout_p=None,
        is_causal=None,
        scale=None,
        softmax_mode=None,
        recompute=None,
        valid_seq_len=None,
        seq_len_padding_type="left",
        return_dropout_mask=False,
        return_attn_probs=False,
    ):

        return FusedSDPA.apply(
            q_hpu,
            k_hpu,
            v_hpu,
            attn_mask_hpu,
            dropout_p,
            is_causal,
            scale,
            softmax_mode,
            recompute,
            valid_seq_len,
            seq_len_padding_type,
            return_dropout_mask,
            return_attn_probs,
        )

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        sdpa_fn = torch.compile(sdpa_fn, backend="hpu_backend")

    with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=enable_autocast):
        # Use ht.sdp_kernel() context manager to enable/disable recompute based on pytest recompute parameter
        with ht.sdp_kernel(enable_recompute=recompute):
            sdpa_outs = sdpa_fn(
                q_hpu,
                k_hpu,
                v_hpu,
                attn_mask_hpu,
                dropout_p,
                is_causal,
                None,
                softmax_mode,
                None,
                None,
                "left",
                return_dropout_mask,
                return_attn_probs,
            )

    if not return_dropout_mask:
        if not return_attn_probs:
            O_hpu = sdpa_outs
        else:
            O_hpu, P = sdpa_outs
    else:
        if not return_attn_probs:
            O_hpu, DBG_ONLY_dropout_mask_g = sdpa_outs
            DBG_ONLY_dropout_mask_g = DBG_ONLY_dropout_mask_g.to("cpu")
        else:
            O_hpu, P, DBG_ONLY_dropout_mask_g = sdpa_outs
            DBG_ONLY_dropout_mask_g = DBG_ONLY_dropout_mask_g.to("cpu")

    if not inference:
        O_hpu.backward(g_hpu)

    htcore.mark_step()

    # ------------------------------- PT NN SDPA implementation on CPU  for Test ----------------------------
    if dropout_p == 0.0 or dropout_p == 1.0:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=enable_autocast):
            sdp_ref = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
            )
        if not inference:
            sdp_ref.backward(g)
    else:
        vb_print(
            "\ndropout_p is not 0.0 or 1.0; So not running torch.nn.functional.scaled_dot_product_attention for comparison"
        )
        vb_print("Will use vanilla attention implementation for comparison")

    # ------------------------------- Vanilla SDPA implementation on CPU for test----------------------------
    # Can take dropout mask from HPU Fused SDPA atten FWD and use in dropout FWD. In this case
    # Vanilla SDPA and HPU Fused SDPA attention FWD and BWD results are expected to match.
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=enable_autocast):
        O_ref, P_ref = vanilla_attention_impl_for_test(
            q_t,
            k_t,
            v_t,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            dbg_dropout_mask=DBG_ONLY_dropout_mask_g,
            return_attn_probs=return_attn_probs,
        )
    if not inference:
        O_ref.backward(g_t)

    # ------------------------------- Test Results Comparison ----------------------------
    vb_print("\n")
    O_hpu_c = O_hpu.detach().to("cpu")
    if not inference:
        q_grad_hpu_c = q_hpu.grad.detach().to("cpu")
        k_grad_hpu_c = k_hpu.grad.detach().to("cpu")
        v_grad_hpu_c = v_hpu.grad.detach().to("cpu")

    if recompute and (dropout_p != 0.0 or dropout_p != 1.0):
        vb_print("recompute and (dropout_p!=0.0 or dropout_p!=1.0): Can not compare results. Returning")
        return

    compare_tensors(O_ref, O_hpu_c, atol=atol, rtol=rtol)
    if not inference:
        compare_tensors(q_t.grad, q_grad_hpu_c, atol=atol, rtol=rtol)
        compare_tensors(k_t.grad, k_grad_hpu_c, atol=atol, rtol=rtol)
        compare_tensors(v_t.grad, v_grad_hpu_c, atol=atol, rtol=rtol)
    else:
        if return_attn_probs:
            P_c = P.detach().to("cpu")
            compare_tensors(P_ref, P_c, atol=atol, rtol=rtol)

    vb_print("Vanilla SDPA FWD Ref vs FSDPA match? = ", torch.allclose(O_ref, O_hpu_c, rtol=rtol, atol=atol))
    if not inference:
        vb_print(
            "Vanilla SDPA BWD Ref Q grad vs FSDPA match? = ",
            torch.allclose(q_t.grad, q_grad_hpu_c, rtol=rtol, atol=atol),
        )
        vb_print(
            "Vanilla SDPA BWD Ref K grad vs FSDPA match? = ",
            torch.allclose(k_t.grad, k_grad_hpu_c, rtol=rtol, atol=atol),
        )
        vb_print(
            "Vanilla SDPA BWD Ref V grad vs FSDPA match? = ",
            torch.allclose(v_t.grad, v_grad_hpu_c, rtol=rtol, atol=atol),
        )
    vb_print("\n")
    if print_max_diff:
        vb_print("Max diff Vanilla SDPA FWD Ref vs FSDPA ", torch.max(torch.abs(O_ref - O_hpu_c)))
        if not inference:
            vb_print("Max diff Vanilla SDPA BWD Ref Q grad vs FSDPA ", torch.max(torch.abs(q_t.grad - q_grad_hpu_c)))
            vb_print("Max diff Vanilla SDPA BWD Ref K grad vs FSDPA ", torch.max(torch.abs(k_t.grad - k_grad_hpu_c)))
            vb_print("Max diff Vanilla SDPA BWD Ref V grad vs FSDPA ", torch.max(torch.abs(v_t.grad - v_grad_hpu_c)))

    if dropout_p == 0.0 or dropout_p == 1.0:
        vb_print("\n")
        print("\ndropout_p == 0.0 or 1.0 : so, comparing with torch.nn.scaled_dot_product_attention also")
        compare_tensors(sdp_ref, O_hpu_c, atol=atol, rtol=rtol)
        if not inference:
            compare_tensors(q.grad, q_grad_hpu_c, atol=atol, rtol=rtol)
            compare_tensors(k.grad, k_grad_hpu_c, atol=atol, rtol=rtol)
            compare_tensors(v.grad, v_grad_hpu_c, atol=atol, rtol=rtol)

        vb_print(
            "PT NN SDPA FWD Ref vs FSDPA match? = ", torch.allclose(sdp_ref.detach(), O_hpu_c, rtol=rtol, atol=atol)
        )
        if not inference:
            vb_print(
                "PT NN SDPA BWD Ref Q grad vs FSDPA match? = ",
                torch.allclose(q.grad, q_grad_hpu_c, rtol=rtol, atol=atol),
            )
            vb_print(
                "PT NN SDPA BWD Ref K grad vs FSDPA match? = ",
                torch.allclose(k.grad, k_grad_hpu_c, rtol=rtol, atol=atol),
            )
            vb_print(
                "PT NN SDPA BWD Ref V grad vs FSDPA match? = ",
                torch.allclose(v.grad, v_grad_hpu_c, rtol=rtol, atol=atol),
            )

        vb_print("\n")
        if print_max_diff:
            vb_print("Max diff PT NN SDPA FWD Ref vs FSDPA ", torch.max(torch.abs(sdp_ref - O_hpu_c)))
            if not inference:
                vb_print("Max diff PT NN SDPA BWD Ref Q grad vs FSDPA ", torch.max(torch.abs(q.grad - q_grad_hpu_c)))
                vb_print("Max diff PT NN SDPA BWD Ref K grad vs FSDPA ", torch.max(torch.abs(k.grad - k_grad_hpu_c)))
                vb_print("Max diff PT NN SDPA BWD Ref V grad vs FSDPA ", torch.max(torch.abs(v.grad - v_grad_hpu_c)))


@pytest.mark.skip(reason="Failure only in CI.Works fine locally")
def test_sdpa_fwd_manual_seed():

    dtype = torch.float32
    q_shape = k_shape = v_shape = (8, 2, 32, 16)

    Q = torch.randn(q_shape).to(dtype)
    K = torch.randn(k_shape).to(dtype)
    V = torch.randn(v_shape).to(dtype)

    Q_hpu = Q.to("hpu")
    K_hpu = K.to("hpu")
    V_hpu = K.to("hpu")

    dropout = 0.2
    seed = 20000
    # Just need FWD output; No need to o/p the dropout mask
    os.environ["FSDPA_DBG_USE_DROPOUT_STUB"] = "0"
    # case 1
    torch.manual_seed(seed)
    case1_fwd_out = FusedSDPA.apply(Q_hpu, K_hpu, V_hpu, None, dropout, True)

    # case 2 repeat ; set manual seed again
    torch.manual_seed(seed)
    case2_fwd_out = FusedSDPA.apply(Q_hpu, K_hpu, V_hpu, None, dropout, True)

    assert torch.allclose(case1_fwd_out.to("cpu"), case2_fwd_out.to("cpu")), " Error: Outputs are not equal"

    # Enable the following test if SDPA seed test needs more testing
    # torch.manual_seed(seed + 1000)
    # case3_fwd_out= FusedSDPA.apply(Q_hpu, K_hpu, V_hpu, None, dropout )

    # assert not torch.allclose(case1_fwd_out, case3_fwd_out), " Error: Outputs are equal"
