import math  # for ceil etc
import os
import sys

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as ht
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from habana_frameworks.torch.hpex.kernels import fp8_fused_sdpa
from test_utils import compare_tensors

DBG_FLAG_verbose_print = False
print_max_diff = False


# LNEG = float('-inf')
LNEG = -1e9


def vb_print(*args, **kwargs):
    if DBG_FLAG_verbose_print:
        print(*args, **kwargs)


def check_dbg_env_var(v):
    env_var_set = False
    if int(os.getenv(v, 0)) == 1:
        env_var_set = True
    return env_var_set


# ****************************************************************************************
# ***********************************Test code Follows************************************
# ****************************************************************************************


# reference code from : pytorch/test/test_transformers.py and modified
def create_attention_mask_for_test(batch_size, n_heads, seq_len_N_t, seq_len_N_s, dtype, shape, float_mask=True):
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


def vanilla_attention_impl_for_test(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, is_amax_s=False):

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
        scores.masked_fill_(attn_mask == False, LNEG)

    weight = F.softmax(scores, dim=-1)
    fwd_out = torch.matmul(weight, value)

    if is_amax_s:
        return fwd_out, torch.max(weight).to(torch.float32)
    else:
        return fwd_out, None


tc_list_fp8 = [
    # 4D inference, Non Triangular mask, Non-Fast softmax, no RH slice, amax_s
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
        False,  # rhslice
        True,  # inference
        "None",  # default softmax
        True,  # is_amax_s
    ),
    # 4D inference, Triangular mask, Non-Fast softmax, no RH slice, amax_s
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
        False,  # rhslice
        True,  # inference
        "None",  # default softmax
        True,  # is_amax_s
    ),
    # 4D inference, Non Triangular mask, Fast softmax, no RH slice, amax_s
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
        False,  # rhslice
        True,  # inference
        "fast",  # fast softmax
        True,  # is_amax_s
    ),
    # 4D inference, Non Triangular mask, Non-Fast softmax, no RH slice, no amax_s
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
        False,  # rhslice
        True,  # inference
        "None",  # default softmax
        False,  # is_amax_s
    ),
    # 4D inference, Non Triangular mask, Non-Fast softmax, with RH slice, amax_s
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
        "None",  # default softmax
        True,  # is_amax_s
    ),
    # 4D inference, Triangular mask, Non-Fast softmax, with RH slice, amax_s
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
        "None",  # default softmax
        True,  # is_amax_s
    ),
    # 4D inference, Non Triangular mask, Fast softmax, with RH slice, amax_s
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
        "fast",  # fast softmax
        True,  # is_amax_s
    ),
    # 4D inference, Non Triangular mask, Non-Fast softmax, with RH slice, no amax_s
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
        "None",  # default softmax
        False,  # is_amax_s
    ),
    # 3D inference, Non Triangular mask, Non-Fast softmax, with RH slice, amax_s
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
        "None",  # default softmax
        True,  # is_amax_s
    ),
]

total_tc_list = tc_list_fp8


@pytest.mark.xfail(reason="Temporarily disabled")
@pytest.mark.parametrize(
    "batch_size, n_heads, seq_len_N_t, seq_len_N_s, head_dim_qk, head_dim_v, dropout_p, use_attn_mask, use_float_mask, enable_autocast, is_causal, recompute, rhslice, inference, softmax_mode, is_amax_s",
    total_tc_list,
)
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
    is_amax_s,
):

    torch.manual_seed(1234567)

    dtype = torch.float32
    rtol = 1e-3
    atol = 1e-3

    if enable_autocast:
        dtype = torch.bfloat16
        grad_dtype = torch.bfloat16
        rtol = 1e-3
        atol = 0.08

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

    vb_print("softmax mode = ", softmax_mode)
    vb_print("is_amax_s = ", is_amax_s)

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

    q_t = q.clone().detach()
    k_t = k.clone().detach()
    v_t = v.clone().detach()

    q_hpu = q.to("hpu").detach()
    k_hpu = k.to("hpu").detach()
    v_hpu = v.to("hpu").detach()

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

    # Set the env. var to enable batchsize/Num heads slicing if needed.
    if rhslice:
        os.environ["PT_HPU_SDPA_BATCH_NUMHEADS_SLICE"] = "1"
    else:
        os.environ["PT_HPU_SDPA_BATCH_NUMHEADS_SLICE"] = "0"
    # ----------------------------------HPU Fused SDPA attention---------------------------------------------
    if recompute:
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=enable_autocast):
            # Use ht.sdp_kernel() context manager to enable/disable recompute based on pytest recompute parameter
            with ht.sdp_kernel(enable_recompute=recompute):
                O_hpu, amax_s = fp8_fused_sdpa(
                    q_hpu,
                    k_hpu,
                    v_hpu,
                    attn_mask=attn_mask_hpu,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    softmax_mode=softmax_mode,
                    is_amax_s=is_amax_s,
                )
    else:
        assert False, " F8 is supported only in recompute mode now"
    htcore.mark_step()

    # ------------------------------- Vanilla SDPA implementation on CPU for test----------------------------
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=enable_autocast):
        O_ref, amax_s_ref = vanilla_attention_impl_for_test(
            q_t, k_t, v_t, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, is_amax_s=is_amax_s
        )

    # ------------------------------- Test Results Comparison ----------------------------
    vb_print("\n")
    O_hpu_c = O_hpu.detach().to("cpu")
    compare_tensors(O_ref, O_hpu_c, atol=atol, rtol=rtol)

    if is_amax_s:
        amax_s_hpu_c = amax_s.detach().to("cpu")
        vb_print("cpu amax_s = ", amax_s_ref)
        vb_print("hpu amax_s = ", amax_s_hpu_c)
        compare_tensors(amax_s_ref, amax_s_hpu_c, atol=atol, rtol=rtol)

    vb_print("Vanilla SDPA FWD Ref vs FSDPA match? = ", torch.allclose(O_ref, O_hpu_c, rtol=rtol, atol=atol))
    if is_amax_s:
        vb_print(
            "Vanilla SDPA amax_s Ref vs FSDPA amax_s match? = ",
            torch.allclose(amax_s_ref, amax_s_hpu_c, rtol=rtol, atol=atol),
        )
    vb_print("\n")
    if print_max_diff:
        vb_print("Max diff Vanilla SDPA FWD Ref vs FSDPA = ", torch.max(torch.abs(O_ref - O_hpu_c)))
        if is_amax_s:
            vb_print(
                "Max diff Vanilla SDPA amax_s Ref vs FSDPA amax_s = ", torch.max(torch.abs(amax_s_ref - amax_s_hpu_c))
            )
