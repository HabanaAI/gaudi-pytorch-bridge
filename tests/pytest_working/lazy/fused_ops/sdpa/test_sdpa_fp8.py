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
from test_utils import compare_tensors, is_gaudi1, is_gaudi3

DBG_FLAG_verbose_print = False
print_max_diff = False


from habana_frameworks.torch.core.quantization import _check_params_as_const, _mark_params_as_const

# LNEG = float('-inf')
LNEG = -1e9
attention_scale = None
# test_backward = True
test_with_identity = False


def check_dbg_env_var(v):
    env_var_set = False
    if int(os.getenv(v, 0)) == 1:
        env_var_set = True
    return env_var_set


def dump_bwd_api_params(
    do,
    q,
    k,
    v,
    P_hpu,
    dm=None,
    dropout_p=0.0,
    scale=None,
    d_scale_q=None,
    d_scale_k=None,
    d_scale_v=None,
    d_scale_s=None,
    d_scale_do=None,
    d_scale_ds=None,
    q_scale_s=None,
    q_scale_ds=None,
    is_amax_ds=False,
):
    def print_t_info(name, t, is_scale=False):
        if t is not None:
            if is_scale:
                print(name, " : ", t.to("cpu"))
            else:
                print(name, " : ", t.shape)
        else:
            print(name, " is None")

    if not check_dbg_env_var("PT_HPU_DUMP_FUSED_SDPA_API_PARAMS"):
        return
    print("=" * 40, "FUSED_SDPA_BWD_(NONRECOMP)API_PARAMS", "=" * 40)
    print_t_info("do", do)
    print_t_info("q", q)
    print_t_info("k", k)
    print_t_info("v", v)
    print("dropout_p : ", dropout_p)
    print("scale : ", scale)
    print_t_info("d_scale_q", d_scale_q, is_scale=True)
    print_t_info("d_scale_k", d_scale_k, is_scale=True)
    print_t_info("d_scale_v", d_scale_v, is_scale=True)
    print_t_info("d_scale_s", d_scale_s, is_scale=True)
    print_t_info("d_scale_do", d_scale_do, is_scale=True)
    print_t_info("d_scale_ds", d_scale_ds, is_scale=True)
    print_t_info("q_scale_s", q_scale_s, is_scale=True)
    print_t_info("q_scale_ds", q_scale_ds, is_scale=True)
    print("is_amax_ds : ", is_amax_ds)
    print("=" * 90)


def vb_print(*args, **kwargs):
    if DBG_FLAG_verbose_print:
        print(*args, **kwargs)


def process_results(results):
    for r in results:
        if r["compare"]:
            t_cpu = r["t_cpu"]
            t_hpu = r["t_hpu"]
            rtol = r["rtol"]
            atol = r["atol"]
            vb_print(r["name"], "\t: Match? = ", torch.allclose(t_cpu, t_hpu, rtol=rtol, atol=atol))
            if print_max_diff:
                vb_print(r["name"], "\t: Max diff = ", torch.max(torch.abs(t_cpu - t_hpu)))
    for r in results:
        if r["compare"] and r["assert"]:
            t_cpu = r["t_cpu"]
            t_hpu = r["t_hpu"]
            rtol = r["rtol"]
            atol = r["atol"]
            vb_print(r["name"], "\t: Comparing with assert on misatch with rtol = ", rtol, ", atol = ", atol)
            compare_tensors(t_cpu, t_hpu, atol=atol, rtol=rtol)


def is_fp8_run(fp8_run_out_type, inference, is_amax_s, is_amax_o, is_amax_ds):
    if inference:
        if is_amax_s == False:
            return True

        if is_amax_o == True:
            return False
        if is_amax_ds == True:
            return False
        return False
    else:
        return True


def get_scale_values(name, t, is_t_amax=False, scale_limit=None):
    def map_to_g2_hwa_scales(lg2int):
        g2_hwa_scales = [16, 1, 0.0625, 0.00390625]
        if lg2int >= 4:
            return 16
        elif lg2int >= 0:
            return 1
        elif lg2int >= -4:
            return 0.0625
        else:
            return 0.00390625

    FP8_MAX_143 = 240 * 0.9
    if is_t_amax == False:
        maxT = torch.max(torch.abs(t)).to(torch.float).item()
    else:
        maxT = t.item()
    scaleT = FP8_MAX_143 / maxT

    lg2 = math.log2(scaleT)
    lg2_int = int(lg2)

    if not is_gaudi3():
        scaleT_pow2 = map_to_g2_hwa_scales(lg2_int)
    else:
        scaleT_pow2 = 2.0**lg2_int

    scaleTInv = 1.0 / scaleT_pow2
    vb_print(name, ": scale", scaleT)
    vb_print(name, ": scale pow2", scaleT_pow2)
    vb_print(name, ": Inv scale", scaleTInv)

    # scale_limit = 1.0
    if scale_limit != None and is_gaudi3():
        scaleT_pow2 = scale_limit
        scaleTInv = 1.0 / scaleT_pow2
        vb_print(name, ": after limiting : scale pow2", scaleT_pow2)
        vb_print(name, ": after limiting : Inv scale", scaleTInv)

    scaleT_cpu = torch.tensor(scaleT_pow2, dtype=torch.float)
    scaleTInv_cpu = torch.tensor(scaleTInv, dtype=torch.float)
    scaleT_hpu = scaleT_cpu.to("hpu")
    scaleTInv_hpu = scaleTInv_cpu.to("hpu")
    return scaleT_hpu, scaleTInv_hpu


def get_d_scale_s(scaleSInv_hpu, inference, is_fwd=True):
    if inference:
        return scaleSInv_hpu
    else:
        if is_fwd:
            return None
        else:
            return scaleSInv_hpu


class TestModel(torch.nn.Module):
    def __init__(
        self,
        d_scale_q=None,
        d_scale_k=None,
        d_scale_v=None,
        q_scale_s=None,
        q_scale_o=None,
        d_scale_s=None,
        is_amax_s=False,
        is_amax_o=False,
        inference=False,
    ):
        super(TestModel, self).__init__()
        if inference:  # make scales/descales nn Parameter so that they can be made constants later
            self.d_scale_q = torch.nn.Parameter(d_scale_q) if d_scale_q is not None else None
            self.d_scale_k = torch.nn.Parameter(d_scale_k) if d_scale_k is not None else None
            self.d_scale_v = torch.nn.Parameter(d_scale_v) if d_scale_v is not None else None
            self.q_scale_s = torch.nn.Parameter(q_scale_s) if q_scale_s is not None else None
            self.q_scale_o = torch.nn.Parameter(q_scale_o) if q_scale_o is not None else None
            self.d_scale_s = torch.nn.Parameter(d_scale_s) if d_scale_s is not None else None
        else:  # Training case; No need to make scales/descales nn Parameter
            self.d_scale_q = d_scale_q
            self.d_scale_k = d_scale_k
            self.d_scale_v = d_scale_v
            self.q_scale_s = q_scale_s
            self.q_scale_o = q_scale_o
            self.d_scale_s = d_scale_s
        self.is_amax_s = is_amax_s
        self.is_amax_o = is_amax_o

    def forward(self, q_hpu, k_hpu, v_hpu, attn_mask=None, dropout_p=0.0, is_causal=False, softmax_mode="None"):

        O_hpu, amax_s, amax_o = fp8_fused_sdpa(
            q_hpu,
            k_hpu,
            v_hpu,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            softmax_mode=softmax_mode,
            scale=attention_scale,
            d_scale_q=self.d_scale_q,
            d_scale_k=self.d_scale_k,
            d_scale_v=self.d_scale_v,
            q_scale_s=self.q_scale_s,
            q_scale_o=self.q_scale_o,
            d_scale_s=self.d_scale_s,
            is_amax_s=self.is_amax_s,
            is_amax_o=self.is_amax_o,
        )
        return O_hpu, amax_s, amax_o


# ****************************************************************************************
# ***********************************Test code Follows************************************
# ****************************************************************************************


# reference code from : pytorch/test/test_transformers.py and modified
def create_attention_mask_for_test(batch_size, q_heads, seq_len_N_t, seq_len_N_s, dtype, shape, float_mask=True):
    attn_mask = torch.randint(0, 2, (seq_len_N_s,)).float()
    if float_mask:
        attn_mask = attn_mask.masked_fill(attn_mask == 0, LNEG).masked_fill(attn_mask == 1, float(0.0))
    attn_mask = attn_mask.to(dtype)

    if shape == "Bx1x1xN":
        if q_heads == 0:
            mask_shape = (batch_size, 1, seq_len_N_s)
        else:
            mask_shape = (batch_size, 1, 1, seq_len_N_s)
        attn_mask = attn_mask.expand(mask_shape)
    else:
        if q_heads == 0:
            mask_shape = (batch_size, seq_len_N_t, seq_len_N_s)
        else:
            mask_shape = (batch_size, q_heads, seq_len_N_t, seq_len_N_s)
        attn_mask = attn_mask.expand(mask_shape)
    return attn_mask


def vanilla_attention_impl_for_test(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, is_amax_s=False
):

    sqrt_dim_head = query.shape[-1] ** 0.5
    scores = torch.matmul(query, key.transpose(-2, -1))
    if scale == None:
        scores = scores / sqrt_dim_head
    else:
        scores = scores * scale

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
        return fwd_out, weight, torch.max(torch.abs(weight)).to(torch.float32)
    else:
        return fwd_out, weight, None


def vanilla_attention_impl_bwd_for_test(grad, query, key, value, P, scale=None, is_amax_ds=False):
    if scale == None:
        scale = 1.0 / (query.shape[-1] ** 0.5)

    dV = torch.matmul(P.transpose(-2, -1), grad)
    dPdrp = torch.matmul(grad, value.transpose(-2, -1))
    dP = dPdrp  # when dropout is 0.0, dP = dPdrp
    dS = torch._softmax_backward_data(dP, P, -1, P.dtype)

    dK_tmp = torch.matmul(dS.transpose(-2, -1), query)
    dK = torch.mul(dK_tmp, scale)

    dQ_tmp = torch.matmul(dS, key)
    dQ = torch.mul(dQ_tmp, scale)

    amax_ds = None
    if is_amax_ds:
        amax_ds = torch.max(torch.abs(dS)).to(torch.float32)

    return dQ, dK, dV, amax_ds


def gaudi_llama_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Copied from repeat_kv: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    The only differences are:
        - Append num_key_value_heads == 1 check as kv states can be broadcasted during matmuls so need to expand and reshape them.
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1 or num_key_value_heads == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def is_gqa(q, k):
    gqa = False
    dims = q.dim()
    if dims == 4:
        q_heads = q.shape[1]
        kv_heads = k.shape[1]
        gqa = (q_heads != kv_heads) and kv_heads != 1
    vb_print("IS GQA? : ", gqa)
    return gqa


def is_mqa(q, k):
    mqa = False
    dims = q.dim()
    if dims == 4:
        q_heads = q.shape[1]
        kv_heads = k.shape[1]
        mqa = (q_heads != kv_heads) and kv_heads == 1
    vb_print("IS MQA? : ", mqa)
    return mqa


def is_param_combo_valid(
    batch_size,
    q_heads,
    kv_heads,
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
    is_amax_o,
    is_amax_ds,
    fp8_run_out_type,
):

    # BWD is not supported in recompute. So only test inference.
    if recompute:
        if not inference:
            return False
    else:
        # In non-recomp inference case, there is an acc diff in non triangular mask case.
        # To be checked if it is an actual issue.
        if inference:
            if is_causal == False:
                return False

    if not inference:  # limiting tests Temporarily for training
        if q_heads != kv_heads:
            return False  # no MQA/GQA test for now

    # fp8 mode supports only inference in Triangular and Non-Triangular mask mode
    # But training is supported only in Triangular mask mode as of now.
    if not inference:
        if is_causal == False:
            return False

    # fp8 mode supports only recompute mode as of now.
    # if not recompute:
    #    return False

    if is_causal:
        if use_attn_mask:
            return False

    if dropout_p != 0.0:
        return False

    # Fp8 measurement or  supported only if tensors are bf16 before convert to fp8
    if enable_autocast == False:
        return False

    is_amax = is_amax_s or is_amax_o

    fp8_run = is_fp8_run(fp8_run_out_type, inference, is_amax_s, is_amax_o, is_amax_ds)
    # if fp8_run: return False

    if inference:
        # inference does not have amax_o
        if is_amax_o:
            return False
        # inference does not have amax_ds
        if is_amax_ds:
            return False

        if is_amax_s:
            if fp8_run == True:
                return False

        if fp8_run:
            if is_amax_s == True:
                return False
            # fp8 run in inference has fast softmax internally.
            # So do not set from test.
            # TODO: See if we should accept this config and ignore.
            if softmax_mode != "None":
                return False
    else:
        # TODO: modify this later
        # Currently support only fast softmax in training measurement/run
        if softmax_mode != "fast":
            return False

    return True


tc_list1 = [
    (  # inf measure, recompute, triangular
        3,  # batch_size
        4,  # q_heads
        4,  # kv_heads,
        16,  # seq_len_N_t
        32,  # seq_len_N_s
        8,  # head_dim_qk
        8,  # head_dim_v
        0.0,  # dropout_p
        False,  # use_attn_mask
        True,  # use_float_mask
        True,  # enable_autocast
        True,  # is_causal
        True,  # recompute
        True,  # rhslice
        True,  # inference
        "fast",  # "None",  # softmax_mode
        True,  # is_amax_s
        False,  # is_amax_o
        False,  # is_amax_ds
        "fp8_143",  # "bf16",  # fp8_run_out_type
    ),
]
tc_list2 = [
    (  # inf run, recompute, triangular
        3,  # batch_size
        4,  # q_heads
        4,  # kv_heads,
        16,  # seq_len_N_t
        32,  # seq_len_N_s
        8,  # head_dim_qk
        8,  # head_dim_v
        0.0,  # dropout_p
        False,  # use_attn_mask
        True,  # use_float_mask
        True,  # enable_autocast
        True,  # is_causal
        True,  # recompute
        True,  # rhslice
        True,  # inference
        "None",  # softmax_mode
        False,  # is_amax_s
        False,  # is_amax_o
        False,  # is_amax_ds
        "fp8_143",  # "bf16",  # fp8_run_out_type
    ),
]

tc_list3 = [
    (  # inf measure, recompute, Non-triangular
        3,  # batch_size
        4,  # q_heads
        4,  # kv_heads,
        16,  # seq_len_N_t
        32,  # seq_len_N_s
        8,  # head_dim_qk
        8,  # head_dim_v
        0.0,  # dropout_p
        True,  # use_attn_mask
        True,  # use_float_mask
        True,  # enable_autocast
        False,  # is_causal
        True,  # recompute
        True,  # rhslice
        True,  # inference
        "fast",  # "None",  # softmax_mode
        True,  # is_amax_s
        False,  # is_amax_o
        False,  # is_amax_ds
        "fp8_143",  # "bf16",  # fp8_run_out_type
    ),
]
tc_list4 = [
    (  # inf run, recompute, Non-triangular
        3,  # batch_size
        4,  # q_heads
        4,  # kv_heads,
        16,  # seq_len_N_t
        32,  # seq_len_N_s
        8,  # head_dim_qk
        8,  # head_dim_v
        0.0,  # dropout_p
        True,  # use_attn_mask
        True,  # use_float_mask
        True,  # enable_autocast
        False,  # is_causal
        True,  # recompute
        True,  # rhslice
        True,  # inference
        "None",  # softmax_mode
        False,  # is_amax_s
        False,  # is_amax_o
        False,  # is_amax_ds
        "fp8_143",  # "bf16",  # fp8_run_out_type
    ),
]

tc_list5 = [
    (  # inf measure, Non-recompute, triangular
        3,  # batch_size
        4,  # q_heads
        4,  # kv_heads,
        16,  # seq_len_N_t
        32,  # seq_len_N_s
        8,  # head_dim_qk
        8,  # head_dim_v
        0.0,  # dropout_p
        False,  # use_attn_mask
        True,  # use_float_mask
        True,  # enable_autocast
        True,  # is_causal
        False,  # recompute
        True,  # rhslice
        True,  # inference
        "fast",  # "None",  # softmax_mode
        True,  # is_amax_s
        False,  # is_amax_o
        False,  # is_amax_ds
        "fp8_143",  # "bf16",  # fp8_run_out_type
    ),
]
tc_list6 = [
    (  # inf run, Non-recompute, triangular
        3,  # batch_size
        4,  # q_heads
        4,  # kv_heads,
        16,  # seq_len_N_t
        32,  # seq_len_N_s
        8,  # head_dim_qk
        8,  # head_dim_v
        0.0,  # dropout_p
        False,  # use_attn_mask
        True,  # use_float_mask
        True,  # enable_autocast
        True,  # is_causal
        False,  # recompute
        True,  # rhslice
        True,  # inference
        "None",  # softmax_mode
        False,  # is_amax_s
        False,  # is_amax_o
        False,  # is_amax_ds
        "fp8_143",  # "bf16",  # fp8_run_out_type
    ),
]

tc_list7 = [
    (  # inf measure, Non-recompute, Non-triangular
        3,  # batch_size
        4,  # q_heads
        4,  # kv_heads,
        16,  # seq_len_N_t
        32,  # seq_len_N_s
        8,  # head_dim_qk
        8,  # head_dim_v
        0.0,  # dropout_p
        True,  # use_attn_mask
        True,  # use_float_mask
        True,  # enable_autocast
        False,  # is_causal
        False,  # recompute
        True,  # rhslice
        True,  # inference
        "fast",  # "None",  # softmax_mode
        True,  # is_amax_s
        False,  # is_amax_o
        False,  # is_amax_ds
        "fp8_143",  # "bf16",  # fp8_run_out_type
    ),
]
tc_list8 = [
    (  # inf run, Non-recompute, Non-triangular
        3,  # batch_size
        4,  # q_heads
        4,  # kv_heads,
        16,  # seq_len_N_t
        32,  # seq_len_N_s
        8,  # head_dim_qk
        8,  # head_dim_v
        0.0,  # dropout_p
        True,  # use_attn_mask
        True,  # use_float_mask
        True,  # enable_autocast
        False,  # is_causal
        False,  # recompute
        True,  # rhslice
        True,  # inference
        "None",  # softmax_mode
        False,  # is_amax_s
        False,  # is_amax_o
        False,  # is_amax_ds
        "fp8_143",  # "bf16",  # fp8_run_out_type
    ),
]
tc_list9 = [
    (  # train run
        3,  # batch_size
        4,  # q_heads
        4,  # kv_heads,
        16,  # seq_len_N_t
        32,  # seq_len_N_s
        8,  # head_dim_qk
        8,  # head_dim_v
        0.0,  # dropout_p
        False,  # use_attn_mask
        True,  # use_float_mask
        True,  # enable_autocast
        True,  # is_causal
        False,  # recompute
        True,  # rhslice
        False,  # inference
        "fast",  # "None",  # softmax_mode
        False,  # is_amax_s
        False,  # is_amax_o
        False,  # is_amax_ds
        "bf16",  # "fp8_143", #"bf16",  # fp8_run_out_type
    ),
]
tc_list10 = [
    (  # Train Meas
        3,  # batch_size
        4,  # q_heads
        4,  # kv_heads,
        16,  # seq_len_N_t
        32,  # seq_len_N_s
        8,  # head_dim_qk
        8,  # head_dim_v
        0.0,  # dropout_p
        False,  # use_attn_mask
        True,  # use_float_mask
        True,  # enable_autocast
        True,  # is_causal
        False,  # recompute
        True,  # rhslice
        False,  # inference
        "fast",  # "None",  # softmax_mode
        True,  # is_amax_s
        False,  # is_amax_o
        False,  # is_amax_ds
        "bf16",  # "fp8_143", #"bf16",  # fp8_run_out_type
    ),
]

tc_list_inf = tc_list1 + tc_list2 + tc_list3 + tc_list4 + tc_list5 + tc_list6 + tc_list7 + tc_list8
tc_list_train = tc_list9 + tc_list10

tc_list = tc_list_inf + tc_list_train


# DONOT remove next line:Disable black formatting for easier parameter update
# fmt: off
"""
@pytest.mark.parametrize(
    "batch_size",
    (
        3,
    ),
    ids=lambda batch_size: f"batch_size-{batch_size}"
    )
@pytest.mark.parametrize(
    "q_heads",
    (
        4,
    ),
    ids=lambda q_heads: f"q_heads-{q_heads}"
)
@pytest.mark.parametrize(
    "kv_heads",
    (
        4, # same kv heads as q
        1, # MQA
        2, # GQA
    ),
    ids=lambda kv_heads: f"kv_heads-{kv_heads}"
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
        #False,  # enable for detailed test
    ),
    ids=lambda use_float_mask: f"use_float_mask-{use_float_mask}"
    )
@pytest.mark.parametrize(
    "enable_autocast",
    (
        True,
        #False, # not applicable for fp8 measurement or run
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
    "is_amax_s",
    (
        True,
        False,
    ),
    ids=lambda is_amax_s: f"is_amax_s-{is_amax_s}"
)
@pytest.mark.parametrize(
    "is_amax_o",
    (
        True,
        False,
    ),
    ids=lambda is_amax_o: f"is_amax_o-{is_amax_o}"
)
@pytest.mark.parametrize(
    "is_amax_ds",
    (
        True,
        False,
    ),
    ids=lambda is_amax_ds: f"is_amax_ds-{is_amax_ds}"
)
@pytest.mark.parametrize(
    "fp8_run_out_type",
    (
        "fp8_143",
        "bf16",
        #"None", # Not an fp8 run; can be a run for amax measurement
    ),
    ids=lambda fp8_run_out_type: f"fp8_run_out_type-{fp8_run_out_type}"
)
"""
# DONOT remove following line: re-enable black formatting
# fmt: on
@pytest.mark.parametrize(
    "batch_size,q_heads,kv_heads,seq_len_N_t,seq_len_N_s,head_dim_qk,head_dim_v,dropout_p,use_attn_mask,use_float_mask,enable_autocast,is_causal,recompute,rhslice,inference,softmax_mode,is_amax_s,is_amax_o,is_amax_ds,fp8_run_out_type",
    tc_list,
)

# @pytest.mark.skip(reason="Temporarily disabled : Lots of tests")
def test_sdpa(
    batch_size,
    q_heads,
    kv_heads,
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
    is_amax_o,
    is_amax_ds,
    fp8_run_out_type,
):
    test_case_valid = is_param_combo_valid(
        batch_size,
        q_heads,
        kv_heads,
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
        is_amax_o,
        is_amax_ds,
        fp8_run_out_type,
    )
    if is_gaudi1():
        pytest.skip("Fp8 tests not supported on G1")

    if not is_gaudi3():
        if not inference:
            if not check_dbg_env_var("PT_HPU_SDPA_FP8_152_152_FMT"):
                pytest.skip("Fp8 training tests with hybrid precision(143 -152) not currently supported on G1 or G2")

    # print("test_case_valid = ", test_case_valid)
    if not test_case_valid:
        pytest.skip("This testcase is not valid for fp8 measurement or run")

    if inference:
        os.environ["ENABLE_EXPERIMENTAL_FLAGS"] = "1"
        htcore.hpu_set_inference_env()

    torch.manual_seed(1234567)

    dtype = torch.float32
    grad_dtype = torch.float32
    rtol = 1e-3
    atol = 1e-3

    if enable_autocast:
        dtype = torch.bfloat16
        grad_dtype = torch.bfloat16
        rtol = 1e-3
        atol = 0.08

    fp8_run = is_fp8_run(fp8_run_out_type, inference, is_amax_s, is_amax_o, is_amax_ds)

    if is_amax_s and inference:
        assert fp8_run == False, "Fp8 measurement and run can not be True at the same time in inference"

    amax_s_ref = None
    amax_ds_ref = None
    amax_s_hpu_c = None
    amax_ds_hpu_c = None
    amax_o_hpu_c = None
    amax_o_atol = None
    amax_s_atol = 0.05

    if fp8_run:
        rtol = 1e-3
        atol = 0.4
        amax_o_atol = 2.0
        amax_ds_atol = 0.1

    if check_dbg_env_var("PT_HPU_SDPA_FP8_152_152_FMT"):
        fp8_dtype = torch.float8_e5m2
        grad_qk_atol = 1.0
        grad_v_atol = 0.12
        vb_print(" Running in 152_152_FMT")
    else:
        fp8_dtype = torch.float8_e4m3fn
        grad_qk_atol = 4.0
        grad_v_atol = 0.05
        vb_print(" Running in 143_152_FMT")

    if fp8_run:
        vb_print("fp8_dtype = ", fp8_dtype)

    attn_mask_shape = "Bx1x1xN"
    if use_float_mask:
        mask_dtype = dtype
    else:
        mask_dtype = torch.bool

    attn_scale = attention_scale
    vb_print("\nbatch_size = ", batch_size)
    vb_print("num_q_heads = ", q_heads)
    vb_print("num_kv_heads = ", kv_heads)
    vb_print("seq_len_N_s = ", seq_len_N_s)
    vb_print("head dim q k = ", head_dim_qk)
    vb_print("head dim v = ", head_dim_v)

    vb_print("dropout probability = ", dropout_p)
    vb_print("Using float attention mask = ", use_float_mask)

    vb_print("softmax mode = ", softmax_mode)
    vb_print("is_amax_s = ", is_amax_s)

    if q_heads == 0:  # special meaning ; no multi head attn . i.e, use 3d tensors
        q_shape = (batch_size, seq_len_N_t, head_dim_qk)
        k_shape = (batch_size, seq_len_N_s, head_dim_qk)
        v_shape = (batch_size, seq_len_N_s, head_dim_v)
        fwd_out_shape = (batch_size, seq_len_N_t, head_dim_v)
    else:  # Multi head attn with q_heads
        q_shape = (batch_size, q_heads, seq_len_N_t, head_dim_qk)
        k_shape = (batch_size, kv_heads, seq_len_N_s, head_dim_qk)
        v_shape = (batch_size, kv_heads, seq_len_N_s, head_dim_v)
        fwd_out_shape = (batch_size, q_heads, seq_len_N_t, head_dim_v)

    vb_print("q shape = ", q_shape)
    vb_print("k shape = ", k_shape)
    vb_print("v shape = ", v_shape)

    # q = torch.randn(q_shape).to(dtype).detach()
    if test_with_identity:
        q1 = torch.randn(q_shape).to(dtype).detach()
        qq = torch.eye(seq_len_N_t)
        q = qq.expand_as(q1).to(dtype).detach()

        kk = torch.eye(seq_len_N_s)
        k = kk.expand_as(q1).to(dtype).detach()

        vv = torch.eye(seq_len_N_s)
        v = vv.expand_as(q1).to(dtype).detach()

        g1 = torch.ones(fwd_out_shape).to(grad_dtype)
        gg = torch.eye(seq_len_N_t)
        g = gg.expand_as(g1).to(dtype).detach()
    else:
        q = torch.randn(q_shape).to(dtype).detach()
        k = torch.randn(k_shape).to(dtype).detach()
        v = torch.randn(v_shape).to(dtype).detach()
        g = torch.ones(fwd_out_shape).to(grad_dtype)

    # print(" dO = ", g)
    scaleQInv_hpu = scaleKInv_hpu = scaleVInv_hpu = scaleSInv_hpu = q_scale_s = q_scale_o = None

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
    g_hpu = g.to("hpu")

    if not inference:
        q_hpu = q_hpu.requires_grad_()
        k_hpu = k_hpu.requires_grad_()
        v_hpu = v_hpu.requires_grad_()

    if use_attn_mask:
        attn_mask = create_attention_mask_for_test(
            batch_size, q_heads, seq_len_N_t, seq_len_N_s, mask_dtype, attn_mask_shape, float_mask=use_float_mask
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

    # ------------------------------- Vanilla SDPA implementation on CPU for test----------------------------

    is_mqa(q_t, k_t)  # Just for info: For printing on console.

    if is_gqa(q_t, k_t):
        num_key_value_groups_ = q_heads // kv_heads
        k_t = gaudi_llama_repeat_kv(k_t, num_key_value_groups_)
        v_t = gaudi_llama_repeat_kv(v_t, num_key_value_groups_)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=enable_autocast):
        O_ref, P, amax_s_ref = vanilla_attention_impl_for_test(
            q_t,
            k_t,
            v_t,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            scale=attn_scale,
            is_causal=is_causal,
            is_amax_s=True,
        )

    vb_print("amax_s_ref = ", amax_s_ref)
    amax_o_ref = torch.max(O_ref).to(torch.float32)
    vb_print("amax_o_ref = ", amax_o_ref)

    if fp8_run:
        vb_print("TESTING fp8 FWD")

        scaleQ_hpu, scaleQInv_hpu = get_scale_values("q", q)
        scaleK_hpu, scaleKInv_hpu = get_scale_values("k", k)
        scaleV_hpu, scaleVInv_hpu = get_scale_values("v", v)

        q_hpu, _ = torch.ops.hpu.cast_to_fp8_v2(q_hpu, scaleQ_hpu, False, False, fp8_dtype)
        k_hpu, _ = torch.ops.hpu.cast_to_fp8_v2(k_hpu, scaleK_hpu, False, False, fp8_dtype)
        v_hpu, _ = torch.ops.hpu.cast_to_fp8_v2(v_hpu, scaleV_hpu, False, False, fp8_dtype)

        # scaleS_hpu = torch.tensor(1.0, dtype = torch.float32).to("hpu")
        scaleS_hpu, scaleSInv_hpu = get_scale_values("s", amax_s_ref, is_t_amax=True, scale_limit=128)
        q_scale_s = scaleS_hpu

        if fp8_run_out_type == "fp8_143":
            # Non-recomp does not support FWD output in Fp8. No q_scale_o
            if recompute:
                scaleO_hpu, _ = get_scale_values("o", O_ref)
                q_scale_o = scaleO_hpu

        # Let fp8 conversions and scale transfer to HPU be in a separate graph
        htcore.mark_step()

    # ----------------------------------HPU Fused SDPA attention---------------------------------------------
    with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=enable_autocast):
        # Use ht.sdp_kernel() context manager to enable/disable recompute based on pytest recompute parameter
        with ht.sdp_kernel(enable_recompute=recompute):
            model = TestModel(
                d_scale_q=scaleQInv_hpu,
                d_scale_k=scaleKInv_hpu,
                d_scale_v=scaleVInv_hpu,
                q_scale_s=q_scale_s,
                q_scale_o=q_scale_o,
                d_scale_s=get_d_scale_s(scaleSInv_hpu, inference, is_fwd=True),
                is_amax_s=is_amax_s,
                is_amax_o=is_amax_o,
                inference=inference,
            )

            if inference:
                # make scale tensors constant
                _mark_params_as_const(model)
                _check_params_as_const(model)
            O_hpu, amax_s, amax_o = model(
                q_hpu,
                k_hpu,
                v_hpu,
                attn_mask=attn_mask_hpu,
                dropout_p=dropout_p,
                is_causal=is_causal,
                softmax_mode=softmax_mode,
            )

    # ----------------------------------HPU Fused SDPA attention---------------------------------------------

    htcore.mark_step()

    # ................................................................................................

    # ................................................................................................
    # ------------------------------- Test Results Comparison ----------------------------
    vb_print("\n")
    O_hpu_c = O_hpu.detach().to("cpu")
    vb_print("DPA output dtype from HPU = ", O_hpu_c.dtype)
    if fp8_run and fp8_run_out_type == "fp8_143" and recompute == True:
        O_hpu_c = O_hpu_c.to(q_t.dtype) / q_scale_o.to("cpu").to(q_t.dtype)

    if is_amax_s:
        amax_s_hpu_c = amax_s.detach().to("cpu")
        vb_print("cpu amax_s = ", amax_s_ref)
        vb_print("hpu amax_s = ", amax_s_hpu_c)
    if is_amax_o and inference == False and recompute == True:
        amax_o_hpu_c = amax_o.detach().to("cpu")
        vb_print("cpu amax_o = ", amax_o_ref)
        vb_print("hpu amax_o = ", amax_o_hpu_c)
    else:
        amax_o_hpu_c = None
        amax_o_atol = None
    inference_results = [
        {
            "name": "amax_s",
            "compare": is_amax_s,
            "assert": True,
            "t_cpu": amax_s_ref,
            "t_hpu": amax_s_hpu_c,
            "rtol": rtol,
            "atol": amax_s_atol,
        },
        {
            "name": "amax_o",
            "compare": is_amax_o and inference == False and recompute == True,
            "assert": True,
            "t_cpu": amax_o_ref,
            "t_hpu": amax_o_hpu_c,
            "rtol": rtol,
            "atol": amax_o_atol,
        },
        {
            "name": "FWD out",
            "compare": True,
            "assert": True,
            "t_cpu": O_ref,
            "t_hpu": O_hpu_c,
            "rtol": rtol,
            "atol": atol,
        },
    ]
    process_results(inference_results)

    if inference:
        htcore.hpu_teardown_inference_env()
        os.environ["ENABLE_EXPERIMENTAL_FLAGS"] = "0"
        return

    # BWD is not supported in recomp mode
    if recompute:
        return

    # if test_backward == False:
    #    return

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=enable_autocast):

        dQ, dK, dV, amax_ds_ref = vanilla_attention_impl_bwd_for_test(
            g_t,
            q_t,
            k_t,
            v_t,
            P,
            scale=attn_scale,
            is_amax_ds=True,
        )

    vb_print("amax_ds_ref = ", amax_ds_ref)
    dm = None
    # print(" P = ", P)
    P_hpu = P.to("hpu").detach()
    if fp8_run:
        vb_print("TESTING fp8 BWD")
        grad_fp8_type = torch.float8_e5m2

        scaleG_hpu, scaleGInv_hpu = get_scale_values("g", g)
        scaleP_hpu, scalePInv_hpu = get_scale_values("P", P)

        # grad is converted to 152 format. But the diff with ref is more if done with stochastic rounding
        # TODO: check this
        # g_hpu, _ = torch.ops.hpu.cast_to_fp8_v2(g_hpu, scaleG_hpu, True, False, grad_fp8_type)
        g_hpu, _ = torch.ops.hpu.cast_to_fp8_v2(g_hpu, scaleG_hpu, False, False, grad_fp8_type)
        P_hpu, _ = torch.ops.hpu.cast_to_fp8_v2(P_hpu, scaleP_hpu, False, False, fp8_dtype)

        scaledS_hpu, scaledSInv_hpu = get_scale_values("dS", amax_ds_ref, is_t_amax=True, scale_limit=128)
        # Let fp8 conversions and scale transfer to HPU be in a separate graph
        htcore.mark_step()

    d_scale_q = scaleQInv_hpu
    d_scale_k = scaleKInv_hpu
    d_scale_v = scaleVInv_hpu
    d_scale_s = get_d_scale_s(scaleSInv_hpu, inference, is_fwd=False)
    d_scale_do = scaleGInv_hpu
    d_scale_ds = scaledSInv_hpu
    q_scale_s = None
    q_scale_ds = scaledS_hpu

    if attention_scale == None:
        scale = q_hpu.shape[-1] ** 0.5
        scale = 1.0 / scale
    else:
        scale = attention_scale

    # print("Phpu = ", P_hpu.to("cpu"))
    # print("dOhpu = ", g_hpu.to("cpu"))
    dump_bwd_api_params(
        g_hpu,
        q_hpu,
        k_hpu,
        v_hpu,
        P_hpu,
        dm,
        dropout_p,
        scale,
        d_scale_q,
        d_scale_k,
        d_scale_v,
        d_scale_s,
        d_scale_do,
        d_scale_ds,
        q_scale_s,
        q_scale_ds,
        is_amax_ds,
    )
    dq, dk, dv, amax_ds = torch.ops.hpu.fp8_sdpa_bwd(
        g_hpu,
        q_hpu,
        k_hpu,
        v_hpu,
        P_hpu,
        dm,
        is_causal,
        dropout_p,
        scale,
        d_scale_q,
        d_scale_k,
        d_scale_v,
        d_scale_s,
        d_scale_do,
        d_scale_ds,
        q_scale_s,
        q_scale_ds,
        is_amax_ds,
        O_hpu,
    )
    htcore.mark_step()
    if is_amax_ds:
        amax_ds_hpu_c = amax_ds.to("cpu")
        vb_print("amax_ds cpu = ", amax_ds_ref)
        vb_print("amax_ds hpu = ", amax_ds_hpu_c)

    if not inference:
        q_grad_hpu_c = dq.to("cpu")
        k_grad_hpu_c = dk.to("cpu")
        v_grad_hpu_c = dv.to("cpu")

    # print(" v_grad_CPU = ", dV)
    # print(" v_grad_hpu_c = ", v_grad_hpu_c)

    train_results = [
        {
            "name": "amax_ds",
            "compare": is_amax_ds,
            "assert": True,
            "t_cpu": amax_ds_ref,
            "t_hpu": amax_ds_hpu_c,
            "rtol": rtol,
            "atol": amax_ds_atol,
        },
        {
            "name": "dQ",
            "compare": True,
            "assert": True,
            "t_cpu": dQ,
            "t_hpu": q_grad_hpu_c,
            "rtol": rtol,
            "atol": grad_qk_atol,
        },
        {
            "name": "dK",
            "compare": True,
            "assert": True,
            "t_cpu": dK,
            "t_hpu": k_grad_hpu_c,
            "rtol": rtol,
            "atol": grad_qk_atol,
        },
        {
            "name": "dV",
            "compare": True,
            "assert": True,
            "t_cpu": dV,
            "t_hpu": v_grad_hpu_c,
            "rtol": rtol,
            "atol": grad_v_atol,
        },
    ]
    process_results(train_results)
