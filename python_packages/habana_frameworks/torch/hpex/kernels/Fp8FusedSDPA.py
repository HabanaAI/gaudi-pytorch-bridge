###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import math  # for sqrt etc
import os

import habana_frameworks.torch.hpu as ht
import torch


# Please refer to FusedSDPA documentation at:
# https://docs.habana.ai/en/latest/PyTorch/Python_Packages.html#hpex-kernels-fusedsdpa
def check_dbg_env_var(v):
    env_var_set = False
    if int(os.getenv(v, 0)) == 1:
        env_var_set = True
    return env_var_set


def is_gqa(q, k):
    gqa = False
    dims = q.dim()
    if dims == 4:
        q_heads = q.shape[1]
        kv_heads = k.shape[1]
        gqa = (q_heads != kv_heads) and kv_heads != 1
    return gqa


def gqa_input_reshape_bwd(q, v, grad_in):
    new_shape = (q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[-1])
    return grad_in.reshape(new_shape)


def gqa_input_reshape_fwd(q, k, v, attention_mask):
    q_heads = q.shape[1]
    kv_heads = k.shape[1]

    q_heads_per_group = q_heads // kv_heads
    groups = kv_heads

    bs, heads, seq_len, h_dim = q.shape
    new_q_shape = (bs, groups, q_heads_per_group, seq_len, h_dim)
    q = q.reshape(new_q_shape)

    bs, heads, seq_len, h_dim = k.shape
    new_k_shape = (bs, groups, 1, seq_len, h_dim)
    k = k.reshape(new_k_shape)

    bs, heads, seq_len, h_dim = v.shape
    new_v_shape = (bs, groups, 1, seq_len, h_dim)
    v = v.reshape(new_v_shape)

    if attention_mask is not None:
        bs, heads, seq_len_t, seq_len_s = attention_mask.shape
        if heads == q_heads:  # attention mask shape = [batch size, q_heads, *, *]
            new_attn_mask_shape = (bs, groups, q_heads_per_group, seq_len_t, seq_len_s)
            attention_mask = attention_mask.reshape(new_attn_mask_shape)
        else:  # attention mask shape = [batch size, 1, *, *]
            attention_mask = attention_mask.unsqueeze(1)  # add groups dim and set to 1

    return q, k, v, attention_mask


def gqa_output_reshape(tensor):
    bs, groups, heads_per_group, seq_len, h_dim = tensor.shape
    new_shape = (bs, groups * heads_per_group, seq_len, h_dim)
    return tensor.reshape(new_shape)


def fp8_sdpa_fwd_wrapper(
    ctx,
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    softmax_mode="None",
    d_scale_q=None,
    d_scale_k=None,
    d_scale_v=None,
    q_scale_s=None,
    q_scale_o=None,
    d_scale_s=None,
    is_amax_s=False,
    is_amax_o=False,
    valid_seq_len=None,
    seq_padding_type="left",
):

    requires_backward = q.requires_grad or k.requires_grad or v.requires_grad
    softmax_mode = softmax_mode.lower()
    seq_padding_type = seq_padding_type.lower()
    if scale == None:
        scale = 1.0 / math.sqrt(q.size(-1))

    # Check if recompute variant is enabled
    recompute = ht.recompute_sdp_enabled()

    if requires_backward:
        assert is_causal == True, "Fp8 FusedSDPA in trining only supports Triangular mask"
    if valid_seq_len is not None:
        assert (
            is_causal and (requires_backward == False) and (attn_mask == None)
        ), "Valid sequence length is supported only in inference with is_causal(triangular) mask case"

    gqa = is_gqa(q, k)
    if gqa:
        q, k, v, attn_mask = gqa_input_reshape_fwd(q, k, v, attn_mask)

    amax_s = None
    amax_o = None
    if recompute:
        out, m, linv, seed, amax_s, amax_o = torch.ops.hpu.fp8_sdpa_recomp_fwd(
            q,
            k,
            v,
            attn_mask,
            dropout_p,
            scale,
            is_causal,
            requires_backward,
            softmax_mode,
            d_scale_q,
            d_scale_k,
            d_scale_v,
            q_scale_s,
            q_scale_o,
            d_scale_s,
            is_amax_s,
            is_amax_o,
            valid_seq_len,
            seq_padding_type,
        )

        if gqa:
            out = gqa_output_reshape(out)
        if not requires_backward:
            return out, amax_s, amax_o
        ctx.save_for_backward(q, k, v, attn_mask, m, linv, seed)
    else:
        out, P, dm, amax_s = torch.ops.hpu.fp8_sdpa_fwd(
            q,
            k,
            v,
            attn_mask,
            dropout_p,
            scale,
            is_causal,
            softmax_mode,
            d_scale_q,
            d_scale_k,
            d_scale_v,
            q_scale_s,
            q_scale_o,
            d_scale_s,
            is_amax_s,
            valid_seq_len,
            seq_padding_type,
        )
        if gqa:
            out = gqa_output_reshape(out)
        if not requires_backward:
            return out, amax_s, None
        ctx.save_for_backward(q, k, v, P, dm)

    ctx.dropout_p = dropout_p
    ctx.scale = scale
    ctx.is_causal = is_causal
    ctx.recompute = recompute
    ctx.gqa = gqa

    if recompute:
        return out, amax_s, amax_o

    if not check_dbg_env_var("FSDPA_DBG_USE_DROPOUT_STUB"):
        return out, amax_s, amax_o
    else:
        if gqa:
            dm = gqa_output_reshape(dm)
        return out, dm


def fp8_sdpa_bwd_wrapper(ctx, dout, *args):
    if ctx.recompute:
        q, k, v, attn_mask, m, linv, seed = ctx.saved_tensors
        scale = ctx.scale
        dropout_p = ctx.dropout_p
        is_causal = ctx.is_causal
        if ctx.gqa:
            dout = gqa_input_reshape_bwd(q, v, dout)
        dq, dk, dv = torch.ops.hpu.sdpa_recomp_bwd(dout, q, k, v, attn_mask, m, linv, seed, is_causal, dropout_p, scale)
        if ctx.gqa:
            dq = gqa_output_reshape(dq)
            dk = gqa_output_reshape(dk)
            dv = gqa_output_reshape(dv)
        return dq, dk, dv, None, None, None, None, None, None, None
    else:
        q, k, v, P, dm = ctx.saved_tensors
        scale = ctx.scale
        is_causal = ctx.is_causal
        dropout_p = ctx.dropout_p
        if ctx.gqa:
            dout = gqa_input_reshape_bwd(q, v, dout)
        dq, dk, dv = torch.ops.hpu.sdpa_bwd(dout, q, k, v, P, dm, is_causal, dropout_p, scale)
        if ctx.gqa:
            dq = gqa_output_reshape(dq)
            dk = gqa_output_reshape(dk)
            dv = gqa_output_reshape(dv)
        return dq, dk, dv, None, None, None, None, None, None, None


class Fp8FusedSDPA(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        softmax_mode="None",
        d_scale_q=None,
        d_scale_k=None,
        d_scale_v=None,
        q_scale_s=None,
        q_scale_o=None,
        d_scale_s=None,
        is_amax_s=False,
        is_amax_o=False,
        valid_seq_len=None,
        seq_padding_type="left",
    ):
        return fp8_sdpa_fwd_wrapper(
            ctx,
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            softmax_mode=softmax_mode,
            d_scale_q=d_scale_q,
            d_scale_k=d_scale_k,
            d_scale_v=d_scale_v,
            q_scale_s=q_scale_s,
            q_scale_o=q_scale_o,
            d_scale_s=d_scale_s,
            is_amax_s=is_amax_s,
            is_amax_o=is_amax_o,
            valid_seq_len=valid_seq_len,
            seq_padding_type=seq_padding_type,
        )

    @staticmethod
    def backward(ctx, dout, *args):
        return fp8_sdpa_bwd_wrapper(ctx, dout, *args)


def dump_api_params(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    softmax_mode="None",
    d_scale_q=None,
    d_scale_k=None,
    d_scale_v=None,
    q_scale_s=None,
    q_scale_o=None,
    d_scale_s=None,
    is_amax_s=False,
    is_amax_o=False,
    valid_seq_len=None,
    seq_padding_type="left",
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
    print("=" * 40, "FUSED_SDPA_API_PARAMS", "=" * 40)
    print_t_info("q", q)
    print_t_info("k", k)
    print_t_info("v", v)
    print_t_info("attn_mask", attn_mask)
    print("dropout_p : ", dropout_p)
    print("is_causal : ", is_causal)
    print("scale : ", scale)
    print("softmax_mode : ", softmax_mode)
    print_t_info("d_scale_q", d_scale_q, is_scale=True)
    print_t_info("d_scale_k", d_scale_k, is_scale=True)
    print_t_info("d_scale_v", d_scale_v, is_scale=True)
    print_t_info("q_scale_s", q_scale_s, is_scale=True)
    print_t_info("q_scale_o", q_scale_o, is_scale=True)
    print_t_info("d_scale_s", d_scale_s, is_scale=True)
    print("is_amax_s : ", is_amax_s)
    print("is_amax_o : ", is_amax_o)
    print_t_info("valid_seq_len", valid_seq_len)
    print("seq_padding_type : ", seq_padding_type)
    print("=" * 90)


def fp8_fused_sdpa(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    softmax_mode="None",
    d_scale_q=None,
    d_scale_k=None,
    d_scale_v=None,
    q_scale_s=None,
    q_scale_o=None,
    d_scale_s=None,
    is_amax_s=False,
    is_amax_o=False,
    valid_seq_len=None,
    seq_padding_type="left",
):
    dump_api_params(
        q,
        k,
        v,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        softmax_mode,
        d_scale_q,
        d_scale_k,
        d_scale_v,
        q_scale_s,
        q_scale_o,
        d_scale_s,
        is_amax_s,
        is_amax_o,
        valid_seq_len,
        seq_padding_type,
    )
    out, amax_s, amax_o = Fp8FusedSDPA.apply(
        q,
        k,
        v,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        softmax_mode,
        d_scale_q,
        d_scale_k,
        d_scale_v,
        q_scale_s,
        q_scale_o,
        d_scale_s,
        is_amax_s,
        is_amax_o,
        valid_seq_len,
        seq_padding_type,
    )

    return out, amax_s, amax_o
