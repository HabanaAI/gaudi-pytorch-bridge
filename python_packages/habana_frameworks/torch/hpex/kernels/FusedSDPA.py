###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
        gqa = kv_heads not in (q_heads, 1)
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


def sdpa_fwd_wrapper(
    ctx,
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    softmax_mode="None",
    recompute_mode=None,
    valid_seq_len=None,
    seq_padding_type="left",
    return_dropout_mask=False,
    return_attn_probs=False,
    window_size=(-1, -1),
):
    requires_backward = q.requires_grad or k.requires_grad or v.requires_grad

    # Handle zero sized tensors(for now only in inference) by returning a dummy output.
    if not requires_backward and (q.numel() == 0 or k.numel() == 0 or v.numel() == 0):
        out_shape = list(q.shape)
        out_shape[-1] = v.shape[-1]
        dummy_out = q.new_empty(out_shape, requires_grad=requires_backward, layout=q.layout)
        return dummy_out

    softmax_mode = softmax_mode.lower()
    seq_padding_type = seq_padding_type.lower()
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))

    # Check if recompute variant is enabled
    recompute = recompute_mode
    if recompute is None:
        recompute = ht.recompute_sdp_enabled()

    q_seq_len = q.size(-2)
    # In case of inference, override the mode set by user and set the mode internally
    # and go via recmpute mode unless returing attn prob is requested (since attn prob
    # is returned only in non-recomp mode). For now take recomp path only in case Q seq len is 1.
    # Later see if this condition can be removed.
    if not requires_backward:
        if q_seq_len == 1:
            recompute = True
        # force recomp to False if attn prob is to be returned
        if return_attn_probs:
            recompute = False

    if return_attn_probs is True:
        assert requires_backward is False, "return_attn_probs is supported only for inference mode"

    if recompute and requires_backward and softmax_mode == "fast":
        assert is_causal, (
            "Optimized softmax mode is supported in recompute training mode only in causal(triangular) mask case"
        )

    if valid_seq_len is not None:
        assert is_causal and (requires_backward is False), (
            "Valid sequence length is supported only in inference with is_causal(triangular) mask case"
        )

    if recompute:
        assert return_dropout_mask is False, "Return_dropout_mask is not supported in recompute mode"

    if softmax_mode == "fp32":
        q_dtype = q.dtype
        assert requires_backward is False and q_dtype == torch.bfloat16, (
            "softmax_mode = fp32 is supported only for inference mode and when q/k/v inputs are BF16"
        )

    # if attention mask is 2D (Nt, Ns) expand it to Q rank
    # along with the same batch size
    if attn_mask is not None and attn_mask.dim() == 2:
        assert attn_mask.shape[0] == q.shape[-2] and attn_mask.shape[1] == k.shape[-2], (
            "unsupported 2D attention mask, the shape is not (Nt, Ns)"
        )
        new_attn_mask_shape = [q.shape[0], attn_mask.shape[0], attn_mask.shape[1]]
        # the -1 is an already added batch_size
        rank_diff = q.dim() - attn_mask.dim() - 1
        if rank_diff > 0:
            new_attn_mask_shape[1:1] = rank_diff * [1]
        attn_mask = attn_mask.expand(new_attn_mask_shape)

    gqa = is_gqa(q, k)
    if gqa:
        q, k, v, attn_mask = gqa_input_reshape_fwd(q, k, v, attn_mask)
    if recompute:
        out, m, linv, seed = torch.ops.hpu.sdpa_recomp_fwd(
            q,
            k,
            v,
            attn_mask,
            dropout_p,
            scale,
            is_causal,
            requires_backward,
            softmax_mode,
            valid_seq_len,
            seq_padding_type,
            window_size,
        )
        if gqa:
            out = gqa_output_reshape(out)
        if not requires_backward:
            return out
        ctx.save_for_backward(q, k, v, attn_mask, m, linv, seed, out)
    else:
        out, P, dm = torch.ops.hpu.sdpa_fwd(
            q,
            k,
            v,
            attn_mask,
            dropout_p,
            scale,
            is_causal,
            softmax_mode,
            valid_seq_len,
            seq_padding_type,
        )
        if gqa:
            out = gqa_output_reshape(out)
        if not requires_backward:
            # return_dropout_mask   return_attn_probs       Outputs
            #       FALSE               FALSE               fwd_out
            #       FALSE               TRUE                fwd_out, P
            #       TRUE                FALSE               fwd_out, dm
            #       TRUE                TRUE                fwd_out, P, dm
            if gqa:
                if dropout_p > 0.0:
                    dm = gqa_output_reshape(dm)
                if return_attn_probs:
                    P = gqa_output_reshape(P)

            if return_dropout_mask:
                if return_attn_probs:
                    return out, P, dm
                else:
                    return out, dm
            elif return_attn_probs:
                return out, P
            else:
                return out

        ctx.save_for_backward(q, k, v, P, dm, out)

    ctx.dropout_p = dropout_p
    ctx.scale = scale
    ctx.is_causal = is_causal
    ctx.recompute = recompute
    ctx.gqa = gqa
    ctx.softmax_mode = softmax_mode

    if recompute:
        return out

    if not return_dropout_mask:
        return out
    else:
        if dropout_p > 0.0 and gqa:
            dm = gqa_output_reshape(dm)
        return out, dm


def sdpa_bwd_wrapper(ctx, dout, *args):
    if ctx.recompute:
        q, k, v, attn_mask, m, linv, seed, fwd_out = ctx.saved_tensors
        scale = ctx.scale
        dropout_p = ctx.dropout_p
        is_causal = ctx.is_causal
        softmax_mode = ctx.softmax_mode
        if ctx.gqa:
            dout = gqa_input_reshape_bwd(q, v, dout)
            fwd_out = gqa_input_reshape_bwd(q, v, fwd_out)
        dq, dk, dv = torch.ops.hpu.sdpa_recomp_bwd(
            dout,
            q,
            k,
            v,
            attn_mask,
            m,
            linv,
            seed,
            is_causal,
            dropout_p,
            scale,
            softmax_mode,
            fwd_out,
        )
        if ctx.gqa:
            dq = gqa_output_reshape(dq)
            dk = gqa_output_reshape(dk)
            dv = gqa_output_reshape(dv)
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None
    else:
        q, k, v, P, dm, fwd_out = ctx.saved_tensors
        scale = ctx.scale
        is_causal = ctx.is_causal
        dropout_p = ctx.dropout_p
        if ctx.gqa:
            dout = gqa_input_reshape_bwd(q, v, dout)
            fwd_out = gqa_input_reshape_bwd(q, v, fwd_out)
        dq, dk, dv = torch.ops.hpu.sdpa_bwd(dout, q, k, v, P, dm, is_causal, dropout_p, scale, fwd_out)
        if ctx.gqa:
            dq = gqa_output_reshape(dq)
            dk = gqa_output_reshape(dk)
            dv = gqa_output_reshape(dv)
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


class FusedSDPA(torch.autograd.Function):
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
        recompute_mode=None,
        valid_seq_len=None,
        seq_padding_type="left",
        return_dropout_mask=False,
        return_attn_probs=False,
        window_size=(-1, -1),
    ):
        return sdpa_fwd_wrapper(
            ctx,
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            softmax_mode=softmax_mode,
            recompute_mode=recompute_mode,
            valid_seq_len=valid_seq_len,
            seq_padding_type=seq_padding_type,
            return_dropout_mask=return_dropout_mask,
            return_attn_probs=return_attn_probs,
            window_size=window_size,
        )

    @staticmethod
    def backward(ctx, dout, *args):
        return sdpa_bwd_wrapper(ctx, dout, *args)
