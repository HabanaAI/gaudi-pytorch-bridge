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

import torch
import math # for sqrt etc
import os
import habana_frameworks.torch.hpu as ht

# Please refer to FusedSDPA documentation at:
# https://docs.habana.ai/en/latest/PyTorch/Python_Packages.html#hpex-kernels-fusedsdpa
def check_dbg_env_var(v):
    env_var_set = False
    if int(os.getenv(v, 0)) == 1 :
        env_var_set = True
    return env_var_set

def sdpa_fwd_wrapper(ctx, q, k, v, attn_mask = None, dropout_p=0.0, is_causal = False, scale = None):

    requires_backward = q.requires_grad or k.requires_grad or v.requires_grad
    if scale == None:
        scale = 1.0/math.sqrt(q.size(-1))

    # Check if recompute variant is enabled
    recompute = ht.recompute_sdp_enabled()

    # Work around to handle is_causal in case source seq len < target seq len.
    # Create the triangular mask and pass it as usual attention mask. So clear is_causal flag.
    # Make it a float mask that can be added to the S tensor (S = q@k.transpose)
    if recompute :
        if requires_backward:
            assert not attn_mask, "In recompute mode, Attention mask(attn_mask !=None) is supported only in inference case"

    if is_causal:
        seq_len_N_t = q.size(-2)
        seq_len_N_s = k.size(-2)
        if seq_len_N_s < seq_len_N_t:
            assert recompute == False, "Recompute is supported only if is_causal = True and source seq Len >= target seq Len"
            LNG = -3.0e38 #Close to -ve max for bfloat or float
            if q.dtype == torch.float16:
                LNG = -6.5e4
            inv_causal_mask = torch.ones(seq_len_N_t, seq_len_N_s, dtype=q.dtype, device = q.device).triu(diagonal=1)
            attn_mask =  inv_causal_mask * LNG
            is_causal = False
            recompute = False

    if recompute:
        out, m, linv, seed = torch.ops.hpu.sdpa_recomp_fwd(q, k, v, attn_mask, dropout_p, scale, is_causal, requires_backward)
        if not requires_backward:
            return out
        ctx.save_for_backward(q, k, v, attn_mask, m, linv, seed)
    else:
       out, P, dm = torch.ops.hpu.sdpa_fwd(q, k, v, attn_mask, dropout_p, scale, is_causal)
       ctx.save_for_backward(q, k, v, P, dm)

    ctx.dropout_p = dropout_p
    ctx.scale = scale
    ctx.is_causal = is_causal
    ctx.recompute = recompute

    if recompute:
        return out

    if not check_dbg_env_var('FSDPA_DBG_USE_DROPOUT_STUB'):
        return out
    else:
        return out, dm

def sdpa_bwd_wrapper(ctx, dout, *args):
    if ctx.recompute:
        q, k, v, attn_mask, m, linv, seed = ctx.saved_tensors
        scale = ctx.scale
        dropout_p = ctx.dropout_p
        dq, dk, dv = torch.ops.hpu.sdpa_recomp_bwd(dout,q,k,v, attn_mask, m, linv, seed, dropout_p, scale)
        return dq, dk, dv, None, None, None, None, None
    else:
        q, k, v, P, dm = ctx.saved_tensors
        scale = ctx.scale
        dropout_p = ctx.dropout_p
        dq, dk, dv = torch.ops.hpu.sdpa_bwd(dout,q,k,v,P, dm, dropout_p, scale)
        return dq, dk, dv, None, None, None, None, None


class FusedSDPA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, attn_mask = None, dropout_p=0.0, is_causal = False, scale = None):
        return sdpa_fwd_wrapper(ctx, q, k, v, attn_mask = attn_mask,
                     dropout_p=dropout_p, is_causal = is_causal, scale = scale)


    @staticmethod
    def backward(ctx, dout, *args):
        return sdpa_bwd_wrapper(ctx, dout, *args)


