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
import habana_frameworks.torch.hpu.random as rand_hpu

def check_dbg_env_var(v):
    env_var_set = False
    if int(os.getenv(v, 0)) == 1 :
        env_var_set = True
    return env_var_set

def sdpa_fwd_wrapper(ctx, q, k, v, attn_mask = None, dropout_p=0.0, is_causal = False, scale = None):

    if scale == None:
        scale = 1.0/math.sqrt(q.size(-1))

    # Work around to handle is_causal in in case source seq len < target seq len
    # Create the triangular mask, pass it usual attention mask. So clear is_causal flag
    if is_causal:
        seq_len_N_t = q.size(-2)
        seq_len_N_s = k.size(-2)
        if seq_len_N_s < seq_len_N_t:
            attn_mask = torch.ones(seq_len_N_t, seq_len_N_s, dtype=torch.bool, device = q.device).tril(diagonal=0)
            is_causal = False

    fwd = torch.ops.hpu.sdpa_fwd
    out, P, dm = fwd(q, k, v, attn_mask, dropout_p, scale, is_causal)
    ctx.save_for_backward(q, k, v, P, dm)

    ctx.dropout_p = dropout_p
    ctx.scale = scale
    if not check_dbg_env_var('FSDPA_DBG_USE_DROPOUT_STUB'):
        return out
    else:
        return out, dm

def sdpa_bwd_wrapper(ctx, dout, *args):
    bwd = torch.ops.hpu.sdpa_bwd
    q, k, v, P, dm = ctx.saved_tensors
    scale = ctx.scale
    dropout_p = ctx.dropout_p
    dq, dk, dv = bwd(dout,q,k,v,P, dm, dropout_p, scale)
    return dq, dk, dv, None, None, None, None


class FusedSDPA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, attn_mask = None, dropout_p=0.0, is_causal = False, scale = None):
        return sdpa_fwd_wrapper(ctx, q, k, v, attn_mask = attn_mask,
                     dropout_p=dropout_p, is_causal = is_causal, scale = scale)


    @staticmethod
    def backward(ctx, dout, *args):
        return sdpa_bwd_wrapper(ctx, dout, *args)


