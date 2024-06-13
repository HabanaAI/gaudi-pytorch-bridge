###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._higher_order_ops.hinted_context import hinted_context

# PREFERRED_SLICE_SIZE = 16
# MINIMUM_SIZE = 16
# PREFERRED_RESHAPE_SIZE = 16

# BATCH_AND_HEADS_SLICE_SIZE = 1
# SOFTMAX_NT_SLICE_SIZE = 1
# USE_RETAIN_TENSORS = True


def sdpa_fwd(ctx, q, k, v, is_causal, with_slice):
    """
    1. using retain tensor
    2. if slice enabled, using default size 1
    """
    # no slice on batch heads dimensions
    batch_heads = 1
    if with_slice:
        batch_heads = q.shape[0] * q.shape[1]

    scale = 1 / math.sqrt(q.size(-1))

    Q = torch.flatten(q, end_dim=1)
    K = torch.flatten(k, end_dim=1)
    V = torch.flatten(v, end_dim=1)

    Qi = torch.tensor_split(Q, batch_heads)
    Ki = torch.tensor_split(K, batch_heads)
    Vi = torch.tensor_split(V, batch_heads)

    retain_exp_slices = []
    retain_max_slices = []
    output_slices = []
    for slice in range(batch_heads):
        current_Qi = Qi[slice]
        current_Ki = Ki[slice]
        current_Vi = Vi[slice]

        Si = torch.matmul(current_Qi, current_Ki.transpose(-2, -1))

        if is_causal:
            Pi, retain_exp, retain_max = torch.ops.hpu.scaled_triangular_softmax_retain(Si, scale)

            retain_exp_slices.append(retain_exp.unsqueeze(0))
            retain_max_slices.append(retain_max.unsqueeze(0))
        else:
            Si = torch.mul(Si, scale)
            Pi = F.softmax(Si, dim=-1)

        output_slices.append(torch.matmul(Pi, current_Vi))

    retain_exp = None
    retain_max = None
    if is_causal:
        retain_exp = torch.cat(retain_exp_slices)
        retain_max = torch.cat(retain_max_slices)

    return (
        torch.cat(output_slices).reshape((q.shape[0], q.shape[1], v.shape[2], v.shape[3])),
        retain_exp,
        retain_max,
    )


def sdpa_bwd(do, q, k, v, O, is_causal, retain_exp, retain_max, with_slice):
    """
    1. using retain tensor
    2. if slice enabled, using default size 1
    """
    assert q.dim() == 4, " Currently support only 4D"

    # no slice on batch heads dimensions
    batch_heads = 1
    if with_slice:
        batch_heads = q.shape[0] * q.shape[1]

    scale = 1 / math.sqrt(q.size(-1))

    Q = torch.flatten(q, end_dim=1)
    K = torch.flatten(k, end_dim=1)
    V = torch.flatten(v, end_dim=1)
    dO = torch.flatten(do, end_dim=1)

    Qi = torch.tensor_split(Q, batch_heads)
    Ki = torch.tensor_split(K, batch_heads)
    Vi = torch.tensor_split(V, batch_heads)
    dOi = torch.tensor_split(dO, batch_heads)

    dQ_slices = []
    dK_slices = []
    dV_slices = []
    for slice in range(batch_heads):
        current_Qi = Qi[slice]
        current_Ki = Ki[slice]
        current_Vi = Vi[slice]
        current_dOi = dOi[slice]

        Si = torch.matmul(current_Qi, current_Ki.transpose(-2, -1))

        if is_causal:
            # current_exp_i = None
            # current_max_i = None

            current_exp_i = retain_exp.select(0, slice)
            current_max_i = retain_max.select(0, slice)

            Pi = torch.ops.hpu.scaled_triangular_softmax(Si, scale, current_exp_i, current_max_i)

        else:
            Si = torch.mul(Si, scale)
            Pi = F.softmax(Si, dim=-1)

        dV_slices.append(torch.matmul(Pi.transpose(-2, -1), current_dOi))

        dP = torch.matmul(current_dOi, current_Vi.transpose(-2, -1))
        dS = torch._softmax_backward_data(dP, Pi, -1, Pi.dtype)

        dK_tmp = torch.matmul(dS.transpose(-2, -1), current_Qi)
        dK_slices.append(torch.mul(dK_tmp, scale))

        dQ_tmp = torch.matmul(dS, current_Ki)
        dQ_slices.append(torch.mul(dQ_tmp, scale))

    return (
        torch.cat(dQ_slices).reshape_as(q),
        torch.cat(dK_slices).reshape_as(k),
        torch.cat(dV_slices).reshape_as(v),
    )


class PySDPA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False, with_slice=False):

        out, retain_exp, retain_max = sdpa_fwd(ctx, q, k, v, is_causal, with_slice)

        ctx.save_for_backward(q, k, v, out, retain_exp, retain_max)

        ctx.is_causal = is_causal
        ctx.with_slice = with_slice

        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, retain_exp, retain_max = ctx.saved_tensors
        dq, dk, dv = sdpa_bwd(dout, q, k, v, out, ctx.is_causal, retain_exp, retain_max, ctx.with_slice)
        return dq, dk, dv, None, None, None


class PySDPAHinted(torch.autograd.Function):
    """
    1. using retain tensor
    2. if slice enabled, using default size 1
    """

    @staticmethod
    def forward(ctx, q, k, v, is_causal=False, with_slice=True):
        def forward_hinted(ctx, q, k, v, is_causal, with_slice, hint):
            # using default slice size 1
            batch_heads = q.shape[0] * q.shape[1]

            scale = 1 / math.sqrt(q.size(-1))

            Q = torch.flatten(q, end_dim=1)
            K = torch.flatten(k, end_dim=1)
            V = torch.flatten(v, end_dim=1)

            Qi = torch.tensor_split(Q, batch_heads)
            Ki = torch.tensor_split(K, batch_heads)
            Vi = torch.tensor_split(V, batch_heads)

            retain_exp_slices = []
            retain_max_slices = []
            output_slices = []
            for slice in range(batch_heads):
                current_Qi = Qi[slice]
                current_Ki = Ki[slice]
                current_Vi = Vi[slice]

                Si = torch.matmul(current_Qi, current_Ki.transpose(-2, -1))

                if is_causal:
                    Pi, retain_exp, retain_max = torch.ops.hpu.scaled_triangular_softmax_retain(Si, scale)

                    retain_exp_slices.append(retain_exp.unsqueeze(0))
                    retain_max_slices.append(retain_max.unsqueeze(0))
                else:
                    Si = torch.mul(Si, scale)
                    Pi = F.softmax(Si, dim=-1)

                output_slices.append(torch.matmul(Pi, current_Vi))

            # workaround for the issue:
            #  "HigherOrderOperator body's output must consist of tensors only"
            outputs = [torch.cat(output_slices).reshape((q.shape[0], q.shape[1], v.shape[2], v.shape[3]))]

            retain_exp = None
            retain_max = None
            if is_causal:
                outputs.append(torch.cat(retain_exp_slices))
                outputs.append(torch.cat(retain_max_slices))

            return tuple(outputs)

        if is_causal:
            out, retain_exp, retain_max = hinted_context(
                forward_hinted,
                ctx,
                q,
                k,
                v,
                is_causal,
                with_slice,
                hint='{"preserve_order": "true", "group_id": 0}',
            )
            ctx.save_for_backward(q, k, v, out, retain_exp, retain_max)
        else:
            (out,) = hinted_context(
                forward_hinted,
                ctx,
                q,
                k,
                v,
                is_causal,
                with_slice,
                hint='{"preserve_order": "true", "group_id": 0}',
            )
            ctx.save_for_backward(q, k, v, out)

        ctx.with_slice = with_slice
        ctx.is_causal = is_causal

        return out

    @staticmethod
    def backward(ctx, dout):
        def backward_hinted(do, q, k, v, O, is_causal, with_slice, retain_exp, retain_max, hint):
            assert q.dim() == 4, " Currently support only 4D"

            # using default slice size 1
            batch_heads = q.shape[0] * q.shape[1]

            scale = 1 / math.sqrt(q.size(-1))

            Q = torch.flatten(q, end_dim=1)
            K = torch.flatten(k, end_dim=1)
            V = torch.flatten(v, end_dim=1)
            dO = torch.flatten(do, end_dim=1)

            Qi = torch.tensor_split(Q, batch_heads)
            Ki = torch.tensor_split(K, batch_heads)
            Vi = torch.tensor_split(V, batch_heads)
            dOi = torch.tensor_split(dO, batch_heads)

            dQ_slices = []
            dK_slices = []
            dV_slices = []
            for slice in range(batch_heads):
                current_Qi = Qi[slice]
                current_Ki = Ki[slice]
                current_Vi = Vi[slice]
                current_dOi = dOi[slice]

                Si = torch.matmul(current_Qi, current_Ki.transpose(-2, -1))

                if is_causal:
                    current_exp_i = None
                    current_max_i = None

                    current_exp_i = retain_exp.select(0, slice)
                    current_max_i = retain_max.select(0, slice)

                    Pi = torch.ops.hpu.scaled_triangular_softmax(Si, scale, current_exp_i, current_max_i)

                else:
                    Si = torch.mul(Si, scale)
                    Pi = F.softmax(Si, dim=-1)

                dV_slices.append(torch.matmul(Pi.transpose(-2, -1), current_dOi))

                dP = torch.matmul(current_dOi, current_Vi.transpose(-2, -1))
                dS = torch._softmax_backward_data(dP, Pi, -1, Pi.dtype)

                dK_tmp = torch.matmul(dS.transpose(-2, -1), current_Qi)
                dK_slices.append(torch.mul(dK_tmp, scale))

                dQ_tmp = torch.matmul(dS, current_Ki)
                dQ_slices.append(torch.mul(dQ_tmp, scale))

            return (
                torch.cat(dQ_slices).reshape_as(q),
                torch.cat(dK_slices).reshape_as(k),
                torch.cat(dV_slices).reshape_as(v),
            )

        if ctx.is_causal:
            q, k, v, out, retain_exp, retain_max = ctx.saved_tensors
        else:
            q, k, v, out = ctx.saved_tensors
            retain_exp, retain_max = None, None

        dq, dk, dv = hinted_context(
            backward_hinted,
            dout,
            q,
            k,
            v,
            out,
            ctx.is_causal,
            ctx.with_slice,
            retain_exp,
            retain_max,
            hint='{"preserve_order": "true", "group_id": 1}',
        )
        return dq, dk, dv, None, None, None
