###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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


import math

# PREFERRED_SLICE_SIZE = 16
# MINIMUM_SIZE = 16
# PREFERRED_RESHAPE_SIZE = 16
# BATCH_AND_HEADS_SLICE_SIZE = 1
# SOFTMAX_NT_SLICE_SIZE = 1
# USE_RETAIN_TENSORS = True
from enum import IntEnum

import torch
import torch.nn.functional as F
from torch._higher_order_ops.hints_wrap import hints_wrapper
from torch._inductor.utils import argsort


class sdpa_q_slice_flow_selection_mask(IntEnum):
    SDPA_Q_SLICE_RECOMP_INFERENCE = 1 << 0
    SDPA_Q_SLICE_RECOMP_TRAIN_FWD = 1 << 1
    SDPA_Q_SLICE_RECOMP_TRAIN_BWD = 1 << 2
    SDPA_Q_SLICE_NON_RECOMP_INFERENCE = 1 << 3
    SDPA_Q_SLICE_NON_RECOMP_TRAIN_FWD = 1 << 4
    SDPA_Q_SLICE_NON_RECOMP_TRAIN_BWD = 1 << 5


class sdpaRHSliceFactors:
    def __init__(self):
        self.R = 0
        self.H = 0
        self.Rc = 0
        self.Hc = 0
        self.rMultiple = 0
        self.hMultiple = 0
        self.rRemainder = 0
        self.hRemainder = 0


class sdpaQSliceFactors:
    def __init__(self):
        self.requireQSlice = False
        self.multiple = 0
        self.remainder = 0
        self.q_slice_size = 0
        self.num_q_slices = 0
        self.q_seq_len = 0
        self.qSliceVec = []


def getQSliceSize():
    return 1024  # Gaudi2


def calcSoftmaxBufferSize(QShapes, KShapes, useQslice):
    io_fac = 2
    rank = QShapes.dim()
    r = QShapes.shape[0]
    h = QShapes.shape[1]
    ns = KShapes.shape[rank - 2]
    nt = QShapes.shape[rank - 2]
    if useQslice:
        nt = getQSliceSize()

    u = 1  # product of sizes at dims 2, ... rank-3
    for i in range(2, rank - 2):
        u *= QShapes[i]

    data_bytes = 4  # float type size
    total_size_in_use = io_fac * r * h * nt * ns * u * data_bytes
    return total_size_in_use


isTriangularMask = False


def FillQSliceFactors(QShapes, KShapes, flow_mask):
    qFactor = sdpaQSliceFactors()
    Nt = QShapes.shape[QShapes.dim() - 2]

    isTrm = False
    q_slice_size = getQSliceSize()
    multiple = Nt // q_slice_size
    remainder = Nt % q_slice_size

    # Calculate the buffer size needed to hold one q slice
    slice_buffer_size = calcSoftmaxBufferSize(QShapes, KShapes, True)
    requireQSlice = True

    qFactor.requireQSlice = requireQSlice
    qFactor.q_seq_len = Nt
    if requireQSlice:
        if isTriangularMask:
            reshapedSoftmaxSliceSize = 1024
            if (Nt > reshapedSoftmaxSliceSize) and (q_slice_size % reshapedSoftmaxSliceSize):
                raise AssertionError()
        useVariableQs = False
        if not useVariableQs:  # use uniform q slicing
            num_q_slices = multiple + 1 if remainder else multiple
            qFactor.multiple = multiple
            qFactor.remainder = remainder
            qFactor.q_slice_size = q_slice_size
            qFactor.num_q_slices = num_q_slices
    else:
        qFactor.multiple = 0
        qFactor.remainder = 0
        qFactor.q_slice_size = 0
        qFactor.num_q_slices = 0
    return qFactor


RIDX = 0  # Batch size axis as per FW order
HIDX = 1  # Num Heads axis as per FW order
NIDX = 2
DIDX = 3

useBatchNumHeadsSlicingAlgo = True
useBatchNumHeadsSlicingAlgoNoRecomp = True


def getMaxTensorBufferSize():
    defaultval = 2 * 1024 * 1024  # 256 MB on G2
    return defaultval


def FillRHSliceFactors(QShapes, KShapes, useQslice):
    rhFactor = sdpaRHSliceFactors()
    R = QShapes.shape[RIDX]
    H = QShapes.shape[HIDX]

    # compute Rc and Hc values either through env or through rules
    rhFactor.R = R
    rhFactor.H = H
    # No Slicing By default
    rhFactor.Rc = R
    rhFactor.Hc = H
    rc = 0  # getRc()
    hc = 0  # getHc()

    if not useBatchNumHeadsSlicingAlgo:
        # Do Nothing as rhFactor.Rc = R and rhFactor.Hc = H -> No Slice
        pass
    elif rc > 0 or hc > 0:
        # if only one env variable is enabled, we should use slice factor in
        # that dim and should not slice on another dim
        if hc == 0:
            hc = H
        if rc == 0:
            rc = R

        isValidSlice = (rc <= R) and (hc <= H)
        if not isValidSlice:
            raise AssertionError()

        rhFactor.Rc = rc
        rhFactor.Hc = hc
    else:
        total_size_in_use = calcSoftmaxBufferSize(QShapes, KShapes, useQslice)
        buffer_size = getMaxTensorBufferSize()

        if total_size_in_use <= buffer_size:
            # Do Nothing - No Slice
            pass
        elif (total_size_in_use // R) <= buffer_size:
            rhFactor.Rc = buffer_size // (total_size_in_use // R)
        elif total_size_in_use // (R * H) <= buffer_size:
            rhFactor.Rc = 1
            rhFactor.Hc = buffer_size // (total_size_in_use // (R * H))
        else:
            rhFactor.Rc = 1
            rhFactor.Hc = 1

    rhFactor.rMultiple = R // rhFactor.Rc
    rhFactor.hMultiple = H // rhFactor.Hc

    rhFactor.rRemainder = R % rhFactor.Rc
    rhFactor.hRemainder = H % rhFactor.Hc
    return rhFactor


def flex_attention_fwd(q, k, v, block_size=128, is_noop_mask=False, is_ret_lse=False):
    orig_dtype = q.dtype
    is_fp8 = False
    compatible_dtype = orig_dtype
    compatible_dtype_for_softmax = torch.float32
    if orig_dtype == torch.float8_e5m2 or orig_dtype == torch.float8_e4m3fn:
        is_fp8 = True
        compatible_dtype = torch.bfloat16
    batch = q.shape[q.dim() - 4]
    head = q.shape[q.dim() - 3]

    q_bucket_size = block_size
    k_bucket_size = block_size
    device = q.device
    working_precision = torch.float64 if q.dtype == torch.float64 else torch.float32
    if is_fp8:
        working_precision = torch.bfloat16
    max_neg_value = -torch.finfo(compatible_dtype_for_softmax).max
    neg_inf = float("-inf")
    scale = 1 / math.sqrt(q.size(-1))

    # gqa split Q across K heads
    kv_heads = k.shape[k.dim() - 3]
    q_heads = head // kv_heads
    is_gqa_enabled = q_heads > 1
    q_head_splits = torch.split(q, q_heads if is_gqa_enabled else head, dim=1)
    k_head_split = torch.split(k, 1 if is_gqa_enabled else kv_heads, dim=1)
    v_head_split = torch.split(v, 1 if is_gqa_enabled else kv_heads, dim=1)
    headqkv_splits = list(
        zip(
            q_head_splits,
            k_head_split,
            v_head_split,
            strict=False,
        )
    )
    out_o = []
    lse_o = []
    for _h_idx, (qhc, khc, vhc) in enumerate(headqkv_splits):
        row_splits = torch.split(qhc, q_bucket_size, dim=2)
        if not khc.is_contiguous():
            khc = khc.contiguous()
        if not vhc.is_contiguous():
            vhc = vhc.contiguous()
        khc = khc.repeat(1, q_heads, 1, 1) if is_gqa_enabled else k
        vhc = vhc.repeat(1, q_heads, 1, 1) if is_gqa_enabled else v
        out = []
        out_row_sums = []
        out_row_maxes = []

        for q_ind, qc in enumerate(row_splits):
            col_splits = zip(
                khc.split(k_bucket_size, dim=-2),
                vhc.split(k_bucket_size, dim=-2),
                strict=False,
            )
            out_c = torch.zeros_like(qc, dtype=compatible_dtype_for_softmax)
            row_sums = torch.zeros((*qc.shape[:-1], 1), device=device, dtype=compatible_dtype_for_softmax)
            row_maxes = torch.full(
                (*qc.shape[:-1], 1), max_neg_value, device=device, dtype=compatible_dtype_for_softmax
            )
            for k_ind, (kc, vc) in enumerate(col_splits):
                row_sums_c = row_sums.clone()
                row_maxes_c = row_maxes.clone()

                # non-contiguous Q, K, V
                if not q.is_contiguous():
                    qc = qc.contiguous()
                if not k.is_contiguous():
                    kc = kc.contiguous()
                if not v.is_contiguous():
                    vc = vc.contiguous()

                # For FP8, QK is in FP8 precision and output is in FP32
                qc = qc.to(compatible_dtype_for_softmax)
                kc = kc.to(compatible_dtype_for_softmax)
                attn_weights = torch.matmul(qc, kc.transpose(-2, -1)).to(working_precision)

                attn_weights = (attn_weights * scale).to(working_precision)

                # # apply score_mod
                b_blocks = [
                    torch.full(
                        (q_heads, qc.shape[qc.dim() - 2], kc.shape[kc.dim() - 2]),
                        fill_value=i,
                        dtype=torch.int64,
                        device=q.device,
                    )
                    for i in range(batch)
                ]
                b = torch.stack(b_blocks)
                # attn_weights = attn_weights + b

                h_base = (
                    torch.arange(q_heads, device=q.device, dtype=compatible_dtype_for_softmax)
                    .view(q_heads, 1, 1)
                    .expand(q_heads, qc.shape[qc.dim() - 2], kc.shape[kc.dim() - 2])
                )
                h = h_base.unsqueeze(0).repeat(batch, 1, 1, 1)
                # attn_weights = attn_weights + h

                q_base = (
                    torch.arange(
                        q_ind * qc.shape[qc.dim() - 2],
                        (q_ind + 1) * qc.shape[qc.dim() - 2],
                        device=q.device,
                        dtype=compatible_dtype_for_softmax,
                    )
                    .unsqueeze(1)
                    .repeat(1, attn_weights.shape[qc.dim() - 1])
                )
                q_head = q_base.unsqueeze(0).repeat(q_heads, 1, 1)
                q_idx = q_head.unsqueeze(0).repeat(batch, 1, 1, 1)
                # attn_weights = attn_weights + q_idx

                kv_base = (
                    torch.arange(
                        k_ind * kc.shape[kc.dim() - 2],
                        (k_ind + 1) * kc.shape[kc.dim() - 2],
                        device=q.device,
                        dtype=compatible_dtype_for_softmax,
                    )
                    .unsqueeze(0)
                    .repeat(attn_weights.shape[kc.dim() - 2], 1)
                )
                kv_head = kv_base.unsqueeze(0).repeat(q_heads, 1, 1)
                kv_idx = kv_head.unsqueeze(0).repeat(batch, 1, 1, 1)
                # attn_weights = attn_weights + kv_idx

                post_mod_scores = torch.ops.hpu.flex_attention_score_mod(attn_weights, b, h, q_idx, kv_idx)

                block_row_maxes = post_mod_scores.amax(dim=-1, keepdims=True)
                new_row_maxes = torch.maximum(block_row_maxes, row_maxes_c)

                # apply mask_mod
                if not is_noop_mask:
                    safe_scores = post_mod_scores - new_row_maxes
                    mask_mod_out = torch.ops.hpu.flex_attention_mask_mod(b, h, q_idx, kv_idx)
                    safe_post_mod_scores = torch.where(mask_mod_out, safe_scores, neg_inf)
                else:
                    safe_post_mod_scores = post_mod_scores - new_row_maxes

                exp_weights = torch.exp(safe_post_mod_scores)
                block_row_sums = exp_weights.sum(dim=-1, keepdims=True)

                # TODO: (QK) * V MatMul is precision of original dtype (FP8/BF16/FP32)
                exp_values = torch.matmul(exp_weights, vc.to(compatible_dtype_for_softmax)).to(
                    compatible_dtype_for_softmax
                )

                exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)
                new_row_sums = exp_row_max_diff * row_sums_c + block_row_sums

                out_c = out_c * exp_row_max_diff
                out_c = out_c + exp_values

                row_maxes = new_row_maxes
                row_sums = new_row_sums

            # div not supported for FP8 so computing on BF16
            # Reciprocal + multiply = div
            out_c = out_c / row_sums  # out_c * (1 / row_sums)
            out.append(out_c.to(orig_dtype))
            out_row_sums.append(row_sums)
            out_row_maxes.append(row_maxes)

        out_sums = torch.cat(out_row_sums, -2)
        out_maxes = torch.cat(out_row_maxes, -2)
        lse = (out_sums.to(compatible_dtype_for_softmax).log()) + out_maxes
        lse = lse.to(compatible_dtype_for_softmax) * 1.0 / math.log(2)
        lse_leaf = lse.squeeze(-1)
        # remove this WA leaf nodes runs eagerly on hpu
        lse = lse_leaf * 1.0
        ret = torch.cat(out, -2)
        lse_o.append(lse)
        out_o.append(ret)
    ret_o = torch.cat(out_o, -3)

    # inline output_stride computation since it is being traced
    # can't make it a function
    fill_order = argsort(q.stride())
    sizes = ret_o.shape
    out_strides = [0] * len(sizes)
    current_stride = 1
    for dim in fill_order:
        out_strides[dim] = current_stride
        current_stride *= sizes[dim]

    if not q.is_contiguous():
        if not is_fp8:
            ret_oas = torch.as_strided(ret_o, size=ret_o.shape, stride=out_strides)
            ret_o = ret_oas * 1.0
        else:
            ret_oas = torch.as_strided(ret_o, size=ret_o.shape, stride=out_strides)
            ret_oasbf16 = ret_oas.to(compatible_dtype)
            ret_ot = ret_oasbf16 * 1.0
            ret_o = ret_ot.to(orig_dtype)

    ret_lse = torch.cat(lse_o, -2)
    if is_ret_lse:
        packed_tensors = torch.ops.hpu.flex_attention_pack_tensors(ret_o, ret_lse.to(working_precision), None)
        return packed_tensors
    packed_tensors = torch.ops.hpu.flex_attention_pack_tensors(ret_o, None, None)
    return packed_tensors


# FlexAttention BWD with GQA support
def flex_attention_bwd(q, k, v, o, lse, do, glse, block_size=128, is_noop_mask=False):
    # (batch, number of heads, sequence length, dimension of each head)
    orig_dtype = q.dtype
    is_fp8 = False
    if orig_dtype == torch.float8_e5m2 or orig_dtype == torch.float8_e4m3fn:
        is_fp8 = True
    batch = q.shape[q.dim() - 4]
    head = q.shape[q.dim() - 3]
    kv_heads = k.shape[k.dim() - 3]

    # Number of Qs per KV in GQA
    q_heads = head // kv_heads
    q_head_orig = q_heads
    gqa_enabled = q_heads > 1

    q_bucket_size = block_size
    k_bucket_size = block_size
    working_precision = torch.float64 if q.dtype == torch.float64 else torch.float32
    neg_inf = float("-inf")

    scale = 1 / math.sqrt(q.size(-1))

    # Split Q, K, V according to GQA
    q_head_splits = torch.split(q, q_heads if gqa_enabled else head, dim=1)
    k_head_splits = torch.split(k, 1 if gqa_enabled else kv_heads, dim=1)
    v_head_splits = torch.split(v, 1 if gqa_enabled else kv_heads, dim=1)

    # Split auxillary tensors
    o_splits = torch.split(o, q_heads if gqa_enabled else head, dim=1)
    do_splits = torch.split(do, q_heads if gqa_enabled else head, dim=1)
    lse_splits = torch.split(lse, q_heads if gqa_enabled else head, dim=1)
    glse_splits = torch.split(glse, q_heads if gqa_enabled else head, dim=1)

    dq_out = []
    dk_out = []
    dv_out = []

    headqkv_splits = list(
        zip(
            q_head_splits,
            k_head_splits,
            v_head_splits,
            o_splits,
            do_splits,
            lse_splits,
            glse_splits,
            strict=False,
        )
    )

    # GQA loop
    for _h_idx, (qh, kh, vh, oh, doh, lseh, glseh) in enumerate(headqkv_splits):
        # Common K, V for current Q slice
        kh = kh.repeat(1, q_heads, 1, 1) if gqa_enabled else k
        vh = vh.repeat(1, q_heads, 1, 1) if gqa_enabled else v

        # Allocate gradients for current Q slice
        dq = torch.zeros_like(qh)
        dk = torch.zeros_like(kh)
        dv = torch.zeros_like(vh)

        # Split Q along sequence length
        row_splits = list(
            zip(
                qh.split(q_bucket_size, dim=-2),
                oh.split(q_bucket_size, dim=-2),
                doh.split(q_bucket_size, dim=-2),
                lseh.split(q_bucket_size, dim=-1),
                glseh.split(q_bucket_size, dim=-1),
                dq.split(q_bucket_size, dim=-2),
                strict=False,
            )
        )

        # Split K, V along sequence length
        col_splits = list(
            zip(
                kh.split(k_bucket_size, dim=-2),
                vh.split(k_bucket_size, dim=-2),
                strict=False,
            )
        )

        dqc_list = []
        dkc_list = []
        dvc_list = []

        for q_ind, (qc, oc, doc, lsec, glsec, dqc) in enumerate(row_splits):
            lsec = lsec * math.log(2)
            glsec = glsec / math.log(2)
            for k_ind, (kc, vc) in enumerate(col_splits):
                # non-contiguous q, k, v, o, do
                if not q.is_contiguous():
                    qc = qc.contiguous()
                if not k.is_contiguous():
                    kc = kc.contiguous()
                if not v.is_contiguous():
                    vc = vc.contiguous()
                if not o.is_contiguous():
                    oc = oc.contiguous()
                if not do.is_contiguous():
                    doc = doc.contiguous()
                attn_weights = torch.matmul(qc.to(working_precision), kc.transpose(-2, -1).to(working_precision)).to(
                    dtype=working_precision
                )
                attn_weights = (attn_weights * scale).to(working_precision)
                scores = attn_weights.clone()
                arg1 = scores.clone()

                # apply score_mod
                b_blocks = [
                    torch.full(
                        (q_head_orig, qc.shape[qc.dim() - 2], kc.shape[kc.dim() - 2]),
                        fill_value=i,
                        dtype=torch.int64,
                        device=q.device,
                    )
                    for i in range(batch)
                ]
                b = torch.stack(b_blocks)
                # attn_weights = attn_weights + b

                h_base = (
                    torch.arange(q_head_orig, device=q.device)
                    .view(q_head_orig, 1, 1)
                    .expand(q_head_orig, qc.shape[qc.dim() - 2], kc.shape[kc.dim() - 2])
                )
                h = h_base.unsqueeze(0).repeat(batch, 1, 1, 1)
                # attn_weights = attn_weights + h

                q_base = (
                    torch.arange(q_ind * qc.shape[qc.dim() - 2], (q_ind + 1) * qc.shape[qc.dim() - 2], device=q.device)
                    .unsqueeze(1)
                    .repeat(1, qc.shape[qc.dim() - 2])
                )

                q_head = q_base.unsqueeze(0).repeat(q_head_orig, 1, 1)
                q_idx = q_head.unsqueeze(0).repeat(batch, 1, 1, 1)
                # attn_weights = attn_weights + q_idx

                kv_base = (
                    torch.arange(k_ind * kc.shape[kc.dim() - 2], (k_ind + 1) * kc.shape[kc.dim() - 2], device=q.device)
                    .unsqueeze(0)
                    .repeat(kc.shape[kc.dim() - 2], 1)
                )
                kv_head = kv_base.unsqueeze(0).repeat(q_head_orig, 1, 1)
                kv_idx = kv_head.unsqueeze(0).repeat(batch, 1, 1, 1)
                # attn_weights = attn_weights + kv_idx

                post_score_mod_scores = torch.ops.hpu.flex_attention_score_mod(attn_weights, b, h, q_idx, kv_idx)

                # apply mask_mod
                if not is_noop_mask:
                    mask_mod_out = torch.ops.hpu.flex_attention_mask_mod(b, h, q_idx, kv_idx)
                    post_mod_scores = torch.where(mask_mod_out, post_score_mod_scores, neg_inf)
                else:
                    post_mod_scores = post_score_mod_scores

                p = torch.exp(post_mod_scores - lsec.unsqueeze(-1))
                dv_chunk = torch.matmul(p.transpose(-2, -1).to(working_precision), doc.to(working_precision)).to(
                    dtype=working_precision
                )
                if k_ind < len(dvc_list):
                    dvc_c = dvc_list[k_ind]
                    dvc_list[k_ind] = dvc_c + dv_chunk
                else:
                    dvc_list.append(dv_chunk)

                dp = torch.matmul(doc.to(working_precision), vc.transpose(-2, -1).to(working_precision)).to(
                    dtype=working_precision
                )
                D = (doc * oc).sum(dim=-1, keepdims=True)
                ds_pre_score_mod = p * (
                    dp.to(dtype=working_precision)
                    - D.to(dtype=working_precision)
                    + glsec.unsqueeze(-1).to(dtype=working_precision)
                )

                # print("Apply Score Mod")
                ds_post_score_mods = torch.ops.hpu.flex_attention_bwd_score_mod(
                    scores, b, h, q_idx, kv_idx, ds_pre_score_mod
                )
                ds_post_score_mod = ds_post_score_mods * scale

                # print("Apply Mask Mod")
                if not is_noop_mask:
                    mask_mod_out = torch.ops.hpu.flex_attention_mask_mod(b, h, q_idx, kv_idx)
                    ds = torch.where(mask_mod_out, ds_post_score_mod, 0.0)
                else:
                    ds = ds_post_score_mod
                dqc_c = dqc.clone()
                dq_chunk = torch.matmul(ds.to(working_precision), kc.to(working_precision))
                dqc_new = dqc_c.to(dtype=working_precision) + dq_chunk.to(dtype=working_precision)
                dqc = dqc_new * 1.0

                dk_chunk = torch.matmul(ds.transpose(-2, -1).to(working_precision), qc.to(working_precision)).to(
                    dtype=working_precision
                )
                if k_ind < len(dkc_list):
                    dkc_c = dkc_list[k_ind]
                    dkc_list[k_ind] = dkc_c + dk_chunk
                else:
                    dkc_list.append(dk_chunk)
            dqc_list.append(dqc)

        dq_group = torch.cat(dqc_list, dim=-2)
        dk_group = torch.cat(dkc_list, dim=-2)
        dv_group = torch.cat(dvc_list, dim=-2)

        dq_out.append(dq_group)
        dk_out.append(dk_group)
        dv_out.append(dv_group)

    # Concatenate across head dimension
    dv1 = torch.cat(dv_out, -3)
    dq1 = torch.cat(dq_out, -3)
    dk1 = torch.cat(dk_out, -3)

    B, C_qh, H, W = dk1.shape
    C = C_qh // q_heads
    grad_k = dk1.view(B, C, q_heads, H, W).sum(dim=2)
    B, C_qh, H, W = dv1.shape
    C = C_qh // q_heads
    grad_v = dv1.view(B, C, q_heads, H, W).sum(dim=2)
    dk1 = grad_k
    dv1 = grad_v

    # Q WA for tail view in fx graph
    #   inline output_stride computation since it is being traced
    #   can't make it a function
    fill_order = argsort(q.stride())
    sizes = dq1.shape
    dq_strides = [0] * len(sizes)
    current_stride = 1
    for dim in fill_order:
        dq_strides[dim] = current_stride
        current_stride *= sizes[dim]

    if not q.is_contiguous():
        if not is_fp8:
            dq1as = torch.as_strided(dq1, size=dq1.shape, stride=dq_strides)
            dq1 = dq1as * 1.0
        else:
            dq1as = torch.as_strided(dq1, size=dq1.shape, stride=dq_strides)
            dq1asbf16 = dq1as.to(torch.bfloat16)
            dq1t = dq1asbf16 * 1.0
            dq1 = dq1t.to(orig_dtype)

    # K WA for tail view in fx graph
    #   inline output_stride computation since it is being traced
    #   can't make it a function
    fill_order = argsort(k.stride())
    sizes = dk1.shape
    dk_strides = [0] * len(sizes)
    current_stride = 1
    for dim in fill_order:
        dk_strides[dim] = current_stride
        current_stride *= sizes[dim]

    if not k.is_contiguous():
        if not is_fp8:
            dk1as = torch.as_strided(dk1, size=dk1.shape, stride=dk_strides)
            dk1 = dk1as * 1.0
        else:
            dk1as = torch.as_strided(dk1, size=dk1.shape, stride=dk_strides)
            dk1asbf16 = dk1as.to(torch.bfloat16)
            dk1t = dk1asbf16 * 1.0
            dk1 = dk1t.to(orig_dtype)

    # V WA for tail view in fx graph
    #   inline output_stride computation since it is being traced
    #   can't make it a function
    fill_order = argsort(v.stride())
    sizes = dv1.shape
    dv_strides = [0] * len(sizes)
    current_stride = 1
    for dim in fill_order:
        dv_strides[dim] = current_stride
        current_stride *= sizes[dim]

    if not v.is_contiguous():
        if not is_fp8:
            dv1as = torch.as_strided(dv1, size=dv1.shape, stride=dv_strides)
            dv1 = dv1as * 1.0
        else:
            dv1as = torch.as_strided(dv1, size=dv1.shape, stride=dv_strides)
            dv1asbf16 = dv1as.to(torch.bfloat16)
            dv1t = dv1asbf16 * 1.0
            dv1 = dv1t.to(orig_dtype)

    packed_tensors = torch.ops.hpu.flex_attention_pack_tensors(
        dq1.to(orig_dtype), dk1.to(orig_dtype), dv1.to(orig_dtype)
    )
    return packed_tensors


def sdpa_fwd_4m_cguid(q, k, v, is_causal=False, with_slice=False):
    score_mode = []
    scale = 1 / math.sqrt(q.size(-1))
    qFactor = FillQSliceFactors(
        q,
        k,
        (
            sdpa_q_slice_flow_selection_mask.SDPA_Q_SLICE_NON_RECOMP_INFERENCE
            | sdpa_q_slice_flow_selection_mask.SDPA_Q_SLICE_NON_RECOMP_TRAIN_FWD
        ),
    )
    requireQSlice = qFactor.requireQSlice
    rhFactor = FillRHSliceFactors(q, k, requireQSlice)

    NoSlice = (rhFactor.Rc == rhFactor.R and rhFactor.Hc == rhFactor.H) or not useBatchNumHeadsSlicingAlgoNoRecomp

    if NoSlice:
        print("No support yet")  # [toDo]
    else:
        q = torch.relu(q)
        splitQR = torch.tensor_split(q, q.shape[RIDX], RIDX)
        splitKR = torch.tensor_split(k, k.shape[RIDX], RIDX)
        splitVR = torch.tensor_split(v, v.shape[RIDX], RIDX)
        OutR = []

        for i in range(len(splitQR)):
            QR = splitQR[i]
            KR = splitKR[i]
            VR = splitVR[i]
            splitQRH = torch.tensor_split(QR, QR.shape[HIDX], HIDX)
            splitKRH = torch.tensor_split(KR, KR.shape[HIDX], HIDX)
            splitVRH = torch.tensor_split(VR, VR.shape[HIDX], HIDX)
            OutRHVector = []
            for j in range(len(splitQRH)):
                Q = splitQRH[j]
                K = splitKRH[j]
                V = splitVRH[j]
                q_slice_size = qFactor.q_slice_size
                qSliceSizeVecSize = qFactor.num_q_slices
                q_slice_offset_begin = 0
                q_slice_offset_end = q_slice_size
                OutSliceVec = []
                for _j in range(qSliceSizeVecSize):
                    QSlice = Q[:, :, q_slice_offset_begin:q_slice_offset_end, :]
                    BMM1Outs = torch.matmul(QSlice, K.transpose(-2, -1))
                    Si = torch.mul(BMM1Outs, scale)  # (1, 1, 1k, 4k)
                    b = torch.arange(0, Si.shape[0], device=QSlice.device)
                    h = torch.arange(0, Si.shape[1], device=QSlice.device)
                    m = torch.arange(0, Si.shape[2], device=QSlice.device)
                    n = torch.arange(0, Si.shape[3], device=QSlice.device)
                    Si = torch.ops.hpu.flex_attention_score_mod(Si, b, h, m, n)
                    score_mode.append(Si)
                    Pi = F.softmax(Si, dim=-1)
                    q_slice_offset_begin = q_slice_offset_end
                    q_slice_offset_end = q_slice_offset_end + q_slice_size
                    AW = torch.matmul(Pi, V)
                    OutSliceVec.append(AW)
                fwdOutputsRHNt = torch.cat(OutSliceVec, NIDX)
                OutRHVector.append(fwdOutputsRHNt)
            OutRH = torch.cat(OutRHVector, HIDX)
            OutR.append(OutRH)
        out = torch.cat(OutR, RIDX)
    return out, score_mode


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
            Si = torch.mul(Si.to(torch.float32), scale.to(torch.float32))
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
        def forward_hinted(q, k, v, is_causal, with_slice):
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
                    Si = torch.mul(Si.to(torch.float32), scale.to(torch.float32))
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
            out, retain_exp, retain_max = hints_wrapper(
                forward_hinted,
                (
                    q,
                    k,
                    v,
                    is_causal,
                    with_slice,
                ),
                {},
                hints={"schedule_policy": "strict", "group_id": 0},
            )
            ctx.save_for_backward(q, k, v, out, retain_exp, retain_max)
        else:
            (out,) = hints_wrapper(
                forward_hinted,
                (
                    q,
                    k,
                    v,
                    is_causal,
                    with_slice,
                ),
                {},
                hints={"schedule_policy": "strict", "group_id": 0},
            )
            ctx.save_for_backward(q, k, v, out)

        ctx.with_slice = with_slice
        ctx.is_causal = is_causal

        return out

    @staticmethod
    def backward(ctx, dout):
        def backward_hinted(do, q, k, v, O, is_causal, with_slice, retain_exp, retain_max):
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

        dq, dk, dv = hints_wrapper(
            backward_hinted,
            (
                dout,
                q,
                k,
                v,
                out,
                ctx.is_causal,
                ctx.with_slice,
                retain_exp,
                retain_max,
            ),
            {},
            hints={"schedule_policy": "strict", "group_id": 1},
        )
        return dq, dk, dv, None, None, None
