# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# Changes:
# - Copied from linear.py as starting template
# - Introduced FusedSDPA class

"""Linear API"""
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import habana_frameworks.torch as ht
import torch
from habana_frameworks.torch.hpex.kernels import FusedSDPA as sdpa_kernel
from habana_frameworks.torch.hpex.kernels import fp8_sdpa_bwd_wrapper, fp8_sdpa_fwd_wrapper, gqa_output_reshape
from habana_frameworks.torch.hpex.kernels.FusedSDPA import gqa_input_reshape_bwd
from torch.nn.parameter import Parameter

from ..cpp_extensions import _update_amax_history, cast_to_fp8
from ..fp8 import MetaTensorType, get_fp8_te_dtype, get_fp8_te_sr, get_meta_tensor_key, is_fp8_enabled, is_hybrid_mode
from ..utils import FP8BwdTensors, FP8FwdTensors, is_gaudi3
from .base import TransformerEngineBaseModule, _prepare_backward

__all__ = ["FusedSDPA"]

FP8_META_ID_Q = FP8FwdTensors.GEMM1_INPUT
FP8_META_ID_K = FP8FwdTensors.GEMM1_WEIGHT
FP8_META_ID_V = FP8FwdTensors.GEMM2_INPUT
FP8_META_ID_S = FP8FwdTensors.GEMM2_WEIGHT
FP8_META_ID_DO = FP8BwdTensors.GRAD_OUTPUT1
FP8_META_ID_DS = FP8BwdTensors.GRAD_OUTPUT2

# Debug switches
DUMP_TENSORS_FLAG = "PT_TE_DUMP_TENSORS"
DUMP_TENSORS_PATH = os.getenv("PT_TE_DUMP_TENSORS_PATH", "te_tensors")
OVERRIDE_TE_SDPA_DOUT_PATH = os.getenv("PT_TE_OVERRIDE_SDPA_DOUT", "")
OVERRIDE_SDPA_PRECISSION = os.getenv("PT_TE_OVERRIDE_SDPA_BWD_PRECISSION", "")


def DumpTensor(tensor, path):
    if not os.path.exists(DUMP_TENSORS_PATH):
        os.makedirs(DUMP_TENSORS_PATH)
    torch.save(tensor, f"{DUMP_TENSORS_PATH}/{path}")


class FusedAttnFunc(torch.autograd.Function):
    """Function for FusedAttention with separate Q, K, V tensors"""

    @staticmethod
    def forward(
        ctx,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        attention_dropout,
        is_causal: bool,
        scale,
        softmax_mode,
        fp8: bool,
        fp8_meta: Dict[str, Any],
        amax_measure_state: dict,
        is_scale_update_required: bool,
    ) -> torch.Tensor:
        requires_grad = query_layer.requires_grad or key_layer.requires_grad or value_layer.requires_grad
        fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
        hybrid_mode = is_hybrid_mode(fp8_meta)
        assert fp8
        if int(os.getenv(DUMP_TENSORS_FLAG, 0)) == 1:
            q_path = f"{fp8_meta['name']}_q_{fp8_meta['run_cnt']}.pt"
            k_path = f"{fp8_meta['name']}_k_{fp8_meta['run_cnt']}.pt"
            v_path = f"{fp8_meta['name']}_v_{fp8_meta['run_cnt']}.pt"
            DumpTensor(query_layer, q_path)
            DumpTensor(key_layer, k_path)
            DumpTensor(value_layer, v_path)

        meta_fwd_key = get_meta_tensor_key(MetaTensorType.FORWARD)
        if not hybrid_mode or is_gaudi3():
            query_layer = cast_to_fp8(
                query_layer,
                fp8_meta[meta_fwd_key],
                FP8_META_ID_Q,
                fp8_dtype_forward,
                stochastic_rounding=get_fp8_te_sr(fp8_meta["recipe"], fprop_tensor=True),
                measure_amax=amax_measure_state["fwd_enabled"],
            )
            key_layer = cast_to_fp8(
                key_layer,
                fp8_meta[meta_fwd_key],
                FP8_META_ID_K,
                fp8_dtype_forward,
                stochastic_rounding=get_fp8_te_sr(fp8_meta["recipe"], fprop_tensor=True),
                measure_amax=amax_measure_state["fwd_enabled"],
            )
            value_layer = cast_to_fp8(
                value_layer,
                fp8_meta[meta_fwd_key],
                FP8_META_ID_V,
                fp8_dtype_forward,
                stochastic_rounding=get_fp8_te_sr(fp8_meta["recipe"], fprop_tensor=True),
                measure_amax=amax_measure_state["fwd_enabled"],
            )
            scale_cache_key = meta_fwd_key
        else:
            assert False, "Hybrid mode not supported on Gaudi2 yet"

        # WA for fp8_sdpa_fwd_wrapper relying on .requires_grad flag
        if requires_grad:
            query_layer.requires_grad_()

        out, amax_s, amax_o = fp8_sdpa_fwd_wrapper(
            ctx,
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=attention_dropout,
            is_causal=is_causal,
            scale=scale,
            softmax_mode=softmax_mode,
            d_scale_q=fp8_meta[scale_cache_key].scale_inv[FP8_META_ID_Q],
            d_scale_k=fp8_meta[scale_cache_key].scale_inv[FP8_META_ID_K],
            d_scale_v=fp8_meta[scale_cache_key].scale_inv[FP8_META_ID_V],
            q_scale_s=fp8_meta[scale_cache_key].scale[FP8_META_ID_S],
            q_scale_o=None,
            d_scale_s=None,
            is_amax_s=amax_measure_state["fwd_enabled"] or not hybrid_mode,
            is_amax_o=False,
        )
        if amax_measure_state["fwd_enabled"]:
            _update_amax_history(amax_s, fp8_meta[scale_cache_key], FP8_META_ID_S)

        if requires_grad:
            ctx.fp8 = fp8
            ctx.fp8_meta = fp8_meta
            ctx.amax_measure_state = amax_measure_state.copy()
            ctx.is_scale_update_required = is_scale_update_required
            ctx.fwd_scale_inv = fp8_meta[scale_cache_key].scale_inv.clone()

        return out

    @staticmethod
    def backward(ctx, dout, *args):
        global OVERRIDE_TE_SDPA_DOUT_PATH
        if OVERRIDE_TE_SDPA_DOUT_PATH:
            print(f"Overriding te.FusedAttention dout with {OVERRIDE_TE_SDPA_DOUT_PATH}")
            dout = torch.load(OVERRIDE_TE_SDPA_DOUT_PATH)
        with _prepare_backward(
            ctx.fp8,
            ctx.fp8_meta,
            ctx.amax_measure_state,
            ctx.is_scale_update_required,
            False,
            None,
        ):
            query_layer, key_layer, value_layer, P, dm = ctx.saved_tensors
            scale = ctx.scale
            dropout_p = ctx.dropout_p
            is_causal = ctx.is_causal
            recompute = ctx.recompute
            gqa = ctx.gqa
            fp8_meta = ctx.fp8_meta
            amax_measure_state = ctx.amax_measure_state
            fwd_scale_inv = ctx.fwd_scale_inv
            if int(os.getenv(DUMP_TENSORS_FLAG, 0)) == 1:
                dout_path = f"{fp8_meta['name']}_dout_{fp8_meta['run_cnt']}.pt"
                DumpTensor(dout, dout_path)

            global OVERRIDE_SDPA_PRECISSION
            if OVERRIDE_SDPA_PRECISSION == "bf16":
                if ctx.gqa:
                    dout = gqa_input_reshape_bwd(query_layer, value_layer, dout)
                query_layer = torch.ops.hpu.cast_from_fp8(query_layer, fwd_scale_inv[FP8_META_ID_Q], torch.bfloat16)
                key_layer = torch.ops.hpu.cast_from_fp8(key_layer, fwd_scale_inv[FP8_META_ID_K], torch.bfloat16)
                value_layer = torch.ops.hpu.cast_from_fp8(value_layer, fwd_scale_inv[FP8_META_ID_V], torch.bfloat16)
                P = torch.ops.hpu.cast_from_fp8(P, fwd_scale_inv[FP8_META_ID_S], torch.bfloat16)
                dq, dk, dv = torch.ops.hpu.sdpa_bwd(
                    dout, query_layer, key_layer, value_layer, P, dm, is_causal, dropout_p, scale
                )
                if ctx.gqa:
                    dq = gqa_output_reshape(dq)
                    dk = gqa_output_reshape(dk)
                    dv = gqa_output_reshape(dv)
            else:
                fp8_dtype_backward = get_fp8_te_dtype(ctx.fp8_meta["recipe"], fprop_tensor=False)
                meta_bwd_key = get_meta_tensor_key(MetaTensorType.BACKWARD)
                dout_fp8 = cast_to_fp8(
                    dout,
                    ctx.fp8_meta[meta_bwd_key],
                    FP8_META_ID_DO,
                    fp8_dtype_backward,
                    stochastic_rounding=get_fp8_te_sr(ctx.fp8_meta["recipe"], fprop_tensor=False),
                    measure_amax=amax_measure_state["bwd_enabled"],
                )

                # TODO use fp8_sdpa_bwd_wrapper instead
                if ctx.gqa:
                    dout_fp8 = gqa_input_reshape_bwd(query_layer, value_layer, dout_fp8)
                dq, dk, dv, amax_ds = torch.ops.hpu.fp8_sdpa_bwd(
                    dout_fp8,
                    query_layer,
                    key_layer,
                    value_layer,
                    P,
                    dm,
                    is_causal,
                    dropout_p,
                    scale,
                    fwd_scale_inv[FP8_META_ID_Q],
                    fwd_scale_inv[FP8_META_ID_K],
                    fwd_scale_inv[FP8_META_ID_V],
                    fwd_scale_inv[FP8_META_ID_S],
                    fp8_meta[meta_bwd_key].scale_inv[FP8_META_ID_DO],
                    fp8_meta[meta_bwd_key].scale_inv[FP8_META_ID_DS],
                    None,
                    fp8_meta[meta_bwd_key].scale[FP8_META_ID_DS],
                    amax_measure_state["bwd_enabled"],
                )
                if amax_measure_state["bwd_enabled"]:
                    _update_amax_history(amax_ds, fp8_meta[meta_bwd_key], FP8_META_ID_DS)

            if ctx.gqa:
                dq = gqa_output_reshape(dq)
                dk = gqa_output_reshape(dk)
                dv = gqa_output_reshape(dv)
            if int(os.getenv(DUMP_TENSORS_FLAG, 0)) == 1:
                dq_path = f"{fp8_meta['name']}_dq_{fp8_meta['run_cnt']}.pt"
                dk_path = f"{fp8_meta['name']}_dk_{fp8_meta['run_cnt']}.pt"
                dv_path = f"{fp8_meta['name']}_dv_{fp8_meta['run_cnt']}.pt"
                DumpTensor(dq, dq_path)
                DumpTensor(dk, dk_path)
                DumpTensor(dv, dv_path)
            return dq, dk, dv, None, None, None, None, None, None, None, None, None


class FusedAttention(TransformerEngineBaseModule):
    """Fused dot product attention"""

    def __init__(
        self,
        scale: float,
        attention_dropout: float,
        enable_recompute: bool,
    ) -> None:
        super().__init__()
        self.name = self.name + "_FusedAttention"
        self.enable_recompute = enable_recompute
        self.scale = scale
        self.attention_dropout = attention_dropout

    def get_fp8_weights_scratchpad(
        self,
        is_first_microbatch: Union[bool, None],
    ) -> List[torch.Tensor]:
        """
        No weights in FusedAttention
        """
        return None

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        softmax_mode="None",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply the fused attention transformation to the q, k, v inputs.

        Parameters
        ----------
        q, k, v : torch.Tensor
             Input tensors.
        """

        if not is_fp8_enabled():
            with ht.hpu.sdp_kernel(enable_recompute=self.enable_recompute):
                return sdpa_kernel.apply(
                    query_layer,
                    key_layer,
                    value_layer,
                    attention_mask,
                    self.attention_dropout,
                    is_causal,
                    self.scale,
                    softmax_mode,
                )

        # SDPA requires 4 scaled fwd and 2 backward tensors,
        # this maps to 2 gemms in terms of number of scaling placeholders
        with self.prepare_forward(None, None, num_gemms=2) as (_, is_scale_update_required):
            with ht.hpu.sdp_kernel(enable_recompute=self.enable_recompute):
                out = FusedAttnFunc.apply(
                    query_layer,
                    key_layer,
                    value_layer,
                    attention_mask,
                    self.attention_dropout,
                    is_causal,
                    self.scale,
                    softmax_mode,
                    True,
                    self.fp8_meta,
                    self.get_amax_measure_state(),
                    is_scale_update_required,
                )

        return out
