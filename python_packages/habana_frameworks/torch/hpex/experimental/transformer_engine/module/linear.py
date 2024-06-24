# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# Changes:
# - Changed device type to "hpu"
# - Removed unused code paths

"""Linear API"""
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from habana_frameworks.torch import _hpex_C as tex
from torch.nn.parameter import Parameter

from ..constants import GemmParallelModes, dist_group_type
from ..cpp_extensions import cast_to_fp8, cast_to_fp8_hybrid, fp8_gemm
from ..distributed import (
    allreduce,
    gather_along_first_dim,
    gather_along_last_dim,
    get_distributed_world_size,
    initialize_affine_weight_hpu,
    reduce_scatter_along_first_dim,
    set_tensor_model_parallel_attributes,
)
from ..fp8 import MetaTensorType, get_fp8_te_dtype, get_fp8_te_sr, get_meta_tensor_key, is_fp8_enabled, is_hybrid_mode
from ..utils import FP8BwdTensors, FP8FwdTensors, cast_if_needed, divide, get_default_init_method
from .base import TransformerEngineBaseModule, _prepare_backward

__all__ = ["Linear"]

# Debug switches
DUMP_TENSORS_FLAG = "PT_TE_DUMP_TENSORS"
DUMP_TENSORS_PATH = os.getenv("PT_TE_DUMP_TENSORS_PATH", "te_tensors")


def DumpTensor(tensor, path):
    if not os.path.exists(DUMP_TENSORS_PATH):
        os.makedirs(DUMP_TENSORS_PATH)
    torch.save(tensor, f"{DUMP_TENSORS_PATH}/{path}")


class _Linear(torch.autograd.Function):
    """Linear semi-top level module
    Calls custom hpu extensions.
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        weight_fp8_fwd: torch.Tensor,
        weight_fp8_bwd: torch.Tensor,
        inp: torch.Tensor,
        bias: torch.Tensor,
        use_bias: bool,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_meta: Dict[str, Any],
        tp_group: Union[dist_group_type, None],
        sequence_parallel: bool,
        tensor_parallel: bool,
        activation_dtype: torch.dtype,
        parallel_mode: Union[str, None],
        minimize_memory: bool,
        amax_measure_state: dict,
        is_scale_update_required: bool,
    ) -> torch.Tensor:
        if int(os.getenv(DUMP_TENSORS_FLAG, 0)) == 1:
            inp_path = (
                f"{fp8_meta['name']}_inp_{fp8_meta['run_cnt']}_recompute_{fp8_meta['in_activation_recompute_phase']}.pt"
            )
            weight_path = f"{fp8_meta['name']}_weight_{fp8_meta['run_cnt']}_recompute_{fp8_meta['in_activation_recompute_phase']}.pt"
            DumpTensor(inp, inp_path)
            DumpTensor(weight.data, weight_path)

        # Make sure input dimensions are compatible
        in_features = weight.shape[-1]
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.view((-1, in_features))

        update_fp8_weights = is_first_microbatch is None or is_first_microbatch

        # Cast for native AMP
        inputmat = cast_if_needed(inputmat, activation_dtype)
        inputmat_no_fp8 = inputmat

        assert fp8
        fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)

        hybrid_mode = is_hybrid_mode(fp8_meta)
        if hybrid_mode:
            assert (weight_fp8_fwd is None) == (
                weight_fp8_bwd is None
            ), "Internal TE errror: Either both fp8 weight placeholders need to be None, or both need to be passed"

        meta_fwd_key = get_meta_tensor_key(MetaTensorType.FORWARD)
        if not hybrid_mode:
            inputmat = cast_to_fp8(
                inputmat,
                fp8_meta[meta_fwd_key],
                FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
                stochastic_rounding=get_fp8_te_sr(fp8_meta["recipe"], fprop_tensor=True),
                measure_amax=amax_measure_state["fwd_enabled"],
            )

            if update_fp8_weights:
                casted = cast_to_fp8(
                    weight,
                    fp8_meta[meta_fwd_key],
                    FP8FwdTensors.GEMM1_WEIGHT,
                    fp8_dtype_forward,
                    stochastic_rounding=get_fp8_te_sr(fp8_meta["recipe"], fprop_tensor=True),
                    measure_amax=amax_measure_state["fwd_enabled"],
                )
                if weight_fp8_fwd is None:
                    weight_fp8_fwd = casted
                else:
                    assert (
                        weight.shape == weight_fp8_fwd.shape
                    ), "Module initialized with different shape than received weight"
                    weight_fp8_fwd.copy_(casted)

            inputmat_fp8_for_bwd = inputmat
            weight_fp8_for_bwd = weight_fp8_fwd
            scale_cache_key = meta_fwd_key
        else:
            # NOTE: In case mixed precision gemm is not supported and fwd type differs from bwd type,
            # we need to remember activations in backward type
            # TODO: Support mixed precision
            meta_hybrid_key = get_meta_tensor_key(MetaTensorType.HYBRID)
            fp8_dtype_backward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=False)

            assert (
                fp8_dtype_backward == torch.float8_e5m2 and fp8_dtype_forward == torch.float8_e4m3fn
            ), "Only E4M3 fwd E5M2 bwd hybrid mode supported"

            inputmat_bwd, inputmat_fwd = cast_to_fp8_hybrid(
                inputmat_no_fp8,
                fp8_meta[meta_hybrid_key],
                fp8_meta[meta_fwd_key],
                FP8FwdTensors.GEMM1_INPUT,
                measure_amax=amax_measure_state["fwd_enabled"],
            )

            if update_fp8_weights:
                casted_bwd, casted_fwd = cast_to_fp8_hybrid(
                    weight,
                    fp8_meta[meta_hybrid_key],
                    fp8_meta[meta_fwd_key],
                    FP8FwdTensors.GEMM1_WEIGHT,
                    measure_amax=amax_measure_state["fwd_enabled"],
                )
                if weight_fp8_fwd is None:
                    weight_fp8_fwd = casted_fwd
                    weight_fp8_bwd = casted_bwd
                else:
                    assert (
                        weight.shape == weight_fp8_bwd.shape
                    ), "Module initialized with different shape than received weight"
                    weight_fp8_fwd.copy_(casted_fwd)
                    weight_fp8_bwd.copy_(casted_bwd)

            inputmat = inputmat_fwd
            inputmat_fp8_for_bwd = inputmat_bwd
            weight_fp8_for_bwd = weight_fp8_bwd
            scale_cache_key = meta_hybrid_key

        # TODO: Column Parallel Linear
        bias_dtype = torch.bfloat16 if activation_dtype == torch.float32 else activation_dtype
        bias = cast_if_needed(bias, bias_dtype) if use_bias else bias

        out = fp8_gemm(
            weight_fp8_fwd,
            fp8_meta[meta_fwd_key].scale_inv[FP8FwdTensors.GEMM1_WEIGHT],
            inputmat,
            fp8_meta[meta_fwd_key].scale_inv[FP8FwdTensors.GEMM1_INPUT],
            activation_dtype,
            bias=bias,
            use_bias=use_bias,
        )

        fp8_wgrad = fp8 and not fp8_meta["recipe"].override_linear_precision.wgrad

        # NOTE: In case is_first_microbatch is not None, weight_fp8 is stored in the module and is shared
        # between all fwds and bwds. As a result, fp8 weight cannot be cached for backward in the first microbatch,
        # because next fwd will override its value (input bf16 weight will be the same but scale will be different,
        # so the resulting fp8 weight will differ), so in the first microbatch bwd fp8 weight and scale_inv
        # would not match - and the calculated dgrad will be incorrect.
        cache_weight_fp8 = fp8 and not minimize_memory and not is_first_microbatch and inp.requires_grad

        if inp.requires_grad or weight.requires_grad:
            ctx.save_for_backward(
                inputmat_no_fp8 if weight.requires_grad and not fp8_wgrad else None,
                inputmat_fp8_for_bwd if weight.requires_grad and fp8_wgrad else None,
                weight_fp8_for_bwd if cache_weight_fp8 else None,
                weight,
                fp8_meta[scale_cache_key].scale_inv.clone() if fp8 else None,
                fp8_meta[scale_cache_key].scale.clone() if fp8 else None,
            )
            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.fp8_meta = fp8_meta
            ctx.use_bias = use_bias
            ctx.sequence_parallel = sequence_parallel
            ctx.tensor_parallel = tensor_parallel
            ctx.inp_shape = inp.shape
            ctx.parallel_mode = parallel_mode
            ctx.tp_group = tp_group
            ctx.amax_measure_state = amax_measure_state.copy()
            ctx.is_scale_update_required = is_scale_update_required
            ctx.requires_wgrad = weight.requires_grad
            ctx.requires_dgrad = inp.requires_grad

        # Row Parallel Linear
        if parallel_mode == "row" and sequence_parallel:
            out, _ = reduce_scatter_along_first_dim(out, tp_group)
        elif parallel_mode == "row" and tensor_parallel:
            out, _ = allreduce(out, tp_group)

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        return out.view(-1, *inp.shape[1:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        with _prepare_backward(
            ctx.fp8,
            ctx.fp8_meta,
            ctx.amax_measure_state,
            ctx.is_scale_update_required,
            ctx.sequence_parallel,
            ctx.tp_group,
        ):
            (
                inputmat,
                inputmap_fp8,
                weight_fp8,
                weight,
                fwd_scale_inverses,
                fwd_scales,
            ) = ctx.saved_tensors

            if int(os.getenv(DUMP_TENSORS_FLAG, 0)) == 1:
                inputmap_fp8_path = f"{ctx.fp8_meta['name']}_inputmap_fp8_{ctx.fp8_meta['run_cnt']}.pt"
                weight_bwd_path = f"{ctx.fp8_meta['name']}_weight_bwd_{ctx.fp8_meta['run_cnt']}.pt"
                grad_output_path = f"{ctx.fp8_meta['name']}_grad_output_{ctx.fp8_meta['run_cnt']}.pt"
                DumpTensor(inputmap_fp8.to(torch.bfloat16), inputmap_fp8_path)
                DumpTensor(weight, weight_bwd_path)
                DumpTensor(grad_output, grad_output_path)

            (
                grad_output,
                grad_output_c,
                grad_bias,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx, grad_output, ctx.parallel_mode == "row", ctx.amax_measure_state
            )

            # Column Parallel Linear
            # Overlap input AG with dgrad
            if ctx.requires_wgrad and ctx.parallel_mode == "column" and ctx.sequence_parallel:
                if ctx.fp8 and not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                    inputmat_fp8_total, handle = gather_along_last_dim(
                        inputmap_fp8, ctx.tp_group, async_op=ctx.requires_dgrad
                    )
                else:
                    inputmat_total, handle = gather_along_first_dim(inputmat, ctx.tp_group, async_op=ctx.requires_dgrad)
            else:
                inputmat_fp8_total = inputmap_fp8
                inputmat_total = inputmat
                handle = None

            assert ctx.fp8
            fp8_dtype_backward = get_fp8_te_dtype(ctx.fp8_meta["recipe"], fprop_tensor=False)

            if weight_fp8 is None:
                # If weight_fp8 was not remembered from fwd pass, recompute it
                weight_fp8, _ = torch.ops.hpu.cast_to_fp8_v2(
                    weight,
                    fwd_scales[FP8FwdTensors.GEMM1_WEIGHT],
                    stochastic_rounding=get_fp8_te_sr(ctx.fp8_meta["recipe"], fprop_tensor=True),
                    is_amax=False,
                    dtype=fp8_dtype_backward,
                )

            meta_bwd_key = get_meta_tensor_key(MetaTensorType.BACKWARD)
            if ctx.requires_dgrad:
                dgrad = fp8_gemm(
                    weight_fp8,
                    fwd_scale_inverses[FP8FwdTensors.GEMM1_WEIGHT],
                    grad_output_c,
                    ctx.fp8_meta[meta_bwd_key].scale_inv[FP8BwdTensors.GRAD_OUTPUT1],
                    ctx.activation_dtype,
                    transa=False,
                )

                # Overlap dgrad-RS/AR with wgrad
                if ctx.parallel_mode == "column" and ctx.sequence_parallel:
                    if handle is not None:
                        handle.wait()
                    dgrad, handle = reduce_scatter_along_first_dim(dgrad, ctx.tp_group, async_op=True)
                elif ctx.parallel_mode == "column" and ctx.tensor_parallel:
                    dgrad, handle = allreduce(dgrad, ctx.tp_group, async_op=True)

            if ctx.requires_wgrad:
                # WGRAD
                assert not ctx.fp8_meta["recipe"].override_linear_precision.wgrad
                wgrad = fp8_gemm(
                    inputmat_fp8_total,
                    fwd_scale_inverses[FP8FwdTensors.GEMM1_INPUT],
                    grad_output_c,
                    ctx.fp8_meta[meta_bwd_key].scale_inv[FP8BwdTensors.GRAD_OUTPUT1],
                    ctx.activation_dtype,
                    accumulate=False,
                    out=None,
                    transa=False,
                    transb=True,
                )

            # Column Parallel Linear
            if ctx.parallel_mode == "column" and ctx.tensor_parallel and handle is not None:
                handle.wait()

            if not ctx.use_bias:
                grad_bias = None

        return (
            wgrad if ctx.requires_wgrad else None,
            None,
            None,
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class Linear(TransformerEngineBaseModule):
    """
    Applies a linear transformation to the incoming data :math:`y = xA^T + b`

    On HPUs it is a drop-in replacement for `torch.nn.Linear`.

    Parameters
    ----------
    in_features : int
                 size of each input sample.
    out_features : int
                  size of each output sample.
    bias : bool, default = `True`
          if set to `False`, the layer will not learn an additive bias.
    init_method : Callable, default = `None`
                 used for initializing weights in the following way: `init_method(weight)`.
                 When set to `None`, defaults to `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    device : Union[torch.device, str], default = "hpu"
          The device on which the parameters of the model will allocated. It is the user's
          responsibility to ensure all parameters are moved to the HPU before running the
          forward pass.

    Parallelism parameters
    ----------------------
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    tp_size : int, default = 1
             used as TP (tensor parallel) world size when TP groups are not formed during
             initialization. In this case, users must call the
             `set_tensor_parallel_group(tp_group)` method on the initialized module before the
             forward pass to supply the tensor parallel group needed for tensor and sequence
             parallel collectives.
    parallel_mode : {None, 'Column', 'Row'}, default = `None`
                   used to decide whether this Linear layer is Column Parallel Linear or Row
                   Parallel Linear as described `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
                   When set to `None`, no communication is performed.
    skip_weight_param_allocation: bool, default = `False`
                                 if set to `True`, weight parameter is not allocated and must be
                                 passed as a keyword argument `weight` during the forward pass.

    Optimization parameters
    -----------------------
    return_bias : bool, default = `False`
                 when set to `True`, this module will not apply the additive bias itself, but
                 instead return the bias value during the forward pass together with the
                 output of the linear transformation :math:`y = xA^T`. This is useful when
                 the bias addition can be fused to subsequent operations.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in HPU memory.
    minimize_memory : bool, default = `False`
                     when set to `True`, memory usage is decreased by recalculating fp8 weight
                     in backward pass. This reduces memory usage but obviously degrades perf.
                     It works especially well with deepspeed pipelining mechanism.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sequence_parallel: bool = False,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        return_bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        parallel_mode: Optional[str] = None,
        skip_weight_param_allocation: bool = False,
        device: Union[torch.device, str] = "hpu",
        minimize_memory: bool = False,
    ) -> None:
        super().__init__()
        self.name = self.name + "_Linear"

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.minimize_memory = minimize_memory

        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)

        self.parallel_mode = parallel_mode
        assert self.parallel_mode in GemmParallelModes, f"parallel_mode {parallel_mode} not supported"

        assert not sequence_parallel, "sequence_parallel not supported"

        if self.parallel_mode == "column":
            self.out_features = divide(self.out_features, self.tp_size)
        elif self.parallel_mode == "row":
            self.in_features = divide(self.in_features, self.tp_size)

        if init_method is None:
            init_method = get_default_init_method()

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel

        if not skip_weight_param_allocation:
            self.weight = Parameter(
                torch.empty(
                    self.out_features,
                    self.in_features,
                    device=device,
                    dtype=params_dtype,
                )
            )

            initialize_affine_weight_hpu(
                self.weight,
                init_method,
                get_rng_state_tracker,
                partition_dim=1 if self.parallel_mode == "row" else 0,
                stride=1,
            )

            if self.use_bias:
                self.bias = Parameter(
                    torch.empty(
                        self.out_features,
                        device=device,
                        dtype=params_dtype,
                    )
                )
                if self.parallel_mode == "column":
                    set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
            else:
                self.bias = torch.Tensor().to(dtype=params_dtype, device=device)

            with torch.no_grad():
                self.bias.zero_()

        self.fp8_weight_shapes.append(torch.Size((self.out_features, self.in_features)))

        # For RPL, bias has to be added after TP collectives
        # So it cannot be fused with the GEMM
        if self.parallel_mode == "row" and self.apply_bias:
            self.gemm_bias_unfused_add = True
        else:
            self.gemm_bias_unfused_add = False

    def get_fp8_weights_scratchpad(
        self,
        is_first_microbatch: Union[bool, None],
    ) -> List[torch.Tensor]:
        """
        Fetch the fp8 weight tensor placeholders if they exist (when
        `is_first_microbatch` is not `None`), return None otherwise
        """
        if not self.fp8 or is_first_microbatch is None:
            return [None, None]

        # These persistent weight placeholders should've been created in
        # `set_fp8_weights` method
        if is_hybrid_mode(self.fp8_meta):
            return [self.weight1_fp8_fwd, self.weight1_fp8_bwd]

        return [self.weight1_fp8_fwd, None]

    def forward(
        self,
        inp: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        is_first_microbatch: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply the linear transformation to the input.

        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
        weight : torch.Tensor, default = None
                An optional weight tensor for the module. This argument is compulsory if module
                is initialized with `skip_weight_param_allocation=True`
        bias : torch.Tensor, default = None
              An optional bias tensor for the module. This argument is compulsory if module
              is initialized with `skip_weight_param_allocation=True` and one of `use_bias`
              or `return_bias`
        is_first_microbatch : {True, False, None}, default = None
                             During training using either gradient accumulation or
                             pipeline parallelism a minibatch of data is further split
                             into microbatches. Between the microbatches of the same minibatch
                             the model weights are not updated. Setting this parameter indicates
                             whether the current microbatch is the first in a minibatch or not.
                             When set, this parameter enables additional optimizations:

                             * during FP8 training, it allows caching of the FP8 versions of
                               the weights
                             * it also allows skipping gradient accumulation during the
                               first microbatch (since it is the first gradient being
                               produced)
        """

        bias_tensor = bias if bias is not None else self.bias if self.use_bias or self.return_bias else None
        weight_tensor = weight if weight is not None else self.weight

        if not is_fp8_enabled():
            return torch.nn.functional.linear(
                inp,
                weight_tensor,
                bias_tensor,
            )

        with self.prepare_forward(inp, is_first_microbatch, num_gemms=1) as (inp, is_scale_update_required):
            # Fetch the fp8 weight placeholder (for linear/gemm)
            weight1_fp8_fwd, weight1_fp8_bwd = self.get_fp8_weights_scratchpad(is_first_microbatch)

            out = _Linear.apply(
                weight_tensor,
                weight1_fp8_fwd,
                weight1_fp8_bwd,
                inp,
                bias_tensor,
                self.apply_bias and not self.gemm_bias_unfused_add,
                is_first_microbatch,
                self.fp8,
                self.fp8_meta,
                self.tp_group,
                self.sequence_parallel,
                self.tp_size > 1,
                self.activation_dtype,
                self.parallel_mode,
                self.minimize_memory,
                self.get_amax_measure_state(),
                is_scale_update_required,
            )

        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(bias_tensor, self.activation_dtype)

        if self.return_bias:
            return out, cast_if_needed(bias_tensor, self.activation_dtype)
        return out
