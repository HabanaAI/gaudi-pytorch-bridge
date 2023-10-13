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
# - Added MatMul layer
# - Added SelfAttention layer
# - Removed unused code paths

"""Top level Transformer Engine PyTorch modules"""
import os
import warnings
from abc import ABC, abstractmethod
from typing import Union, Optional, Callable, Tuple, Dict, List, Any, Mapping
from functools import partial
from contextlib import contextmanager

import torch
from torch.nn.parameter import Parameter
from torch.nn import init

from habana_frameworks.torch import _hpex_C as tex
from .fp8 import (
    is_fp8_enabled,
    get_fp8_recipe,
    get_fp8_group,
    get_default_fp8_recipe,
    get_fp8_te_dtype,
    get_fp8_te_sr,
    is_first_fp8_module,
    set_fp8_context_id,
    get_fp8_context_id,
    get_run_id_key,
    add_amax_to_global_buffer,
    copy_amax_from_global_buffer,
    global_amax_reduction,
    amax_and_scale_update,
    get_manual_measurement_mode,
    get_global_fp8_buffer,
    set_global_fp8_buffer,
    set_amax_buffer_key_deletion,
    delete_key_from_amax_buffer,
    copy_forward_fp8_meta_tensors_for_recompute,
    get_old_fp8_meta_tensors_for_recompute,
    restore_fp8_meta_tensors,
)
from .utils import (
    divide,
    get_default_init_method,
    cast_if_needed,
)
from .distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
    allreduce,
    initialize_affine_weight_gpu,
    reduce_scatter_along_first_dim,
    gather_along_first_dim,
    gather_along_last_dim,
    is_fp8_activation_recompute_enabled,
    in_fp8_activation_recompute_phase,
)
from .cpp_extensions import (
    fp8_gemm,
    cast_to_fp8,
)
from .constants import GemmParallelModes, dist_group_type

@contextmanager
def _prepare_backward(fp8: bool,
                      fp8_meta: Dict[str, Any],
                      amax_measure_state: dict,
                      is_scale_update_required: bool,
                      reduce_amax_across_tp_group: bool,
                      tp_group: Union[dist_group_type, None]):
    """Checks and prep for BWD."""
    if fp8:
        if fp8_meta["update_amax_bwd"].get("enabled", False):
            # Update amax and scale; Skip all setup for global amax reduction
            if not fp8_meta["recipe"].reduce_amax:
                amax_and_scale_update(fp8_meta, False, is_scale_update_required)
            else:
                # From previous iteration
                copy_amax_from_global_buffer(fp8_meta, forward=False)
                amax_and_scale_update(fp8_meta, False, is_scale_update_required)
                if fp8_meta["first_module"]:
                    set_amax_buffer_key_deletion(fp8_meta, forward=False)

        if amax_measure_state["enabled"] and fp8_meta["recipe"].reduce_amax:
            # Get new backward key.
            if amax_measure_state["enabled"] and fp8_meta["recipe"].reduce_amax:
                fp8_meta[get_run_id_key(forward=False)] = fp8_meta["run_id_fwd_stack"].pop(0)

        fp8_meta["update_amax_bwd"] = amax_measure_state

    yield

    """Checks and prep for BWD."""
    if not fp8 or not fp8_meta["recipe"].reduce_amax:
        return

    if amax_measure_state["enabled"]:
        add_amax_to_global_buffer(fp8_meta, forward=False)
        if fp8_meta["first_module"]:
            global_amax_reduction(fp8_meta, reduce_amax_across_tp_group, tp_group, forward=False)
    if fp8_meta["first_module"]:
        delete_key_from_amax_buffer(forward=False)

class TransformerEngineBaseModule(torch.nn.Module, ABC):
    """Base TE module."""

    def __init__(self) -> None:
        super().__init__()
        self.fp8 = False
        self.fp8_meta = {}
        self.fp8_meta["fp8_group"] = None
        self.fp8_meta["recipe"] = get_default_fp8_recipe()
        self.fp8_meta_tensors_initialized = False
        self.tp_group = None
        self.tp_group_initialized = False
        self.tp_size = 1
        self.sequence_parallel = False
        self.fp8_weight_shapes = []
        self.run_cnt = 0
        self.fp8_meta["run_id_fwd_stack"] = []

    def set_meta_tensor(self, fwd: bool) -> None:
        """Init scales and amaxes for fwd | bwd."""
        fp8_meta_tensor_key = "scaling_fwd" if fwd else "scaling_bwd"
        num_fp8_tensors = (
            self.fp8_meta["num_gemms"] * 2 if fwd else self.fp8_meta["num_gemms"]
        )

        self.fp8_meta[fp8_meta_tensor_key] = tex.FP8TensorMeta()
        self.fp8_meta[fp8_meta_tensor_key].scale = torch.ones(
            num_fp8_tensors, dtype=torch.float32, device="hpu"
        )
        self.fp8_meta[fp8_meta_tensor_key].scale_inv = torch.ones(
            num_fp8_tensors, dtype=torch.float32, device="hpu"
        )
        self.fp8_meta[fp8_meta_tensor_key].amax_history = torch.zeros(
            self.fp8_meta["recipe"].amax_history_len,
            num_fp8_tensors,
            dtype=torch.float32,
            device="hpu",
        )
        self.fp8_meta[fp8_meta_tensor_key].amax_history_index = torch.tensor(
            [0], dtype=torch.int32, device="hpu")

    def init_fp8_meta_tensors(self) -> None:
        """Init scales and amaxes."""
        # Checkpoint loaded
        if self.fp8_meta_tensors_initialized:
            return

        self.set_meta_tensor(True)
        self.set_meta_tensor(False)

    def get_extra_state(self) -> Union[List[Any], None]:
        """Save before checkpointing."""
        if self.fp8:
            state = {}
            state["scale_fwd"] = self.fp8_meta["scaling_fwd"].scale
            state["amax_history_fwd"] = self.fp8_meta["scaling_fwd"].amax_history
            state["scale_bwd"] = self.fp8_meta["scaling_bwd"].scale
            state["amax_history_bwd"] = self.fp8_meta["scaling_bwd"].amax_history
            state["global_fp8_buffer"] = get_global_fp8_buffer()

            # Store other pickelable values.
            extra = {}
            for k, v in self.fp8_meta.items():
                if isinstance(v, (bool, int, float, str)):
                    extra[k] = v
            state["extra_fp8_variables"] = extra

            return state
        return None

    def set_extra_state(self, state: Union[List[Any], None]) -> None:
        """Load previous state."""
        if state is None:
            return

        # Maintain backward compatibility with v0.2.0 and older.
        if isinstance(state, list):
            warnings.warn(
                "This checkpoint format is deprecated and will be"
                "removed in a future release of Transformer Engine"
            )

            # Retrieve checkpointed items.
            scale_fwd = state[0]
            amax_history_fwd = state[1]
            scale_bwd = state[2]
            amax_history_bwd = state[3]
            self.fp8_meta["recipe"].amax_history_len = amax_history_fwd.shape[0]
            self.fp8_meta["num_gemms"] = (
                amax_history_fwd.shape[1] // 2
            )  # Two FWD tensors per GEMM

            # Initialize before loading
            self.init_fp8_meta_tensors()
            self.fp8_meta["scaling_fwd"].scale.copy_(scale_fwd)
            self.fp8_meta["scaling_fwd"].amax_history.copy_(amax_history_fwd)
            self.fp8_meta["scaling_bwd"].scale.copy_(scale_bwd)
            self.fp8_meta["scaling_bwd"].amax_history.copy_(amax_history_bwd)
            self.fp8_meta_tensors_initialized = True

            # Restore global FP8 buffer state.
            set_global_fp8_buffer(state[4])
            self.fp8_meta["update_amax_fwd"] = state[5]
            self.fp8_meta["global_fp8_buffer_pos_fwd"] = state[6]
            self.fp8_meta["global_fp8_buffer_pos_bwd"] = state[7]
            self.fp8_meta[get_run_id_key(forward=True)] = state[8]
            self.fp8_meta[get_run_id_key(forward=False)] = state[9]
            return

        # Restore global FP8 buffer states.
        set_global_fp8_buffer(state["global_fp8_buffer"])
        # Load extra items.
        self.fp8_meta.update(state["extra_fp8_variables"])
        self.fp8_meta["recipe"].amax_history_len = state["amax_history_fwd"].shape[0]
        if "global_fp8_buffer_pos_fwd_recompute" in self.fp8_meta:
            del self.fp8_meta["global_fp8_buffer_pos_fwd_recompute"]

        # Initialize before loading.
        self.init_fp8_meta_tensors()
        self.fp8_meta["scaling_fwd"].scale.copy_(state["scale_fwd"])
        self.fp8_meta["scaling_fwd"].amax_history.copy_(state["amax_history_fwd"])
        self.fp8_meta["scaling_bwd"].scale.copy_(state["scale_bwd"])
        self.fp8_meta["scaling_bwd"].amax_history.copy_(state["amax_history_bwd"])
        self.fp8_meta_tensors_initialized = True

    def set_activation_dtype(self, inp: torch.Tensor) -> None:
        """Get activation data type for AMP."""
        # Native AMP (`torch.autocast`) gets highest priority
        if torch.hpu.is_autocast_hpu_enabled():
            self.activation_dtype = torch.hpu.get_autocast_hpu_dtype()
            return

        # All checks after this have already been performed once, thus skip
        # We assume that user doesn't change input types across iterations
        if hasattr(self, "activation_dtype"):
            return

        assert all(
            (
                (inp.dtype == param.dtype) if param is not None else True
                for param in self.parameters()
            )
        ), (
            "Data type for activations and weights must "
            "match when outside of autocasted region"
        )
        assert all(
            (
                (inp.dtype == buf.dtype) if buf is not None else True
                for buf in self.buffers()
            )
        ), (
            "Data type for activations and buffers must "
            "match when outside of autocasted region"
        )
        self.activation_dtype = inp.dtype

    def _create_fp8_tensor(self, shape) -> torch.Tensor:
        fp8_dtype = get_fp8_te_dtype(
            self.fp8_meta["recipe"], fprop_tensor=True
        )
        result = torch.zeros(
            shape,
            device="hpu",
            dtype=fp8_dtype,
        )

        return result

    def set_fp8_weights(self) -> None:
        """Initializes FP8 weights for the module as class attributes. These
        are not parameters or buffers since we do not want functions such as
        `.to(dtype)` or `.to(device)` to effect them. These also do not need
        to be checkpointed. During `init` phase of the module, the attribute
        `fp8_weight_shapes` must be populated with the tensor shapes for FP8
        weights. This function will iterate over those shapes and initialize
        respective attributed named `weight1_fp8`, `weight2_fp8`, ...
        """
        for i, shape in enumerate(self.fp8_weight_shapes, start=1):
            weight_cast_attr = f"weight{i}_fp8"

            if (
                hasattr(self, weight_cast_attr)
                and getattr(self, weight_cast_attr).shape == shape
            ):
                return

            setattr(
                self,
                weight_cast_attr,
                self._create_fp8_tensor(shape),
            )

    def set_tensor_parallel_group(self, tp_group: Union[dist_group_type, None]) -> None:
        """Set TP group."""
        self.tp_group = tp_group
        self.tp_group_initialized = True

    def fp8_init(self, num_gemms: int = 1) -> None:
        """Initialize fp8 related metadata and tensors during fprop."""
        # If fp8 isn't enabled, turn off and return.
        if not is_fp8_enabled():
            self.fp8 = False
            return

        # FP8 is already enabled and recipe is the same, don't do anything.
        if self.fp8 and get_fp8_recipe() == self.fp8_meta["recipe"]:
            return

        # Set FP8, recipe, and other FP8 metadata
        self.fp8 = True
        self.fp8_meta["recipe"] = get_fp8_recipe()
        self.fp8_meta["num_gemms"] = num_gemms
        self.fp8_meta["fp8_group"] = get_fp8_group()

        # Set FP8_MAX per tensor according to recipe
        self.fp8_meta["fp8_max_fwd"] = self.fp8_meta["recipe"].fp8_format.value.max_fwd
        self.fp8_meta["fp8_max_bwd"] = self.fp8_meta["recipe"].fp8_format.value.max_bwd

        # Allocate scales and amaxes
        self.init_fp8_meta_tensors()

        # Init amax measurement meta state
        self.fp8_meta["update_amax_fwd"] = {}
        self.fp8_meta["update_amax_bwd"] = {}

    def get_amax_measure_state(self) -> dict:
        res = {}
        res['manual'] = get_manual_measurement_mode() is not None
        if get_manual_measurement_mode() is not None:
            res['enabled'] = get_manual_measurement_mode()
        else:
            res['enabled'] = self.fp8_meta["recipe"].interval == 1 or \
            (self.run_cnt + self.fp8_meta["recipe"].interval - 2) % self.fp8_meta["recipe"].interval in \
            range(self.fp8_meta["recipe"].interval - self.fp8_meta["recipe"].amax_history_len, self.fp8_meta["recipe"].interval)
        return res

    def is_scale_update_required(self) -> bool:
        if not self.fp8:
            return False
        manual = self.fp8_meta["update_amax_fwd"].get("manual", False)
        enabled = self.fp8_meta["update_amax_fwd"].get("enabled", False)
        if manual:
            return enabled
        else:
            return (self.fp8_meta["recipe"].interval == 1 or
            (self.run_cnt + self.fp8_meta["recipe"].interval - 2) % self.fp8_meta["recipe"].interval == 0)

    @contextmanager
    def prepare_forward(self, inp: torch.Tensor, num_gemms: int = 1)  -> (None, bool):
        """Checks and prep for FWD.
        The context manager is needed because there isn't a way for a module to know
        if it's the last FP8 module in the forward autocast. It is useful
        to setup the forward aggregated amax reduction for every module
        just in case. The autocast exit will pick up the most recent one.
        """
        self.run_cnt+=1

        # Activation recomputation is used and this is the second forward phase.
        if self.fp8 and in_fp8_activation_recompute_phase():
            get_old_fp8_meta_tensors_for_recompute(self.fp8_meta)
            is_scale_update_required = False
        else:
            if self.tp_size > 1:
                assert self.tp_group_initialized, "TP group not initialized."

            self.set_activation_dtype(inp)
            self.fp8_init(num_gemms=num_gemms)

            is_scale_update_required = self.is_scale_update_required()

            if not "first_module" in self.fp8_meta:
                self.fp8_meta["first_module"] = is_first_fp8_module()
            if self.fp8_meta["first_module"]:
                delete_key_from_amax_buffer(forward=True)

            # Previous iteration was grad_enabled
            if self.fp8_meta["update_amax_fwd"].get("enabled", False):
                if self.fp8_meta["recipe"].reduce_amax:
                    if self.fp8_meta["first_module"]:
                        global_amax_reduction(
                            self.fp8_meta, self.sequence_parallel, self.tp_group, forward=True
                        )
                    copy_amax_from_global_buffer(self.fp8_meta, forward=True)
                    amax_and_scale_update(self.fp8_meta, True, is_scale_update_required)
                    if self.fp8_meta["first_module"]:
                        set_amax_buffer_key_deletion(self.fp8_meta, forward=True)
                else:
                    amax_and_scale_update(self.fp8_meta, True, is_scale_update_required)

            if self.fp8 and self.training:
                # Setup for amax reduction
                if self.get_amax_measure_state()["enabled"] and self.fp8_meta["recipe"].reduce_amax:
                    run_id_key = get_run_id_key(forward=True)
                    if self.fp8_meta["first_module"]:
                        self.fp8_meta[run_id_key] = self.run_cnt
                        set_fp8_context_id(self.fp8_meta[run_id_key])
                    else:
                        self.fp8_meta[run_id_key] = get_fp8_context_id()
                    self.fp8_meta["run_id_fwd_stack"].append(
                        self.fp8_meta[run_id_key]
                    )
                self.fp8_meta["update_amax_fwd"] = self.get_amax_measure_state()
            else:
                self.fp8_meta["update_amax_fwd"] = False

            # Activation recomputation is used and this is the first forward phase.
            if (
                self.fp8
                and self.training
                and is_fp8_activation_recompute_enabled()
                and not in_fp8_activation_recompute_phase()
            ):
                copy_forward_fp8_meta_tensors_for_recompute(self.fp8_meta)

        yield inp.contiguous(), is_scale_update_required

        if self.fp8 and in_fp8_activation_recompute_phase():
            restore_fp8_meta_tensors(self.fp8_meta)
            return

        if self.fp8 and self.training and self.fp8_meta["recipe"].reduce_amax and self.fp8_meta["update_amax_fwd"]["enabled"]:
            add_amax_to_global_buffer(self.fp8_meta, forward=True)

    def set_nccl_overlap_warning_if_tp(self) -> None:
        """When using TP, the NCCL communication needs to be scheduled
        before the GEMM for there to be a guaranteed overlap. From the
        host side in TE, the comm calls are always launched first, but
        to ensure that the GEMM isn't scheduled first, the environment
        variable `CUDA_DEVICE_MAX_CONNECTIONS` needs to be set to 1 to
        force a single channel.
        """
        if self.tp_size == 1:
            return

    @staticmethod
    def grad_output_preprocess(
        ctx,
        grad_output: torch.Tensor,
        row_parallel_mode: bool,
        amax_measure_state: dict,
        grad_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors] = tex.FP8BwdTensors.GRAD_OUTPUT1,
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """Utility function for backward.
        Returns tuple in order (all optional/None based on training precion/recipe):
            R1: gathered `grad_output` in higher precision.
            R2: gathered `grad_output` in FP8.
            R3: bias gradient on R1.

        """
        grad_output = grad_output.contiguous()
        grad_output_mat = grad_output.view((-1, grad_output.shape[-1]))
        gather_grad_output = row_parallel_mode and ctx.sequence_parallel

        # No-FP8 case: bgrad is fused with wgrad for this case.
        if not ctx.fp8:
            if gather_grad_output:
                grad_output_mat, _ = gather_along_first_dim(
                    grad_output_mat, ctx.tp_group
                )
            return grad_output_mat, None, None

        fp8_dtype_backward = get_fp8_te_dtype(
            ctx.fp8_meta["recipe"], fprop_tensor=False
        )

        # FP8 case with non-FP8 wgrad
        if (
            gather_grad_output
            and ctx.fp8_meta["recipe"].override_linear_precision.wgrad
        ):
            grad_output_mat, _ = gather_along_first_dim(grad_output_mat, ctx.tp_group)
        # FP8 case with gather: unfused bgrad, cast, transpose for efficient gather
        elif gather_grad_output:
            if ctx.use_bias:
                grad_bias = grad_output_mat.sum(dim=0)
            else:
                grad_bias = None
            grad_output_c = cast_to_fp8(
                grad_output_mat,
                ctx.fp8_meta["scaling_bwd"],
                grad_tensor,
                fp8_dtype_backward,
                stochastic_rounding=get_fp8_te_sr(ctx.fp8_meta["recipe"], fprop_tensor=False),
                measure_amax=amax_measure_state["enabled"]
            )
            grad_output_c, _ = gather_along_first_dim(grad_output_c, ctx.tp_group)

            return grad_output_mat, grad_output_c, grad_bias

        # FP8 case without gather: unfused cast, transpose, bgrad
        if ctx.use_bias:
            grad_bias = grad_output_mat.sum(dim=0)
        else:
            grad_bias = None
        grad_output_c = cast_to_fp8(
            grad_output_mat,
            ctx.fp8_meta["scaling_bwd"],
            grad_tensor,
            fp8_dtype_backward,
            stochastic_rounding=get_fp8_te_sr(ctx.fp8_meta["recipe"], fprop_tensor=False),
            measure_amax=amax_measure_state["enabled"]
        )

        return grad_output_mat, grad_output_c, grad_bias

    def save_fp8_meta(self):
        scale_fwd = self.fp8_meta["scaling_fwd"].scale.clone()
        scale_inv_fwd = self.fp8_meta["scaling_fwd"].scale_inv.clone()
        amax_history_fwd = self.fp8_meta["scaling_fwd"].amax_history.clone()
        amax_history_index_fwd = self.fp8_meta["scaling_fwd"].amax_history_index.clone()
        scale_bwd = self.fp8_meta["scaling_bwd"].scale.clone()
        scale_inv_bwd = self.fp8_meta["scaling_bwd"].scale_inv.clone()
        amax_history_bwd = self.fp8_meta["scaling_bwd"].amax_history.clone()
        amax_history_index_bwd = self.fp8_meta["scaling_bwd"].amax_history_index.clone()

        return scale_fwd, scale_inv_fwd, amax_history_fwd, amax_history_index_fwd, scale_bwd, scale_inv_bwd, amax_history_bwd, amax_history_index_bwd

    def load_fp8_meta(self, fp8_meta):
        scale_fwd, scale_inv_fwd, amax_history_fwd, amax_history_index_fwd, scale_bwd, scale_inv_bwd, amax_history_bwd, amax_history_index_bwd = fp8_meta

        self.fp8_meta["scaling_fwd"].scale.copy_(scale_fwd)
        self.fp8_meta["scaling_fwd"].scale_inv.copy_(scale_inv_fwd)
        self.fp8_meta["scaling_fwd"].amax_history.copy_(amax_history_fwd)
        self.fp8_meta["scaling_fwd"].amax_history_index.copy_(amax_history_index_fwd)
        self.fp8_meta["scaling_bwd"].scale.copy_(scale_bwd)
        self.fp8_meta["scaling_bwd"].scale_inv.copy_(scale_inv_bwd)
        self.fp8_meta["scaling_bwd"].amax_history.copy_(amax_history_bwd)
        self.fp8_meta["scaling_bwd"].amax_history_index.copy_(amax_history_index_bwd)


    @abstractmethod
    def forward(self):
        """Needs override."""


class _Linear(torch.autograd.Function):
    """Linear semi-top level module
    Calls custom hpu extensions.
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        weight_fp8: torch.Tensor,
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

        inputmat = cast_to_fp8(
            inputmat,
            fp8_meta["scaling_fwd"],
            tex.FP8FwdTensors.GEMM1_INPUT,
            fp8_dtype_forward,
            stochastic_rounding=get_fp8_te_sr(fp8_meta["recipe"], fprop_tensor=True),
            measure_amax=amax_measure_state["enabled"]
        )

        # TODO: Column Parallel Linear

        bias_dtype = (
            torch.bfloat16
            if activation_dtype == torch.float32
            else activation_dtype
        )
        bias = cast_if_needed(bias, bias_dtype) if use_bias else bias

        if update_fp8_weights:
            casted = cast_to_fp8(
                weight,
                fp8_meta["scaling_fwd"],
                tex.FP8FwdTensors.GEMM1_WEIGHT,
                fp8_dtype_forward,
                stochastic_rounding=get_fp8_te_sr(fp8_meta["recipe"], fprop_tensor=True),
                measure_amax=amax_measure_state["enabled"]
            )
            if weight_fp8 is None:
                weight_fp8 = casted
            else:
                assert weight.shape == weight_fp8.shape, "Module initialized with different shape than received weight"
                weight_fp8.copy_(casted)
        out = fp8_gemm(
            weight_fp8,
            fp8_meta["scaling_fwd"].scale_inv[tex.FP8FwdTensors.GEMM1_WEIGHT],
            inputmat,
            fp8_meta["scaling_fwd"].scale_inv[tex.FP8FwdTensors.GEMM1_INPUT],
            activation_dtype,
            bias=bias,
            use_bias=use_bias,
        )

        # NOTE: In case is_first_microbatch is not None, weight_fp8 is stored in the module and is shared
        # between all fwds and bwds. As a result, fp8 weight cannot be cached for backward in the first microbatch,
        # because next fwd will override its value (input bf16 weight will be the same but scale will be different,
        # so the resulting fp8 weight will differ), so in the first microbatch bwd fp8 weight and scale_inv
        # would not match - and the calculated dgrad will be incorrect.
        cache_weight_fp8 = fp8 \
          and not fp8_meta["recipe"].override_linear_precision.dgrad \
          and not minimize_memory \
          and not is_first_microbatch

        ctx.save_for_backward(
            inputmat_no_fp8 if not fp8 or fp8_meta["recipe"].override_linear_precision.wgrad else None,
            inputmat if fp8 and not fp8_meta["recipe"].override_linear_precision.wgrad else None,
            weight_fp8 if cache_weight_fp8 else None,
            weight,
            fp8_meta["scaling_fwd"].scale_inv.clone() if fp8 else None,
            fp8_meta["scaling_fwd"].scale.clone() if fp8 else None,
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

        # Row Parallel Linear
        if parallel_mode == "row" and sequence_parallel:
            out, _ = reduce_scatter_along_first_dim(out, tp_group)
        elif parallel_mode == "row" and tensor_parallel:
            out, _ = allreduce(out, tp_group)

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        return out.view(-1, *inp.shape[1:-1], out.shape[-1])

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        with _prepare_backward(
            ctx.fp8,
            ctx.fp8_meta,
            ctx.amax_measure_state,
            ctx.is_scale_update_required,
            ctx.sequence_parallel,
            ctx.tp_group
        ):
            (
                inputmat,
                inputmap_fp8,
                weight_fp8,
                weight,
                fwd_scale_inverses,
                fwd_scales,
            ) = ctx.saved_tensors

            (
                grad_output,
                grad_output_c,
                grad_bias,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx, grad_output, ctx.parallel_mode == "row", ctx.amax_measure_state
            )

            # Column Parallel Linear
            # Overlap input AG with dgrad
            if ctx.parallel_mode == "column" and ctx.sequence_parallel:
                if ctx.fp8 and not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                    inputmat_fp8_total, handle = gather_along_last_dim(
                        inputmap_fp8, ctx.tp_group, async_op=True
                    )
                else:
                    inputmat_total, handle = gather_along_first_dim(
                        inputmat, ctx.tp_group, async_op=True
                    )
            else:
                inputmat_fp8_total = inputmap_fp8
                inputmat_total = inputmat

            assert ctx.fp8
            fp8_dtype_forward = get_fp8_te_dtype(
                ctx.fp8_meta["recipe"], fprop_tensor=True
            )
            fp8_dtype_backward = get_fp8_te_dtype(
                ctx.fp8_meta["recipe"], fprop_tensor=False
            )

            if weight_fp8 is None:
                # If weight_fp8 was not remembered from fwd pass, recompute it
                weight_fp8, _ = torch.ops.hpu.cast_to_fp8_v2(
                    weight,
                    fwd_scales[tex.FP8FwdTensors.GEMM1_WEIGHT],
                    stochastic_rounding=get_fp8_te_sr(ctx.fp8_meta["recipe"], fprop_tensor=True),
                    is_amax=False,
                    dtype=fp8_dtype_forward,
                )

            # DGRAD
            dgrad = fp8_gemm(
                weight_fp8,
                fwd_scale_inverses[tex.FP8FwdTensors.GEMM1_WEIGHT],
                grad_output_c,
                ctx.fp8_meta["scaling_bwd"].scale_inv[tex.FP8BwdTensors.GRAD_OUTPUT1],
                ctx.activation_dtype,
                transa=False,
            )

            # Overlap dgrad-RS/AR with wgrad
            if ctx.parallel_mode == "column" and ctx.sequence_parallel:
                handle.wait()
                dgrad, handle = reduce_scatter_along_first_dim(
                    dgrad, ctx.tp_group, async_op=True
                )
            elif ctx.parallel_mode == "column" and ctx.tensor_parallel:
                dgrad, handle = allreduce(dgrad, ctx.tp_group, async_op=True)

            if weight.requires_grad:
                # WGRAD
                assert not ctx.fp8_meta["recipe"].override_linear_precision.wgrad
                wgrad = fp8_gemm(
                    inputmat_fp8_total,
                    fwd_scale_inverses[tex.FP8FwdTensors.GEMM1_INPUT],
                    grad_output_c,
                    ctx.fp8_meta["scaling_bwd"].scale_inv[
                        tex.FP8BwdTensors.GRAD_OUTPUT1
                    ],
                    ctx.activation_dtype,
                    accumulate=False,
                    fp32_output=False,
                    out=None,
                    transa=False,
                    transb=True
                )

            # Column Parallel Linear
            if ctx.parallel_mode == "column" and ctx.tensor_parallel and handle is not None:
                handle.wait()

            if not ctx.use_bias:
                grad_bias = None

        return (
            wgrad if weight.requires_grad else None,
            None,
            dgrad.view(ctx.inp_shape),
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

    On NVIDIA GPUs it is a drop-in replacement for `torch.nn.Linear`.

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
    params_dtype : torch.dtype, default = `torch.float32`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
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
        params_dtype: torch.dtype = torch.float32,
        parallel_mode: Optional[str] = None,
        skip_weight_param_allocation: bool = False,
        minimize_memory: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.return_bias = return_bias
        self.skip_weight_param_allocation = skip_weight_param_allocation
        self.minimize_memory = minimize_memory

        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)
        self.set_nccl_overlap_warning_if_tp()

        self.parallel_mode = parallel_mode
        assert (
            self.parallel_mode in GemmParallelModes
        ), f"parallel_mode {parallel_mode} not supported"

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
                    device="hpu",
                    dtype=params_dtype,
                )
            )

            initialize_affine_weight_gpu(
                self.weight,
                init_method,
                get_rng_state_tracker,
                partition_dim=1 if self.parallel_mode == "row" else 0,
                stride=1,
            )

            if self.use_bias or self.return_bias:
                self.bias = Parameter(
                    torch.empty(
                        self.out_features,
                        device="hpu",
                        dtype=params_dtype,
                    )
                )
                if self.parallel_mode == "column":
                    set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
            else:
                self.register_buffer("bias", torch.Tensor(), persistent=False)

            with torch.no_grad():
                self.bias.zero_()

        self.fp8_weight_shapes.append(torch.Size((self.out_features, self.in_features)))

        # For RPL, bias has to be added after TP collectives
        # So it cannot be fused with the GEMM
        if self.parallel_mode == "row" and self.use_bias:
            self.gemm_bias_unfused_add = True
            self.use_bias = False
        else:
            self.gemm_bias_unfused_add = False

        # To initialize weights stored in fp8. Notice that original implementation calls it every fwd,
        # but we call it once to reduce host overhead
        self.set_fp8_weights()

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

        with self.prepare_forward(inp) as (inp, is_scale_update_required):
            out = _Linear.apply(
                weight_tensor,
                self.weight1_fp8 if is_first_microbatch is not None else None,
                inp,
                bias_tensor,
                self.use_bias,
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


class _LayerNorm(torch.autograd.Function):
    """functional LayerNorm"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        # Make sure input dimensions are compatible
        in_features = ln_weight.numel()
        assert inp.shape[-1] == in_features, "LayerNorm not possible"
        inputmat = inp.view((-1, in_features))

        ln_out, mu, rsigma = tex.layernorm_fwd(inputmat, ln_weight, ln_bias, eps)
        ctx.save_for_backward(inputmat, ln_weight, mu, rsigma)
        ctx.inp_shape = inp.shape
        return ln_out.view_as(inp)

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        inputmat, ln_weight, mu, rsigma = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        d_ln_out = grad_output.view(inputmat.shape)
        dxmat, dgamma, dbeta = tex.layernorm_bwd(
            d_ln_out, inputmat, mu, rsigma, ln_weight
        )
        return dxmat.view(ctx.inp_shape), dgamma, dbeta, None


class LayerNorm(torch.nn.Module):
    r"""
    Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    size :attr:`hidden_size`

    Parameters
    ----------
    hidden_size : int
                size of each input sample.
    eps : float, default = 1e-5
        a value added to the denominator of layer normalization for numerical stability.
    sequence_parallel : bool, default = `False`
                        if set to `True`, uses sequence parallelism.
    params_dtype : torch.dtype, default = `torch.float32`
                    it controls the type used to allocate the initial parameters. Useful when
                    the model is trained with lower precision and the original FP32 parameters
                    would not fit in GPU memory.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        params_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = Parameter(
            torch.empty(
                hidden_size,
                device="hpu",
                dtype=params_dtype,
            )
        )
        self.bias = Parameter(
            torch.empty(
                hidden_size,
                device="hpu",
                dtype=params_dtype,
            )
        )
        setattr(self.weight, "sequence_parallel", sequence_parallel)
        setattr(self.bias, "sequence_parallel", sequence_parallel)
        self.reset_layer_norm_parameters()

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
    ) -> None:
        """Override PyTorch loader to maintain backward compatibility
        with previous version of LayerNorm parameter names.
        """
        if "layer_norm_weight" in state_dict:
            state_dict["weight"] = state_dict["layer_norm_weight"]
            del state_dict["layer_norm_weight"]
        if "layer_norm_bias" in state_dict:
            state_dict["bias"] = state_dict["layer_norm_bias"]
            del state_dict["layer_norm_bias"]

        super().load_state_dict(state_dict, strict)

    def reset_layer_norm_parameters(self) -> None:
        """Init LN params"""
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """LayerNorm FWD"""
        # Maintain backward compatibility.
        if hasattr(self, "layer_norm_weight"):
            setattr(self, "weight", self.layer_norm_weight)
        if hasattr(self, "layer_norm_bias"):
            setattr(self, "bias", self.layer_norm_bias)

        return _LayerNorm.apply(inp, self.weight, self.bias, self.eps)


class _MatMul(torch.autograd.Function):
    """MatMul semi-top level module
    Calls custom hpu extensions.
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        inp: torch.Tensor,
        transpose_weight: bool,
        transpose_inp: bool,
        fp8: bool,
        fp8_meta: Dict[str, Any],
        tp_group: Union[dist_group_type, None],
        sequence_parallel: bool,
        tensor_parallel: bool,
        activation_dtype: torch.dtype,
        parallel_mode: Union[str, None],
        amax_measure_state: dict,
        is_scale_update_required: bool,
    ) -> torch.Tensor:
        # Make sure input dimensions are compatible
        dim0 = weight.shape[-2] if not transpose_weight else weight.shape[-1]
        dim1 = inp.shape[-1] if not transpose_inp else inp.shape[-2]
        assert dim0 == dim1, "GEMM not possible"

        # Cast for native AMP
        inputmat = cast_if_needed(inp, activation_dtype)

        assert fp8
        fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)

        assert not fp8_meta["recipe"].override_linear_precision.fprop
        inputmat = cast_to_fp8(
            inputmat,
            fp8_meta["scaling_fwd"],
            tex.FP8FwdTensors.GEMM1_INPUT,
            fp8_dtype_forward,
            stochastic_rounding=get_fp8_te_sr(fp8_meta["recipe"], fprop_tensor=True),
            measure_amax=amax_measure_state["enabled"]
        )

        weight_fp8 = cast_to_fp8(
            weight,
            fp8_meta["scaling_fwd"],
            tex.FP8FwdTensors.GEMM1_WEIGHT,
            fp8_dtype_forward,
            stochastic_rounding=get_fp8_te_sr(fp8_meta["recipe"], fprop_tensor=True),
            measure_amax=amax_measure_state["enabled"]
        )

        out = fp8_gemm(
            weight_fp8,
            fp8_meta["scaling_fwd"].scale_inv[tex.FP8FwdTensors.GEMM1_WEIGHT],
            inputmat,
            fp8_meta["scaling_fwd"].scale_inv[tex.FP8FwdTensors.GEMM1_INPUT],
            activation_dtype,
            transa = transpose_weight,
            transb = transpose_inp
        )

        ctx.save_for_backward(
            inputmat,
            weight,
            weight_fp8,
            fp8_meta["scaling_fwd"].scale_inv.clone() if fp8 else None,
        )
        ctx.activation_dtype = activation_dtype
        ctx.fp8 = fp8
        ctx.fp8_meta = fp8_meta
        ctx.transpose_weight = transpose_weight
        ctx.transpose_inp = transpose_inp
        ctx.sequence_parallel = sequence_parallel
        ctx.tensor_parallel = tensor_parallel
        ctx.inp_shape = inp.shape
        ctx.parallel_mode = parallel_mode
        ctx.tp_group = tp_group
        ctx.use_bias = False
        ctx.amax_measure_state = amax_measure_state.copy()
        ctx.is_scale_update_required = is_scale_update_required

        # Row Parallel Linear
        if parallel_mode == "row" and sequence_parallel:
            out, _ = reduce_scatter_along_first_dim(out, tp_group)
        elif parallel_mode == "row" and tensor_parallel:
            out, _ = allreduce(out, tp_group)

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        return out

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        with _prepare_backward(
            ctx.fp8,
            ctx.fp8_meta,
            ctx.amax_measure_state,
            ctx.is_scale_update_required,
            ctx.sequence_parallel,
            ctx.tp_group
        ):
            (
                inputmat,
                weight,
                weight_fp8,
                fwd_scale_inverses,
            ) = ctx.saved_tensors

            inputmat_total = inputmat

            assert ctx.fp8
            fp8_dtype_forward = get_fp8_te_dtype(
                ctx.fp8_meta["recipe"], fprop_tensor=True
            )
            fp8_dtype_backward = get_fp8_te_dtype(
                ctx.fp8_meta["recipe"], fprop_tensor=False
            )

            grad_output_c = cast_to_fp8(
                grad_output,
                ctx.fp8_meta["scaling_bwd"],
                tex.FP8BwdTensors.GRAD_OUTPUT1,
                fp8_dtype_backward,
                stochastic_rounding=get_fp8_te_sr(ctx.fp8_meta["recipe"], fprop_tensor=False),
                measure_amax=ctx.amax_measure_state["enabled"]
            )

            # DGRAD
            dgrad = fp8_gemm(
                weight_fp8,
                fwd_scale_inverses[tex.FP8FwdTensors.GEMM1_WEIGHT],
                grad_output_c,
                ctx.fp8_meta["scaling_bwd"].scale_inv[tex.FP8BwdTensors.GRAD_OUTPUT1],
                ctx.activation_dtype,
                transa = not ctx.transpose_weight,
                transb = False,
            )
            if ctx.transpose_inp:
                dgrad = dgrad.transpose(-1, -2)

            # Overlap dgrad-RS/AR with wgrad
            if ctx.parallel_mode == "column" and ctx.sequence_parallel:
                handle.wait()
                dgrad, handle = reduce_scatter_along_first_dim(
                    dgrad, ctx.tp_group, async_op=True
                )
            elif ctx.parallel_mode == "column" and ctx.tensor_parallel:
                dgrad, handle = allreduce(dgrad, ctx.tp_group, async_op=True)

            if weight.requires_grad:
                # WGRAD
                assert not ctx.fp8_meta["recipe"].override_linear_precision.wgrad
                wgrad = fp8_gemm(
                    grad_output_c,
                    ctx.fp8_meta["scaling_bwd"].scale_inv[
                        tex.FP8BwdTensors.GRAD_OUTPUT1
                    ],
                    inputmat_total,
                    fwd_scale_inverses[tex.FP8FwdTensors.GEMM1_INPUT],
                    ctx.activation_dtype,
                    transa = False,
                    transb = not ctx.transpose_inp
                )
                if ctx.transpose_weight:
                    wgrad = wgrad.transpose(-1, -2)

            # Column Parallel Linear
            if ctx.parallel_mode == "column" and ctx.tensor_parallel and handle is not None:
                handle.wait()

        return (
            wgrad if weight.requires_grad else None,
            dgrad,
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

class MatMul(TransformerEngineBaseModule):
    """
    Applies a MatMul operation to the incoming data

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

    """

    def __init__(
        self,
        sequence_parallel: bool = False,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        parallel_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)
        self.set_nccl_overlap_warning_if_tp()

        self.parallel_mode = parallel_mode
        assert (
            self.parallel_mode in GemmParallelModes
        ), f"parallel_mode {parallel_mode} not supported"

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel

    def forward(
        self,
        inp: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        transa: bool = False,
        transb: bool = False,
        is_first_microbatch: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply the MatMul transformation to the input.

        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
        weight : torch.Tensor, default = None
                An optional weight tensor for the module. This argument is compulsory if module
                is initialized with `skip_weight_param_allocation=True`
        """

        with self.prepare_forward(inp) as (inp, is_scale_update_required):
            out = _MatMul.apply(
                weight,
                inp,
                transb,
                transa,
                self.fp8,
                self.fp8_meta,
                self.tp_group,
                self.sequence_parallel,
                self.tp_size > 1,
                self.activation_dtype,
                self.parallel_mode,
                self.get_amax_measure_state(),
                is_scale_update_required,
            )

        return out

class _SelfAttentionScoresAndValue(torch.autograd.Function):
    """SelfAttentionScoresAndValue semi-top level module
    Calls custom hpu extensions.
    """

    @staticmethod
    def _transpose_for_scores_fp8(
        x: torch.Tensor,
        num_attention_heads: int,
        attention_head_size: int
    ) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        x = torch.reshape(x, new_x_shape)
        out = torch.permute(x, [0,2,1,3])
        return out

    @staticmethod
    def _grad_transpose_for_scores(x: torch.Tensor) -> torch.Tensor:
        # Permutation opposite to (0,2,1,3)
        x = x.permute(0, 2, 1, 3)
        # Collapse last two dimentions
        return x.reshape(x.size()[:-2] + (-1,))

    @staticmethod
    def _transpose_key_for_scores_fp8(
        x: torch.Tensor,
        num_attention_heads: int,
        attention_head_size: int
    ) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        x = torch.reshape(x, new_x_shape)
        out = torch.permute(x, [0,2,3,1])
        return out

    @staticmethod
    def _grad_transpose_key_for_scores(x: torch.Tensor) -> torch.Tensor:
        # Permutation opposite to (0,2,3,1)
        x = x.permute(0,3,1,2)
        # Collapse last two dimentions
        return x.reshape(x.size()[:-2] + (-1,))

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        query_weight: torch.Tensor,
        query_bias: torch.Tensor,
        key_weight: torch.Tensor,
        key_bias: torch.Tensor,
        value_weight: torch.Tensor,
        value_bias: torch.Tensor,
        num_attention_heads: int,
        attention_head_size: int,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_meta: Dict[str, Any],
        fuse_wgrad_accumulation: bool,
        tp_group: Union[dist_group_type, None],
        sequence_parallel: bool,
        tensor_parallel: bool,
        activation_dtype: torch.dtype,
        amax_measure_state: dict,
        is_scale_update_required: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Make sure input dimensions are compatible
        in_features = query_weight.shape[-1]
        assert inp.shape[-1] == in_features, "GEMM not possible"
        assert key_weight.shape[-1] == in_features, "GEMM not possible"
        assert value_weight.shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.view((-1, in_features))

        update_fp8_weights = is_first_microbatch is None or is_first_microbatch

        # Cast for native AMP
        inputmat = cast_if_needed(inputmat, activation_dtype)

        assert fp8
        fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)

        assert not fp8_meta["recipe"].override_linear_precision.fprop
        # Input cast - common for query, key and value
        inputmat = cast_to_fp8(
            inputmat,
            fp8_meta["scaling_fwd"],
            tex.FP8FwdTensors.GEMM1_INPUT,
            fp8_dtype_forward,
            stochastic_rounding=get_fp8_te_sr(fp8_meta["recipe"], fprop_tensor=True),
            measure_amax=amax_measure_state["enabled"]
        )

        # Query matmul
        bias_dtype = (
            torch.bfloat16
            if activation_dtype == torch.float32
            else activation_dtype
        )
        query_bias = cast_if_needed(query_bias, bias_dtype)

        if update_fp8_weights:
            query_weight_fp8 = cast_to_fp8(
                query_weight,
                fp8_meta["scaling_fwd"],
                tex.FP8FwdTensors.GEMM1_WEIGHT,
                fp8_dtype_forward,
                stochastic_rounding=get_fp8_te_sr(fp8_meta["recipe"], fprop_tensor=True),
                measure_amax=amax_measure_state["enabled"]
            )

        query_out = fp8_gemm(
            query_weight_fp8,
            fp8_meta["scaling_fwd"].scale_inv[tex.FP8FwdTensors.GEMM1_WEIGHT],
            inputmat,
            fp8_meta["scaling_fwd"].scale_inv[tex.FP8FwdTensors.GEMM1_INPUT],
            activation_dtype,
            bias=query_bias,
            use_bias=True,
        )

        # Key matmul
        bias_dtype = (
            torch.bfloat16
            if activation_dtype == torch.float32
            else activation_dtype
        )
        key_bias = cast_if_needed(key_bias, bias_dtype)

        if update_fp8_weights:
            key_weight_fp8 = cast_to_fp8(
                key_weight,
                fp8_meta["scaling_fwd"],
                tex.FP8FwdTensors.GEMM2_WEIGHT,
                fp8_dtype_forward,
                stochastic_rounding=get_fp8_te_sr(fp8_meta["recipe"], fprop_tensor=True),
                measure_amax=amax_measure_state["enabled"]
            )

        key_out = fp8_gemm(
            key_weight_fp8,
            fp8_meta["scaling_fwd"].scale_inv[tex.FP8FwdTensors.GEMM2_WEIGHT],
            inputmat,
            fp8_meta["scaling_fwd"].scale_inv[tex.FP8FwdTensors.GEMM1_INPUT],
            activation_dtype,
            bias=key_bias,
            use_bias=True,
        )

        # Value matmul
        bias_dtype = (
            torch.bfloat16
            if activation_dtype == torch.float32
            else activation_dtype
        )
        value_bias = cast_if_needed(value_bias, bias_dtype)

        if update_fp8_weights:
            value_weight_fp8 = cast_to_fp8(
                value_weight,
                fp8_meta["scaling_fwd"],
                tex.FP8FwdTensors.GEMM3_WEIGHT,
                fp8_dtype_forward,
                stochastic_rounding=get_fp8_te_sr(fp8_meta["recipe"], fprop_tensor=True),
                measure_amax=amax_measure_state["enabled"]
            )

        value_out = fp8_gemm(
            value_weight_fp8,
            fp8_meta["scaling_fwd"].scale_inv[tex.FP8FwdTensors.GEMM3_WEIGHT],
            inputmat,
            fp8_meta["scaling_fwd"].scale_inv[tex.FP8FwdTensors.GEMM1_INPUT],
            activation_dtype,
            bias=value_bias,
            use_bias=True,
        )

        query_out = query_out.view(-1, *inp.shape[1:-1], query_out.shape[-1])
        key_out = key_out.view(-1, *inp.shape[1:-1], key_out.shape[-1])
        value_out = value_out.view(-1, *inp.shape[1:-1], value_out.shape[-1])

        # Take the dot product between "query" and "key" to get the raw attention scores.
        query_layer_fp8 = cast_to_fp8(
            query_out,
            fp8_meta["scaling_fwd"],
            tex.FP8FwdTensors.GEMM4_INPUT,
            fp8_dtype_forward,
            stochastic_rounding=get_fp8_te_sr(fp8_meta["recipe"], fprop_tensor=True),
            measure_amax=amax_measure_state["enabled"]
        )

        key_layer_fp8 = cast_to_fp8(
            key_out,
            fp8_meta["scaling_fwd"],
            tex.FP8FwdTensors.GEMM4_WEIGHT,
            fp8_dtype_forward,
            stochastic_rounding=get_fp8_te_sr(fp8_meta["recipe"], fprop_tensor=True),
            measure_amax=amax_measure_state["enabled"]
        )

        query_layer_fp8 = _SelfAttentionScoresAndValue._transpose_for_scores_fp8(
            query_layer_fp8, num_attention_heads, attention_head_size)
        key_layer_fp8 = _SelfAttentionScoresAndValue._transpose_key_for_scores_fp8(
            key_layer_fp8, num_attention_heads, attention_head_size)

        attention_scores = fp8_gemm(
            key_layer_fp8,
            fp8_meta["scaling_fwd"].scale_inv[tex.FP8FwdTensors.GEMM4_WEIGHT],
            query_layer_fp8,
            fp8_meta["scaling_fwd"].scale_inv[tex.FP8FwdTensors.GEMM4_INPUT],
            activation_dtype,
            transa = False,
            transb = False,
        )

        ctx.save_for_backward(
            inputmat
            if fp8 and not fp8_meta["recipe"].override_linear_precision.wgrad
            else None,
            query_weight_fp8
            if fp8 and not fp8_meta["recipe"].override_linear_precision.dgrad
            else None,
            query_weight,
            key_weight_fp8
            if fp8 and not fp8_meta["recipe"].override_linear_precision.dgrad
            else None,
            key_weight,
            value_weight_fp8
            if fp8 and not fp8_meta["recipe"].override_linear_precision.dgrad
            else None,
            value_weight,
            query_layer_fp8,
            key_layer_fp8,
            fp8_meta["scaling_fwd"].scale_inv.clone() if fp8 else None,
        )
        ctx.activation_dtype = activation_dtype
        ctx.fp8 = fp8
        ctx.fp8_meta = fp8_meta
        ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        ctx.is_first_microbatch = is_first_microbatch
        ctx.sequence_parallel = sequence_parallel
        ctx.tensor_parallel = tensor_parallel
        ctx.inp_shape = inp.shape
        ctx.tp_group = tp_group
        ctx.amax_measure_state = amax_measure_state.copy()
        ctx.is_scale_update_required = is_scale_update_required

        return attention_scores, value_out


    @staticmethod
    def backward(
        ctx,
        attention_scores_grad_output: torch.Tensor,
        value_grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        with _prepare_backward(
            ctx.fp8,
            ctx.fp8_meta,
            ctx.amax_measure_state,
            ctx.is_scale_update_required,
            ctx.sequence_parallel,
            ctx.tp_group
        ):
            (
                inputmat_fp8,
                query_weight_fp8,
                query_weight,
                key_weight_fp8,
                key_weight,
                value_weight_fp8,
                value_weight,
                query_layer_fp8,
                key_layer_fp8,
                fwd_scale_inverses,
            ) = ctx.saved_tensors

            assert ctx.fp8
            fp8_dtype_forward = get_fp8_te_dtype(
                ctx.fp8_meta["recipe"], fprop_tensor=True
            )
            fp8_dtype_backward = get_fp8_te_dtype(
                ctx.fp8_meta["recipe"], fprop_tensor=False
            )

            attention_scores_grad_output_c = cast_to_fp8(
                attention_scores_grad_output,
                ctx.fp8_meta["scaling_bwd"],
                tex.FP8BwdTensors.GRAD_OUTPUT4,
                fp8_dtype_backward,
                stochastic_rounding=get_fp8_te_sr(ctx.fp8_meta["recipe"], fprop_tensor=False),
                measure_amax=ctx.amax_measure_state["enabled"]
            )

            # attention scores DGRAD
            query_layer_grad = fp8_gemm(
                key_layer_fp8,
                fwd_scale_inverses[tex.FP8FwdTensors.GEMM4_WEIGHT],
                attention_scores_grad_output_c,
                ctx.fp8_meta["scaling_bwd"].scale_inv[tex.FP8BwdTensors.GRAD_OUTPUT4],
                ctx.activation_dtype,
                transa = True,
                transb = False
            )

            key_layer_grad = fp8_gemm(
                attention_scores_grad_output_c,
                ctx.fp8_meta["scaling_bwd"].scale_inv[
                    tex.FP8BwdTensors.GRAD_OUTPUT4
                ],
                query_layer_fp8,
                fwd_scale_inverses[tex.FP8FwdTensors.GEMM4_INPUT],
                ctx.activation_dtype,
                transa = False,
                transb = True,
            )

            query_grad_output = _SelfAttentionScoresAndValue._grad_transpose_for_scores(query_layer_grad)
            key_grad_output = _SelfAttentionScoresAndValue._grad_transpose_key_for_scores(key_layer_grad)

            ctx.use_bias = True
            (
                query_grad_output,
                query_grad_output_c,
                query_grad_bias,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx, query_grad_output, row_parallel_mode=False, amax_measure_state=ctx.amax_measure_state,
                grad_tensor=tex.FP8BwdTensors.GRAD_OUTPUT1
            )

            (
                key_grad_output,
                key_grad_output_c,
                key_grad_bias,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx, key_grad_output, row_parallel_mode=False, amax_measure_state=ctx.amax_measure_state,
                grad_tensor=tex.FP8BwdTensors.GRAD_OUTPUT2
            )

            (
                value_grad_output,
                value_grad_output_c,
                value_grad_bias,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx, value_grad_output, row_parallel_mode=False, amax_measure_state=ctx.amax_measure_state,
                grad_tensor=tex.FP8BwdTensors.GRAD_OUTPUT3
            )

            inputmat_fp8_total = inputmat_fp8

            if ctx.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                )
            else:
                accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

            # query DGRAD
            query_dgrad = fp8_gemm(
                query_weight_fp8,
                fwd_scale_inverses[tex.FP8FwdTensors.GEMM1_WEIGHT],
                query_grad_output_c,
                ctx.fp8_meta["scaling_bwd"].scale_inv[tex.FP8BwdTensors.GRAD_OUTPUT1],
                ctx.activation_dtype,
                transa=False,
                transb=False,
            )

            if query_weight.requires_grad:
                # query WGRAD
                assert not ctx.fp8_meta["recipe"].override_linear_precision.wgrad
                query_wgrad = fp8_gemm(
                    inputmat_fp8_total,
                    fwd_scale_inverses[tex.FP8FwdTensors.GEMM1_INPUT],
                    query_grad_output_c,
                    ctx.fp8_meta["scaling_bwd"].scale_inv[
                        tex.FP8BwdTensors.GRAD_OUTPUT1
                    ],
                    ctx.activation_dtype,
                    accumulate=accumulate_wgrad_into_param_main_grad,
                    fp32_output=ctx.fuse_wgrad_accumulation,
                    out=query_weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                    transa=False,
                    transb=True,
                )

            # key DGRAD
            key_dgrad = fp8_gemm(
                key_weight_fp8,
                fwd_scale_inverses[tex.FP8FwdTensors.GEMM2_WEIGHT],
                key_grad_output_c,
                ctx.fp8_meta["scaling_bwd"].scale_inv[tex.FP8BwdTensors.GRAD_OUTPUT2],
                ctx.activation_dtype,
                transa=False,
                transb=False,
            )

            if key_weight.requires_grad:
                # key WGRAD
                assert not ctx.fp8_meta["recipe"].override_linear_precision.wgrad
                key_wgrad = fp8_gemm(
                    inputmat_fp8_total,
                    fwd_scale_inverses[tex.FP8FwdTensors.GEMM1_INPUT],
                    key_grad_output_c,
                    ctx.fp8_meta["scaling_bwd"].scale_inv[
                        tex.FP8BwdTensors.GRAD_OUTPUT2
                    ],
                    ctx.activation_dtype,
                    accumulate=accumulate_wgrad_into_param_main_grad,
                    fp32_output=ctx.fuse_wgrad_accumulation,
                    out=key_weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                    transa=False,
                    transb=True,
                )

            # value DGRAD
            value_dgrad = fp8_gemm(
                value_weight_fp8,
                fwd_scale_inverses[tex.FP8FwdTensors.GEMM3_WEIGHT],
                value_grad_output_c,
                ctx.fp8_meta["scaling_bwd"].scale_inv[tex.FP8BwdTensors.GRAD_OUTPUT3],
                ctx.activation_dtype,
                transa=False,
                transb=False,
            )

            if value_weight.requires_grad:
                # value WGRAD
                assert not ctx.fp8_meta["recipe"].override_linear_precision.wgrad
                value_wgrad = fp8_gemm(
                    inputmat_fp8_total,
                    fwd_scale_inverses[tex.FP8FwdTensors.GEMM1_INPUT],
                    value_grad_output_c,
                    ctx.fp8_meta["scaling_bwd"].scale_inv[
                        tex.FP8BwdTensors.GRAD_OUTPUT3
                    ],
                    ctx.activation_dtype,
                    accumulate=accumulate_wgrad_into_param_main_grad,
                    fp32_output=ctx.fuse_wgrad_accumulation,
                    out=value_weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                    transa=False,
                    transb=True,
                )

        # Gradient with respect to input - sum of the gradients of gemms
        dgrad = value_dgrad + key_dgrad + query_dgrad

        return (
            dgrad.view(ctx.inp_shape),
            query_wgrad if query_weight.requires_grad else None,
            query_grad_bias,
            key_wgrad if key_weight.requires_grad else None,
            key_grad_bias,
            value_wgrad if value_weight.requires_grad else None,
            value_grad_bias,
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
            None,
            None,
        )


class SelfAttentionScoresAndValue(TransformerEngineBaseModule):
    """
    Applies the first part of self attention module as implemented in Bert script.
    It replaces query, key and value linear modules, as well as transposes
    and attention scores calculation.

    Parameters
    ----------
    hidden_size : int
                  The last dimension of input tensor.
    num_attention_heads : int
                  Number of attention heads. hidden_size must be divisible by this number.

    Other parameters
    ----------------
    All other parameters have the same meaning as in standard TE modules (e.g. Linear).
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        skip_weight_param_allocation: bool = False,
        params_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation

        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)
        self.set_nccl_overlap_warning_if_tp()

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel

        # Self attention stuff
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Initialize weights and biases
        init_method = get_default_init_method()

        def initialize_weight_and_bias(init_method):
            weight = Parameter(
                torch.empty(
                    self.all_head_size,
                    hidden_size,
                    device="hpu",
                    dtype=params_dtype,
                )
            )

            initialize_affine_weight_gpu(
                weight,
                init_method,
                get_rng_state_tracker,
                partition_dim=0,
                stride=1,
            )

            bias = Parameter(
                torch.empty(
                    self.all_head_size,
                    device="hpu",
                    dtype=params_dtype,
                )
            )

            with torch.no_grad():
                bias.zero_()

            return weight, bias

        if not skip_weight_param_allocation:
            self.query_weight, self.query_bias = initialize_weight_and_bias(init_method)
            self.key_weight, self.key_bias     = initialize_weight_and_bias(init_method)
            self.value_weight, self.value_bias = initialize_weight_and_bias(init_method)

        self.fp8_weight_shapes.append(torch.Size((self.all_head_size, hidden_size,)))
        self.fp8_weight_shapes.append(torch.Size((self.all_head_size, hidden_size,)))
        self.fp8_weight_shapes.append(torch.Size((self.all_head_size, hidden_size,)))

    def forward(self,
        hidden_states,
        is_first_microbatch: Optional[bool] = None
    ):
        # Four gemms are used inside this layer, which means four pairs of
        # meta info about scaling (amax_history, scale, scale_inv).
        # In forward pass, the indexes are used as follows:
        # - GEMM1_INPUT:  input, common for query, key and value gemms,
        # - GEMM1_WEIGHT: query weight,
        # - GEMM2_INPUT:  not used,
        # - GEMM2_WEIGHT: key weight,
        # - GEMM3_INPUT:  not used,
        # - GEMM3_WEIGHT: value weight,
        # - GEMM4_INPUT:  first input to attention scores gemm (output from query gemm)
        # - GEMM4_WEIGHT: second input to attention scores gemm (output from key gemm)
        # And in backward pass:
        # - GRAD_OUTPUT1: query grad,
        # - GRAD_OUTPUT2: key grad,
        # - GRAD_OUTPUT3: value grad,
        # - GRAD_OUTPUT4: attention scores grad,
        with self.prepare_forward(hidden_states, num_gemms = 4) as (inp, is_scale_update_required):
            attention_scores, value_out = _SelfAttentionScoresAndValue.apply(
                inp,
                self.query_weight,
                self.query_bias,
                self.key_weight,
                self.key_bias,
                self.value_weight,
                self.value_bias,
                self.num_attention_heads,
                self.attention_head_size,
                is_first_microbatch,
                self.fp8,
                self.fp8_meta,
                self.fuse_wgrad_accumulation,
                self.tp_group,
                self.sequence_parallel,
                self.tp_size > 1,
                self.activation_dtype,
                self.get_amax_measure_state(),
                is_scale_update_required,
            )

        return attention_scores, value_out


class _SelfAttentionContext(torch.autograd.Function):
    """SelfAttentionContext semi-top level module
    Calls custom hpu extensions.
    """

    @staticmethod
    def forward(
        ctx,
        attention_probs_input: torch.Tensor,
        mixed_value_layer_input: torch.Tensor,
        num_attention_heads,
        attention_head_size,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_meta: Dict[str, Any],
        fuse_wgrad_accumulation: bool,
        tp_group: Union[dist_group_type, None],
        sequence_parallel: bool,
        tensor_parallel: bool,
        activation_dtype: torch.dtype,
        amax_measure_state: dict,
        is_scale_update_required: bool,
    ) -> torch.Tensor:
        dim0 = attention_probs_input.shape[-1]
        dim1 = mixed_value_layer_input.shape[-2]
        assert dim0 == dim1, "GEMM not possible"
        dim2 = attention_probs_input.shape[-3]
        assert dim2 == num_attention_heads, "GEMM not possible"

        # Cast for native AMP
        attention_probs = cast_if_needed(attention_probs_input, activation_dtype)
        mixed_value_layer = cast_if_needed(mixed_value_layer_input, activation_dtype)

        assert fp8
        fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)

        assert not fp8_meta["recipe"].override_linear_precision.fprop
        attention_probs_fp8 = cast_to_fp8(
            attention_probs,
            fp8_meta["scaling_fwd"],
            tex.FP8FwdTensors.GEMM1_INPUT,
            fp8_dtype_forward,
            stochastic_rounding=get_fp8_te_sr(fp8_meta["recipe"], fprop_tensor=True),
            measure_amax=amax_measure_state["enabled"]
        )

        mixed_value_layer_fp8 = cast_to_fp8(
            mixed_value_layer,
            fp8_meta["scaling_fwd"],
            tex.FP8FwdTensors.GEMM1_WEIGHT,
            fp8_dtype_forward,
            stochastic_rounding=get_fp8_te_sr(fp8_meta["recipe"], fprop_tensor=True),
            measure_amax=amax_measure_state["enabled"]
        )

        value_layer_fp8 = _SelfAttentionScoresAndValue._transpose_for_scores_fp8(
            mixed_value_layer_fp8, num_attention_heads, attention_head_size)

        out = fp8_gemm(
            value_layer_fp8,
            fp8_meta["scaling_fwd"].scale_inv[tex.FP8FwdTensors.GEMM1_WEIGHT],
            attention_probs_fp8,
            fp8_meta["scaling_fwd"].scale_inv[tex.FP8FwdTensors.GEMM1_INPUT],
            activation_dtype,
            transa = False,
            transb = False,
        )

        ctx.save_for_backward(
            attention_probs_fp8,
            value_layer_fp8,
            fp8_meta["scaling_fwd"].scale_inv.clone() if fp8 else None,
        )
        ctx.activation_dtype = activation_dtype
        ctx.fp8 = fp8
        ctx.fp8_meta = fp8_meta
        ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        ctx.is_first_microbatch = is_first_microbatch
        ctx.sequence_parallel = sequence_parallel
        ctx.tensor_parallel = tensor_parallel
        ctx.tp_group = tp_group
        ctx.amax_measure_state = amax_measure_state.copy()
        ctx.is_scale_update_required = is_scale_update_required

        return out


    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        with _prepare_backward(
            ctx.fp8,
            ctx.fp8_meta,
            ctx.amax_measure_state,
            ctx.is_scale_update_required,
            ctx.sequence_parallel,
            ctx.tp_group
        ):
            (
                attention_probs_fp8,
                value_layer_fp8,
                fwd_scale_inverses,
            ) = ctx.saved_tensors

            assert ctx.fp8
            fp8_dtype_forward = get_fp8_te_dtype(
                ctx.fp8_meta["recipe"], fprop_tensor=True
            )
            fp8_dtype_backward = get_fp8_te_dtype(
                ctx.fp8_meta["recipe"], fprop_tensor=False
            )

            grad_output_c = cast_to_fp8(
                grad_output,
                ctx.fp8_meta["scaling_bwd"],
                tex.FP8BwdTensors.GRAD_OUTPUT1,
                fp8_dtype_backward,
                stochastic_rounding=get_fp8_te_sr(ctx.fp8_meta["recipe"], fprop_tensor=False),
                measure_amax=ctx.amax_measure_state["enabled"]
            )

            attention_probs_grad = fp8_gemm(
                value_layer_fp8,
                fwd_scale_inverses[tex.FP8FwdTensors.GEMM1_WEIGHT],
                grad_output_c,
                ctx.fp8_meta["scaling_bwd"].scale_inv[tex.FP8BwdTensors.GRAD_OUTPUT1],
                ctx.activation_dtype,
                transa = True,
                transb = False
            )

            value_layer_grad = fp8_gemm(
                grad_output_c,
                ctx.fp8_meta["scaling_bwd"].scale_inv[
                    tex.FP8BwdTensors.GRAD_OUTPUT1
                ],
                attention_probs_fp8,
                fwd_scale_inverses[tex.FP8FwdTensors.GEMM1_INPUT],
                ctx.activation_dtype,
                transa = False,
                transb = True,
            )

            mixed_value_layer_grad = _SelfAttentionScoresAndValue._grad_transpose_for_scores(value_layer_grad)

        return (
            attention_probs_grad,
            mixed_value_layer_grad,
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
            None,
        )


class SelfAttentionContext(TransformerEngineBaseModule):
    """
    Applies the second phase of self attention module as implemented in Bert script.
    It replaces value transposition and context layer calculation.

    Parameters
    ----------
    hidden_size : int
                  The last dimension of input tensor.
    num_attention_heads : int
                  Number of attention heads. hidden_size must be divisible by this number.

    Other parameters
    ----------------
    All other parameters have the same meaning as in standard TE modules (e.g. Linear).
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation

        # TE module stuff
        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)
        self.set_nccl_overlap_warning_if_tp()

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel

        # Self attention stuff
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size


    def forward(self,
        attention_probs,
        value_layer,
        is_first_microbatch: Optional[bool] = None
    ):
        with self.prepare_forward(attention_probs, num_gemms = 1) as (attention_probs, is_scale_update_required):
            self.set_activation_dtype(value_layer)
            value_layer = value_layer.contiguous()

            context_layer = _SelfAttentionContext.apply(
                attention_probs,
                value_layer,
                self.num_attention_heads,
                self.attention_head_size,
                is_first_microbatch,
                self.fp8,
                self.fp8_meta,
                self.fuse_wgrad_accumulation,
                self.tp_group,
                self.sequence_parallel,
                self.tp_size > 1,
                self.activation_dtype,
                self.get_amax_measure_state(),
                is_scale_update_required,
            )

        return context_layer


