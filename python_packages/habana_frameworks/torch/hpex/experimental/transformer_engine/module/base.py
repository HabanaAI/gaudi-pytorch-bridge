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

"""Base modules and utilities for TransformerEngine PyTorch API"""
import os
import pickle
import warnings
from abc import ABC, abstractmethod
from typing import Generator, Union, Optional, Tuple, Dict, Any, List
from contextlib import contextmanager

import numpy as np
import torch

from habana_frameworks.torch import _hpex_C as tex
from ..fp8 import (
    is_fp8_enabled,
    get_fp8_recipe,
    get_fp8_group,
    get_default_fp8_recipe,
    get_fp8_te_dtype,
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
from ..distributed import (
    gather_along_first_dim,
    is_fp8_activation_recompute_enabled,
    in_fp8_activation_recompute_phase,
)
from ..cpp_extensions import (
    cast_to_fp8,
)
from ..constants import GemmParallelModes, dist_group_type

@contextmanager
def _prepare_backward(fp8: bool,
                      fp8_meta: Dict[str, Any],
                      amax_measure_state: dict,
                      is_scale_update_required: bool,
                      reduce_amax_across_tp_group: bool,
                      tp_group: Optional[dist_group_type] = None
) -> Generator[None, None, None]:
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

    if fp8 and fp8_meta["recipe"].reduce_amax:
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
        self.fp8_initialized = False
        self.fp8 = False
        self.fp8_meta = {}
        self.fp8_meta["fp8_group"] = None
        self.fp8_meta["recipe"] = get_default_fp8_recipe()
        self.fp8_meta_tensors_initialized = False
        self.tp_group = None
        self.tp_size = 1
        self.sequence_parallel = False
        self.fp8_weight_shapes = []
        self.run_cnt = 0
        self.fp8_meta["run_id_fwd_stack"] = []
        self.fp8_meta["update_amax_fwd"] = {}
        self.fp8_meta["update_amax_bwd"] = {}

    def _handle_changed_amax_history_size(self, fp8_meta_tensor_key, num_fp8_tensors):
        curr_len = self.fp8_meta[fp8_meta_tensor_key].amax_history.shape[0]
        need_len = self.fp8_meta["recipe"].amax_history_len
        if need_len < curr_len:
            index = self.fp8_meta[fp8_meta_tensor_key].amax_history_index[0]
            begin = index-self.fp8_meta["recipe"].amax_history_len+1
            end = index+1
            if (begin < 0):
                slice0 = self.fp8_meta[fp8_meta_tensor_key].amax_history[begin:]
                slice1 = self.fp8_meta[fp8_meta_tensor_key].amax_history[:end]
                self.fp8_meta[fp8_meta_tensor_key].amax_history = (
                    torch.cat((slice0, slice1))
                )
                self.fp8_meta[fp8_meta_tensor_key].amax_history_index[0] = self.fp8_meta["recipe"].amax_history_len-1
            else:
                self.fp8_meta[fp8_meta_tensor_key].amax_history = (
                    self.fp8_meta[fp8_meta_tensor_key].amax_history[begin: end].clone()
                )
                self.fp8_meta[fp8_meta_tensor_key].amax_history_index[0] = index-begin
        elif need_len > curr_len:
            index = self.fp8_meta[fp8_meta_tensor_key].amax_history_index[0]
            extra_rows = need_len - curr_len
            slice0 = self.fp8_meta[fp8_meta_tensor_key].amax_history[:index+1]
            slice1 = torch.zeros(
                extra_rows,
                num_fp8_tensors,
                dtype=torch.float32,
                device="hpu",
            )
            slice2 = self.fp8_meta[fp8_meta_tensor_key].amax_history[index+1:]
            self.fp8_meta[fp8_meta_tensor_key].amax_history = torch.cat((slice0, slice1, slice2))

    def set_meta_tensor(self, fwd: bool) -> None:
        """Init scales and amaxes for fwd | bwd."""
        fp8_meta_tensor_key = "scaling_fwd" if fwd else "scaling_bwd"

        num_fp8_tensors = (
            self.fp8_meta["num_gemms"] * 2 if fwd else self.fp8_meta["num_gemms"]
        )

        if self.fp8_meta_tensors_initialized:
            self._handle_changed_amax_history_size(fp8_meta_tensor_key, num_fp8_tensors)
            return

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
        self.set_meta_tensor(True)
        self.set_meta_tensor(False)
        self.fp8_meta_tensors_initialized = True

    def get_extra_state(self) -> torch.Tensor:
        """Save before checkpointing."""
        state = None
        if self.fp8:
            state = {}
            state["scale_fwd"] = self.fp8_meta["scaling_fwd"].scale
            state["scale_inv_fwd"] = self.fp8_meta["scaling_fwd"].scale_inv
            state["amax_history_fwd"] = self.fp8_meta["scaling_fwd"].amax_history
            state["amax_history_index_fwd"] = self.fp8_meta["scaling_fwd"].amax_history_index
            state["scale_bwd"] = self.fp8_meta["scaling_bwd"].scale
            state["scale_inv_bwd"] = self.fp8_meta["scaling_bwd"].scale_inv
            state["amax_history_bwd"] = self.fp8_meta["scaling_bwd"].amax_history
            state["amax_history_index_bwd"] = self.fp8_meta["scaling_bwd"].amax_history_index
            state["global_fp8_buffer"] = get_global_fp8_buffer()
            state["update_amax_fwd"] = self.fp8_meta["update_amax_fwd"]
            state["update_amax_bwd"] = self.fp8_meta["update_amax_bwd"]

            # Store other pickelable values.
            extra = {}
            for k, v in self.fp8_meta.items():
                if isinstance(v, (bool, int, float, str)):
                    extra[k] = v
            state["extra_fp8_variables"] = extra

        state_serialized = pickle.dumps(state)
        state_tensor = torch.tensor(np.frombuffer(state_serialized, dtype=np.uint8))

        return state_tensor

    def set_extra_state(self, state: torch.Tensor) -> None:
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

            # Restore global FP8 buffer state.
            set_global_fp8_buffer(state[4])
            self.fp8_meta["update_amax_fwd"] = state[5]
            self.fp8_meta["global_fp8_buffer_pos_fwd"] = state[6]
            self.fp8_meta["global_fp8_buffer_pos_bwd"] = state[7]
            self.fp8_meta[get_run_id_key(forward=True)] = state[8]
            self.fp8_meta[get_run_id_key(forward=False)] = state[9]
            return

        if isinstance(state, torch.Tensor):
            state = pickle.loads(state.detach().cpu().numpy().tobytes())
            if state is None:
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
        self.fp8_meta["scaling_fwd"].amax_history_index.copy_(state["amax_history_index_fwd"])
        self.fp8_meta["scaling_bwd"].scale.copy_(state["scale_bwd"])
        self.fp8_meta["scaling_bwd"].amax_history.copy_(state["amax_history_bwd"])
        self.fp8_meta["scaling_bwd"].amax_history_index.copy_(state["amax_history_index_bwd"])

        # Backwards compatibility: compute scale inv if it wasn't saved in the extra state.
        if "scale_inv_fwd" not in state or "scale_inv_bwd" not in state:
            assert (
                "scale_inv_fwd" not in state and "scale_inv_bwd" not in state
            ), "Invalid state, began saving scale_inv_fwd and scale_inv_bwd at the same time"
            self.fp8_meta["scaling_fwd"].scale_inv.copy_(1.0/state["scale_fwd"])
            self.fp8_meta["scaling_bwd"].scale_inv.copy_(1.0/state["scale_bwd"])
        else:
            self.fp8_meta["scaling_fwd"].scale_inv.copy_(state["scale_inv_fwd"])
            self.fp8_meta["scaling_bwd"].scale_inv.copy_(state["scale_inv_bwd"])

        self.fp8_meta["update_amax_fwd"] = state.get("update_amax_fwd", {})
        self.fp8_meta["update_amax_bwd"] = state.get("update_amax_bwd", {})

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

        dtype = inp.dtype
        for name, param in self.named_parameters():
            if param is not None:
                assert dtype == param.dtype, (
                    "Data types for parameters must match when outside of autocasted region. "
                    f" Found input dtype: {dtype} and {name!r} dtype: {param.dtype}"
                )
        for name, buf in self.named_buffers():
            if buf is not None:
                assert dtype == buf.dtype, (
                    "Data types for buffers must match when outside of autocasted region. "
                    f" Found input dtype: {dtype} and {name!r} dtype: {buf.dtype}"
                )
        self.activation_dtype = dtype

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
        self.fp8 = is_fp8_enabled()

        if self.fp8:
            # FP8 init has already been run and recipe is the same, don't do anything.
            if self.fp8_initialized and get_fp8_recipe() == self.fp8_meta["recipe"]:
                return

            # Set FP8, recipe, and other FP8 metadata
            self.fp8_meta["recipe"] = get_fp8_recipe()
            self.fp8_meta["num_gemms"] = num_gemms
            self.fp8_meta["fp8_group"] = get_fp8_group()

            # Set FP8_MAX per tensor according to recipe
            self.fp8_meta["fp8_max_fwd"] = self.fp8_meta["recipe"].fp8_format.value.max_fwd
            self.fp8_meta["fp8_max_bwd"] = self.fp8_meta["recipe"].fp8_format.value.max_bwd

            # Allocate scales and amaxes
            self.init_fp8_meta_tensors()
            self.fp8_initialized = True
        else:
            # If fp8 isn't enabled, turn off and return.
            self.fp8_initialized = False
            return

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
    def prepare_forward(
        self,
        inp: torch.Tensor,
        is_first_microbatch: Union[bool, None],
        num_gemms: int = 1
    ) -> Generator[tuple, None, None]:
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

            # Create persistent tensors for fp8 weights and their transposes
            # only when fp8 weight caching is used.
            if is_first_microbatch is not None:
                self.set_fp8_weights()

            if self.fp8 and self.sequence_parallel:
                assert self.fp8_meta["recipe"].reduce_amax, \
                "Amax reduction across tensor parallel group is " \
                "necessary when using sequence parallelism with FP8."

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

    @abstractmethod
    def get_fp8_weights_scratchpad(
        self,
        is_first_microbatch: Union[bool, None],
    ) -> List[torch.Tensor]:
        """Needs override."""
