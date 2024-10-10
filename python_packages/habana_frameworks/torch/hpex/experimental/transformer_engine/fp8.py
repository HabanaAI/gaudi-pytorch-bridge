# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE.txt for license information.
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
# - Minor code adaptations

"""FP8 utilities for TransformerEngine"""
from collections import deque
from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from .constants import dist_group_type
from .recipe import DelayedScaling, Format


def check_fp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    from habana_frameworks.torch.hpu import get_device_name

    if get_device_name() == "GAUDI":
        return False, "FP8 not supported on Gaudi, Gaudi2 or higher required"
    return True, ""


def get_default_fp8_recipe() -> DelayedScaling:
    """FP8 recipe if not provided by user"""
    return DelayedScaling()


def get_fp8_te_dtype(fp8_recipe: DelayedScaling, fprop_tensor: bool = True) -> torch.dtype:
    """Get fp8 data type according to recipe and tensor"""
    if fp8_recipe.fp8_format == Format.E4M3 or (fp8_recipe.fp8_format == Format.HYBRID and fprop_tensor):
        return torch.float8_e4m3fn
    return torch.float8_e5m2


class MetaTensorType(Enum):
    FORWARD = 0
    HYBRID = 1
    BACKWARD = 2


class FP8GlobalStateManager:
    FP8_ENABLED = False
    FP8_RECIPE = None
    FP8_DISTRIBUTED_GROUP = None
    IS_FIRST_FP8_MODULE = False
    FP8_AUTOCAST_COUNTER = 0
    FP8_CURRENT_CONTEXT_ID = 0
    FP8_AUTOCAST_DEPTH = 0
    FP8_MANUAL_MEASUREMENT = None
    global_fp8_buffer = {}
    fp8_tensors_recompute_buffer = []
    buffer_delete_key_fwd = None
    buffer_delete_key_bwd = None
    fp8_available = None
    reason_for_no_fp8 = ""

    @classmethod
    def is_fp8_available(cls) -> Tuple[bool, str]:
        """Return if fp8 support is available"""
        if cls.fp8_available is None:
            cls.fp8_available, cls.reason_for_no_fp8 = check_fp8_support()
        return cls.fp8_available, cls.reason_for_no_fp8

    @classmethod
    def get_global_fp8_state_checkpoint(cls) -> Dict[str, Union[int, str]]:
        """Returns global fp8 state variables."""
        # Convert attributes to dictionary to make future proof against
        # changes in global state variables in order to make setting the
        # checkpoint backwards compatible.
        global_fp8_state = {}
        global_fp8_state["FP8_AUTOCAST_COUNTER"] = cls.FP8_AUTOCAST_COUNTER
        global_fp8_state["FP8_CURRENT_CONTEXT_ID"] = cls.FP8_CURRENT_CONTEXT_ID
        global_fp8_state["FP8_AUTOCAST_DEPTH"] = cls.FP8_AUTOCAST_DEPTH
        global_fp8_state["FP8_MANUAL_MEASUREMENT"] = cls.FP8_MANUAL_MEASUREMENT
        global_fp8_state["buffer_delete_key_fwd"] = cls.buffer_delete_key_fwd
        global_fp8_state["buffer_delete_key_bwd"] = cls.buffer_delete_key_bwd
        return global_fp8_state

    @classmethod
    def set_global_fp8_state_checkpoint(cls, state: Dict[str, Union[int, str]]) -> None:
        """Sets global fp8 state variables."""
        for k, v in state.items():
            if hasattr(cls, k):
                setattr(cls, k, v)

    @classmethod
    def set_global_fp8_buffer_checkpoint(cls, buffer: Dict[str, List[torch.Tensor]]) -> None:
        """Sets global fp8 amax buffer."""
        # Map all tensors back to HPU.
        for k, v in buffer.items():
            buffer[k] = [tensor.to("hpu") for tensor in v]

        cls.global_fp8_buffer = buffer

    @classmethod
    def get_fp8_autocast_state(cls) -> Tuple[bool, DelayedScaling, dist_group_type, bool]:
        """FP8 autocast state getter"""
        return (cls.FP8_ENABLED, cls.FP8_RECIPE, cls.FP8_DISTRIBUTED_GROUP, cls.IS_FIRST_FP8_MODULE)

    @classmethod
    def set_fp8_autocast_state(cls, fp8_state: Tuple[bool, bool, DelayedScaling, dist_group_type, bool]) -> None:
        """FP8 autocast state setter"""
        (cls.FP8_ENABLED, cls.FP8_RECIPE, cls.FP8_DISTRIBUTED_GROUP, cls.IS_FIRST_FP8_MODULE) = fp8_state

    @classmethod
    def fp8_autocast_enter(
        cls,
        enabled: bool = False,
        force_measurement: Optional[bool] = None,
        fp8_recipe: Optional[DelayedScaling] = None,
        fp8_group: Optional[dist_group_type] = None,
    ) -> None:
        """Set state and tracking variables for entry into FP8 region."""
        cls.FP8_ENABLED = enabled
        cls.FP8_RECIPE = get_default_fp8_recipe() if fp8_recipe is None else fp8_recipe
        cls.FP8_DISTRIBUTED_GROUP = fp8_group

        if cls.FP8_AUTOCAST_DEPTH == 0:
            cls.IS_FIRST_FP8_MODULE = True
            cls.FP8_AUTOCAST_COUNTER += 1
            cls.FP8_MANUAL_MEASUREMENT = force_measurement
        cls.FP8_AUTOCAST_DEPTH += 1

        if enabled:
            fp8_available, reason_for_no_fp8 = cls.is_fp8_available()
            assert fp8_available, reason_for_no_fp8

    @classmethod
    def fp8_autocast_exit(cls):
        """Set state and tracking variables for exit from FP8 region."""
        cls.FP8_AUTOCAST_DEPTH -= 1

    @classmethod
    def get_old_fp8_meta_tensors_for_recompute(cls, fp8_meta: Dict[str, Any]) -> None:
        """Switch to the copied scaling factors and amaxes from phase
        1 forward for indentical numerical outputs.
        """

        # Store updated amaxes and scales from phase 1 post forward.
        def _store_updated_meta(t: MetaTensorType):
            key = cls.get_meta_tensor_key(t)
            key_suffix = cls.get_key_suffix(t)
            fp8_meta[f"updated_amax_history_{key_suffix}"] = fp8_meta[key].amax_history
            fp8_meta[f"updated_amax_history_index_{key_suffix}"] = fp8_meta[key].amax_history_index
            fp8_meta[f"updated_scale_{key_suffix}"] = fp8_meta[key].scale
            fp8_meta[f"updated_scale_inv_{key_suffix}"] = fp8_meta[key].scale_inv

        _store_updated_meta(MetaTensorType.FORWARD)
        if cls.is_hybrid_mode(fp8_meta):
            _store_updated_meta(MetaTensorType.HYBRID)

        # Retrieve stashed amaxes and scales from phase 1 pre forward.
        buffer_position_key = "global_fp8_buffer_pos_fwd_recompute"
        stashed_fp8_meta = cls.fp8_tensors_recompute_buffer[fp8_meta[buffer_position_key]].popleft()

        # Replace amaxes and scales with stashed values for phase 2 forward
        def _restore_meta(stashed, t: MetaTensorType):
            key = cls.get_meta_tensor_key(t)
            fp8_meta[key].amax_history = stashed[0]
            fp8_meta[key].amax_history_index = stashed[1]
            fp8_meta[key].scale = stashed[2]
            fp8_meta[key].scale_inv = stashed[3]

        _restore_meta(stashed_fp8_meta[:4], MetaTensorType.FORWARD)
        if cls.is_hybrid_mode(fp8_meta):
            _restore_meta(stashed_fp8_meta[4:], MetaTensorType.HYBRID)

    @classmethod
    def is_fp8_enabled(cls) -> bool:
        """Is FP8 enabled"""
        return cls.FP8_ENABLED

    @classmethod
    def get_manual_measurement_mode(cls):
        return cls.FP8_MANUAL_MEASUREMENT

    @classmethod
    def get_fp8_recipe(cls) -> DelayedScaling:
        """Return the fp8 recipe"""
        return cls.FP8_RECIPE

    @classmethod
    def get_fp8_group(cls) -> Union[dist_group_type, None]:
        """Return the fp8 group for scale/amax comm"""
        return cls.FP8_DISTRIBUTED_GROUP

    @classmethod
    def is_first_fp8_module(cls):
        """Returns `True` only the first time when called multiple
        times from within the same `fp8_autocast` context.
        """
        tmp = cls.IS_FIRST_FP8_MODULE
        cls.IS_FIRST_FP8_MODULE = False
        return tmp

    @classmethod
    def delete_key_from_amax_buffer(cls, forward: bool = True) -> None:
        """Delete the key from global amax buffer."""
        if forward:
            if cls.buffer_delete_key_fwd is not None and cls.buffer_delete_key_fwd in cls.global_fp8_buffer:
                del cls.global_fp8_buffer[cls.buffer_delete_key_fwd]
                cls.buffer_delete_key_fwd = None
        else:
            if cls.buffer_delete_key_bwd is not None and cls.buffer_delete_key_bwd in cls.global_fp8_buffer:
                del cls.global_fp8_buffer[cls.buffer_delete_key_bwd]
                cls.buffer_delete_key_bwd = None

    @classmethod
    def reset_global_state(cls):
        cls.set_fp8_autocast_counter(0)
        cls.clear_global_fp8_buffer()

    @classmethod
    def get_global_fp8_buffer_checkpoint(cls) -> Dict[str, List[torch.Tensor]]:
        """Returns global fp8 amax buffer."""
        return cls.global_fp8_buffer

    @classmethod
    def get_fp8_context_id(cls) -> int:
        """Returns an ID for the current FP8 context."""
        return cls.FP8_CURRENT_CONTEXT_ID

    @classmethod
    def set_fp8_context_id(cls, ctx_id: int) -> None:
        """Sets the current FP8 context."""
        cls.FP8_CURRENT_CONTEXT_ID = ctx_id

    @classmethod
    def new_fp8_context_id(cls) -> int:
        """Returns global autocast counter as a proxy to be used
        as the autocast ID for FP8 modules.
        """
        return cls.FP8_AUTOCAST_COUNTER

    @classmethod
    def set_fp8_autocast_counter(cls, value: int = 0):
        cls.FP8_AUTOCAST_COUNTER = value

    @classmethod
    def clear_global_fp8_buffer(cls):
        cls.global_fp8_buffer.clear()

    @classmethod
    def set_measurement_mode(cls, manual: bool, manual_value: bool = True):
        if manual:
            cls.FP8_MANUAL_MEASUREMENT = manual_value
        else:
            cls.FP8_MANUAL_MEASUREMENT = None

    @staticmethod
    def get_meta_tensor_key(t: MetaTensorType):
        """Returns scaling key in `fp8_meta`."""
        assert isinstance(t, MetaTensorType)

        if t == MetaTensorType.FORWARD:
            return "scaling_fwd"
        if t == MetaTensorType.BACKWARD:
            return "scaling_bwd"
        if t == MetaTensorType.HYBRID:
            return "scaling_hybrid"

    @classmethod
    def get_meta_tensor_key_bool(cls, forward: bool = True) -> str:
        """Returns scaling key in `fp8_meta`."""
        if forward:
            return cls.get_meta_tensor_key(MetaTensorType.FORWARD)
        return cls.get_meta_tensor_key(MetaTensorType.BACKWARD)

    @staticmethod
    def get_buffer_position_key(forward: bool = True) -> str:
        """Returns module position key in `fp8_meta`."""
        if forward:
            return "global_fp8_buffer_pos_fwd"
        return "global_fp8_buffer_pos_bwd"

    @staticmethod
    def get_fp8_max_key(t: MetaTensorType):
        if t == MetaTensorType.FORWARD:
            return "fp8_max_fwd"
        if t in [MetaTensorType.HYBRID, MetaTensorType.BACKWARD]:
            return "fp8_max_bwd"

    @staticmethod
    def get_key_suffix(t: MetaTensorType):
        if t == MetaTensorType.FORWARD:
            return "fwd"
        if t == MetaTensorType.BACKWARD:
            return "bwd"
        if t == MetaTensorType.HYBRID:
            return "hybrid"

    @staticmethod
    def is_forward(t: MetaTensorType):
        return t in [MetaTensorType.FORWARD, MetaTensorType.HYBRID]

    @classmethod
    def is_hybrid_mode(cls, fp8_meta: Dict[str, Any]):
        """Checks if hybrid mode without mixed precision is turned on"""
        return cls.get_meta_tensor_key(MetaTensorType.HYBRID) in fp8_meta

    @staticmethod
    def get_autocast_key(forward: bool = True) -> str:
        """Returns module position key in `fp8_meta`."""
        if forward:
            return "autocast_id_fwd"
        return "autocast_id_bwd"

    @staticmethod
    def get_run_id_key(forward: bool = True) -> str:
        """Returns module position key in `fp8_meta`."""
        if forward:
            return "run_id_fwd"
        return "run_id_bwd"

    @classmethod
    def get_amax_buffer_key(cls, fp8_meta: Dict[str, Any], forward: bool = True) -> str:
        """Return a key in `global_fp8_buffer` for the AMAX storage."""
        if forward:
            return f"FWD_AMAX_{fp8_meta[cls.get_run_id_key(forward)]}"
        return f"BWD_AMAX_{fp8_meta[cls.get_run_id_key(forward)]}"

    @classmethod
    def add_amax_to_global_buffer(cls, fp8_meta: Dict[str, Any], forward: bool = True) -> None:
        """Append 1D tensor `amax` to global buffer."""
        buffer_key = cls.get_amax_buffer_key(fp8_meta, forward=forward)
        # NOTE: For hybrid mode amax_history is the same as for forward. To limit the number
        # of reduce operation, we only reduce fwd amax_history (to later copy it to fwd and hybrid, if exists)
        fp8_meta_tensor_key = cls.get_meta_tensor_key_bool(forward=forward)
        buffer_position_key = cls.get_buffer_position_key(forward=forward)

        if buffer_key not in cls.global_fp8_buffer:
            cls.global_fp8_buffer[buffer_key] = [
                fp8_meta[fp8_meta_tensor_key].amax_history[fp8_meta[fp8_meta_tensor_key].amax_history_index][0]
            ]
        else:
            cls.global_fp8_buffer[buffer_key].append(
                fp8_meta[fp8_meta_tensor_key].amax_history[fp8_meta[fp8_meta_tensor_key].amax_history_index][0]
            )

        if buffer_position_key not in fp8_meta:
            fp8_meta[buffer_position_key] = len(cls.global_fp8_buffer[buffer_key]) - 1

    @classmethod
    def copy_amax_from_global_buffer(cls, fp8_meta: Dict[str, Any], forward: bool = True) -> None:
        """Populate current amax with the correct location from buffer."""
        fp8_meta_tensor_key = cls.get_meta_tensor_key_bool(forward=forward)
        buffer_position_key = cls.get_buffer_position_key(forward=forward)
        if buffer_position_key not in fp8_meta:
            return

        amax_buffer_key = cls.get_amax_buffer_key(fp8_meta, forward=forward)
        assert amax_buffer_key in cls.global_fp8_buffer, "TE internal error."

        fp8_meta[fp8_meta_tensor_key].amax_history[fp8_meta[fp8_meta_tensor_key].amax_history_index] = (
            cls.global_fp8_buffer[amax_buffer_key][fp8_meta[buffer_position_key]]
        )

        # NOTE: For hybrid mode amax_history is the same as for forward. To limit the number
        # of reduce operation, only fwd amax_history was reduced. Now the reduction result needs to be copied also to hybrid
        if forward and cls.is_hybrid_mode(fp8_meta):
            hybrid_key = cls.get_meta_tensor_key(MetaTensorType.HYBRID)
            fp8_meta[hybrid_key].amax_history[fp8_meta[hybrid_key].amax_history_index] = cls.global_fp8_buffer[
                amax_buffer_key
            ][fp8_meta[buffer_position_key]]

    @classmethod
    def set_amax_buffer_key_deletion(cls, fp8_meta: Dict[str, Any], forward: bool = True) -> None:
        """Delete this amax key from global buffer during autocast end."""
        if cls.get_run_id_key(forward=forward) not in fp8_meta:
            return
        if forward:
            cls.buffer_delete_key_fwd = cls.get_amax_buffer_key(fp8_meta, forward=forward)
        else:
            cls.buffer_delete_key_bwd = cls.get_amax_buffer_key(fp8_meta, forward=forward)

    @staticmethod
    def reduce_tensor_across_group_op_max(tensor: torch.Tensor, group: dist_group_type) -> None:
        """Reduce tensor across given group."""
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MAX,
                group=group,
                async_op=False,
            )

    @classmethod
    def global_amax_reduction(
        cls,
        fp8_meta: Dict[str, Any],
        reduce_amax_across_tp_group: bool = False,
        tp_group: Optional[dist_group_type] = None,
        forward: bool = True,
    ) -> None:
        """Concatenate, reduce, and split amaxes in the global buffer."""
        amax_buffer_key = cls.get_amax_buffer_key(fp8_meta, forward=forward)

        # Key already deleted.
        if amax_buffer_key not in cls.global_fp8_buffer:
            return

        chunk_sizes = [x.numel() for x in cls.global_fp8_buffer[amax_buffer_key]]
        contiguous_amax = torch.cat(cls.global_fp8_buffer[amax_buffer_key])

        cls.reduce_tensor_across_group_op_max(contiguous_amax, fp8_meta["fp8_group"])
        if reduce_amax_across_tp_group:
            cls.reduce_tensor_across_group_op_max(contiguous_amax, tp_group)

        cls.global_fp8_buffer[amax_buffer_key] = list(contiguous_amax.split(chunk_sizes))

    @classmethod
    def copy_forward_fp8_meta_tensors_for_recompute(cls, fp8_meta: Dict[str, Any]) -> None:
        """Copy the scaling factors and amaxes for recompute forward phase
        to ensure both forward steps are numerically same.
        """
        buffer_position_key = "global_fp8_buffer_pos_fwd_recompute"

        def _append_meta(collection, key):
            collection.append(fp8_meta[key].amax_history.clone())
            collection.append(fp8_meta[key].amax_history_index.clone())
            collection.append(fp8_meta[key].scale.clone())
            collection.append(fp8_meta[key].scale_inv.clone())

        fwd_key = cls.get_meta_tensor_key(MetaTensorType.FORWARD)
        to_copy = []
        _append_meta(to_copy, fwd_key)

        if cls.is_hybrid_mode(fp8_meta):
            hybrid_key = cls.get_meta_tensor_key(MetaTensorType.HYBRID)
            _append_meta(to_copy, hybrid_key)

        if buffer_position_key in fp8_meta:
            cls.fp8_tensors_recompute_buffer[fp8_meta[buffer_position_key]].append(to_copy)
        else:
            if len(cls.fp8_tensors_recompute_buffer) == 0:
                cls.fp8_tensors_recompute_buffer = [deque()]
            else:
                cls.fp8_tensors_recompute_buffer.append(deque())
            cls.fp8_tensors_recompute_buffer[-1].append(to_copy)
            fp8_meta[buffer_position_key] = len(cls.fp8_tensors_recompute_buffer) - 1

    @classmethod
    def restore_fp8_meta_tensors(cls, fp8_meta: Dict[str, Any]) -> None:
        """Restore latest scaling factors and amaxes after recompute forward run."""

        def _restore_updated_meta(t: MetaTensorType):
            key = cls.get_meta_tensor_key(t)
            key_suffix = cls.get_key_suffix(t)
            fp8_meta[key].amax_history = fp8_meta[f"updated_amax_history_{key_suffix}"]
            fp8_meta[key].amax_history_index = fp8_meta[f"updated_amax_history_index_{key_suffix}"]
            fp8_meta[key].scale = fp8_meta[f"updated_scale_{key_suffix}"]
            fp8_meta[key].scale_inv = fp8_meta[f"updated_scale_inv_{key_suffix}"]

        _restore_updated_meta(MetaTensorType.FORWARD)
        if cls.is_hybrid_mode(fp8_meta):
            _restore_updated_meta(MetaTensorType.HYBRID)


def get_fp8_te_dtype(fp8_recipe: DelayedScaling, fprop_tensor: bool = True) -> torch.dtype:
    """Get fp8 data type according to recipe and tensor"""
    if fp8_recipe.fp8_format == Format.E4M3 or (fp8_recipe.fp8_format == Format.HYBRID and fprop_tensor):
        return torch.float8_e4m3fn
    return torch.float8_e5m2


@contextmanager
def fp8_autocast(
    enabled: bool = False,
    force_measurement: Optional[bool] = None,
    fp8_recipe: Optional[DelayedScaling] = None,
    fp8_group: Optional[dist_group_type] = None,
) -> None:
    """
    Context manager for FP8 usage.

    .. code-block:: python

        with fp8_autocast(enabled=True):
            out = model(inp)

    Parameters
    ----------
    enabled: bool, default = `False`
             whether or not to enable fp8
    force_measurement: Optional[bool], default = `None`
                       whether or not to force amax measurement. If left as None,
                       recipe interval will be used to enable and disable measurement.
    fp8_recipe: recipe.DelayedScaling, default = `None`
                recipe used for FP8 training.
    fp8_group: torch._C._distributed_c10d.ProcessGroup, default = `None`
               distributed group over which amaxes for the fp8 tensors
               are reduced at the end of each training step.
    """

    fp8_state = FP8GlobalStateManager.get_fp8_autocast_state()
    try:
        FP8GlobalStateManager.fp8_autocast_enter(enabled, force_measurement, fp8_recipe, fp8_group)

        yield
    finally:
        FP8GlobalStateManager.set_fp8_autocast_state(fp8_state)  # pylint: disable=used-before-assignment
        FP8GlobalStateManager.fp8_autocast_exit()


def set_measurement_mode(manual: bool, manual_value: bool = True):
    FP8GlobalStateManager.set_measurement_mode(manual, manual_value)


def _default_get_amax(
    amax_history: torch.Tensor,
    amax_compute_algo: str,
) -> torch.Tensor:
    """Default function to obtain amax from history."""
    if amax_compute_algo == "max" and amax_history.shape[0] > 1:
        amax = torch.max(amax_history, dim=0).values
    else:  # amax_compute_algo == "most_recent"
        amax = amax_history[0]

    return amax


def _default_sf_compute(
    amax: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: float,
    margin: int,
) -> torch.Tensor:
    """Default function to convert amax to scaling factor."""
    exp = torch.floor(torch.log2(fp8_max / amax)) - margin
    sf = torch.pow(2.0, exp)
    sf = torch.where(amax > 0.0, sf, scale)

    return sf


def _fused_amax_and_scale_update(
    amax_history: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: float,
    margin: int,
    amax_compute_algo: str,
) -> torch.Tensor:
    """Amax to scale conversion."""

    # Get amax from history.
    amax = _default_get_amax(
        amax_history,
        amax_compute_algo,
    )

    # Calculate new scaling factor.
    return _default_sf_compute(
        amax,
        scale,
        fp8_max,
        margin,
    )


def _compute_amax(
    amax_history: torch.Tensor,
    recipe: DelayedScaling,
) -> torch.Tensor:
    """Obtain the amax from the history."""

    if callable(recipe.amax_compute_algo):
        return recipe.amax_compute_algo(amax_history)
    return _default_get_amax(
        amax_history,
        recipe.amax_compute_algo,
    )


def _compute_scaling_factor(
    amax: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: float,
    recipe: DelayedScaling,
) -> torch.Tensor:
    """Convert amax to scaling factor."""

    if recipe.scaling_factor_compute_algo is None:
        return _default_sf_compute(
            amax,
            scale,
            fp8_max,
            recipe.margin,
        )
    return recipe.scaling_factor_compute_algo(amax, scale, fp8_max, recipe)


def update_amax_history_index(fp8_meta: Dict[str, Any], fp8_meta_tensor_key: str):
    if fp8_meta["recipe"].amax_history_len > 1:
        fp8_meta[fp8_meta_tensor_key].amax_history_index.add_(1)
        fp8_meta[fp8_meta_tensor_key].amax_history_index.remainder_(fp8_meta["recipe"].amax_history_len)


def amax_and_scale_update(
    fp8_meta: Dict[str, Any],
    fwd_update: bool,
    perform_scale_update: bool,
) -> None:
    """Updates fp8 amaxes/scales for fwd | bwd."""

    def _update(meta_tensor_type: MetaTensorType):
        fp8_meta_tensor_key = FP8GlobalStateManager.get_meta_tensor_key(meta_tensor_type)
        fp8_max_key = FP8GlobalStateManager.get_fp8_max_key(meta_tensor_type)

        if perform_scale_update:
            amax_compute = fp8_meta["recipe"].amax_compute_algo
            sf_compute = fp8_meta["recipe"].scaling_factor_compute_algo

            if not callable(amax_compute) and sf_compute is None:
                fp8_meta[fp8_meta_tensor_key].scale = _fused_amax_and_scale_update(
                    fp8_meta[fp8_meta_tensor_key].amax_history,
                    fp8_meta[fp8_meta_tensor_key].scale,
                    fp8_meta[fp8_max_key],
                    fp8_meta["recipe"].margin,
                    fp8_meta["recipe"].amax_compute_algo,
                )
            else:
                amax = _compute_amax(
                    fp8_meta[fp8_meta_tensor_key].amax_history,
                    fp8_meta["recipe"],
                )
                fp8_meta[fp8_meta_tensor_key].scale = _compute_scaling_factor(
                    amax,
                    fp8_meta[fp8_meta_tensor_key].scale,
                    fp8_meta[fp8_max_key],
                    fp8_meta["recipe"],
                )

            fp8_meta[fp8_meta_tensor_key].scale_inv = torch.reciprocal(fp8_meta[fp8_meta_tensor_key].scale)

        update_amax_history_index(fp8_meta, fp8_meta_tensor_key)

    if fwd_update:
        _update(MetaTensorType.FORWARD)
        if FP8GlobalStateManager.is_hybrid_mode(fp8_meta):
            _update(MetaTensorType.HYBRID)
    else:
        _update(MetaTensorType.BACKWARD)


def get_fp8_te_sr(fp8_recipe: DelayedScaling, fprop_tensor: bool = True) -> bool:
    """Get fp8 stochastic rounding flag according to recipe, tensor and env flag"""
    # Always disabled in fwd pass
    if fprop_tensor:
        return False

    # Force flag has the priority
    import os

    force_sr_bwd = os.getenv("PT_TE_FORCE_SR_BWD")
    if force_sr_bwd is not None:
        return force_sr_bwd.lower() in ["true", "1"]

    # If force flag not set, decide based on recipe format
    return fp8_recipe.fp8_format == Format.HYBRID
