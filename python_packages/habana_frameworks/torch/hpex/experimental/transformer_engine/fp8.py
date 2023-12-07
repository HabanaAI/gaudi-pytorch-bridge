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
# - Minor code adaptations

"""FP8 utilities for TransformerEngine"""
from collections import deque
from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from .constants import dist_group_type
from .recipe import DelayedScaling, Format

_FP8_ENABLED = False
_FP8_RECIPE = None
_FP8_DISTRIBUTED_GROUP = None
_IS_FIRST_FP8_MODULE = False
_FP8_AUTOCAST_COUNTER = 0
_FP8_CURRENT_CONTEXT_ID = 0
_FP8_AUTOCAST_DEPTH = 0
_FP8_MANUAL_MEASUREMENT = None
_global_fp8_buffer = {}
_fp8_tensors_recompute_buffer = []
_buffer_delete_key_fwd = None
_buffer_delete_key_bwd = None
_is_fp8_available = None
_reason_for_no_fp8 = ""


def _check_fp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    from habana_frameworks.torch.hpu import get_device_name

    if get_device_name() == "GAUDI":
        return False, "FP8 not supported on Gaudi, Gaudi2 or higher required"
    return True, ""


def is_fp8_available() -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    global _is_fp8_available, _reason_for_no_fp8
    if _is_fp8_available is None:
        _is_fp8_available, _reason_for_no_fp8 = _check_fp8_support()
    return _is_fp8_available, _reason_for_no_fp8


class MetaTensorType(Enum):
    FORWARD = 0
    HYBRID = 1
    BACKWARD = 2


def get_meta_tensor_key(t: MetaTensorType):
    """Returns scaling key in `fp8_meta`."""
    assert isinstance(t, MetaTensorType)

    if t == MetaTensorType.FORWARD:
        return "scaling_fwd"
    if t == MetaTensorType.BACKWARD:
        return "scaling_bwd"
    if t == MetaTensorType.HYBRID:
        return "scaling_hybrid"


def get_meta_tensor_key_bool(forward: bool = True) -> str:
    """Returns scaling key in `fp8_meta`."""
    if forward:
        return get_meta_tensor_key(MetaTensorType.FORWARD)
    return get_meta_tensor_key(MetaTensorType.BACKWARD)


def get_fp8_max_key(t: MetaTensorType):
    if t == MetaTensorType.FORWARD:
        return "fp8_max_fwd"
    if t in [MetaTensorType.HYBRID, MetaTensorType.BACKWARD]:
        return "fp8_max_bwd"


def get_key_suffix(t: MetaTensorType):
    if t == MetaTensorType.FORWARD:
        return "fwd"
    if t == MetaTensorType.BACKWARD:
        return "bwd"
    if t == MetaTensorType.HYBRID:
        return "hybrid"


def is_forward(t: MetaTensorType):
    return t in [MetaTensorType.FORWARD, MetaTensorType.HYBRID]


def is_hybrid_mode(fp8_meta: Dict[str, Any]):
    """Checks if hybrid mode without mixed precision is turned on"""
    return get_meta_tensor_key(MetaTensorType.HYBRID) in fp8_meta


def get_buffer_position_key(forward: bool = True) -> str:
    """Returns module position key in `fp8_meta`."""
    if forward:
        return "global_fp8_buffer_pos_fwd"
    return "global_fp8_buffer_pos_bwd"


def get_autocast_key(forward: bool = True) -> str:
    """Returns module position key in `fp8_meta`."""
    if forward:
        return "autocast_id_fwd"
    return "autocast_id_bwd"


def get_run_id_key(forward: bool = True) -> str:
    """Returns module position key in `fp8_meta`."""
    if forward:
        return "run_id_fwd"
    return "run_id_bwd"


def get_global_fp8_buffer() -> Dict[str, List[torch.Tensor]]:
    """Returns global fp8 buffer."""
    return _global_fp8_buffer


def set_global_fp8_buffer(buffer: Dict[str, List[torch.Tensor]]) -> None:
    """Sets global fp8 buffer."""
    global _global_fp8_buffer

    # Map all tensors back to GPU.
    for k, v in buffer.items():
        buffer[k] = [tensor.to("hpu") for tensor in v]

    _global_fp8_buffer = buffer


def get_amax_buffer_key(fp8_meta: Dict[str, Any], forward: bool = True) -> str:
    """Return a key in `_global_fp8_buffer` for the AMAX storage."""
    if forward:
        return f"FWD_AMAX_{fp8_meta[get_run_id_key(forward)]}"
    return f"BWD_AMAX_{fp8_meta[get_run_id_key(forward)]}"


def add_amax_to_global_buffer(fp8_meta: Dict[str, Any], forward: bool = True) -> None:
    """Append 1D tensor `amax` to global buffer."""

    global _global_fp8_buffer
    buffer_key = get_amax_buffer_key(fp8_meta, forward=forward)
    # NOTE: For hybrid mode amax_history is the same as for forward. To limit the number
    # of reduce operation, we only reduce fwd amax_history (to later copy it to fwd and hybrid, if exists)
    fp8_meta_tensor_key = get_meta_tensor_key_bool(forward=forward)
    buffer_position_key = get_buffer_position_key(forward=forward)

    if buffer_key not in _global_fp8_buffer:
        _global_fp8_buffer[buffer_key] = [
            fp8_meta[fp8_meta_tensor_key].amax_history[fp8_meta[fp8_meta_tensor_key].amax_history_index][0]
        ]
    else:
        _global_fp8_buffer[buffer_key].append(
            fp8_meta[fp8_meta_tensor_key].amax_history[fp8_meta[fp8_meta_tensor_key].amax_history_index][0]
        )

    if buffer_position_key not in fp8_meta:
        fp8_meta[buffer_position_key] = len(_global_fp8_buffer[buffer_key]) - 1


def copy_forward_fp8_meta_tensors_for_recompute(fp8_meta: Dict[str, Any]) -> None:
    """Copy the scaling factors and amaxes for recompute forward phase
    to ensure both forward steps are numerically same.
    """
    global _fp8_tensors_recompute_buffer
    buffer_position_key = "global_fp8_buffer_pos_fwd_recompute"

    def _append_meta(collection, key):
        collection.append(fp8_meta[key].amax_history.clone())
        collection.append(fp8_meta[key].amax_history_index.clone())
        collection.append(fp8_meta[key].scale.clone())
        collection.append(fp8_meta[key].scale_inv.clone())

    fwd_key = get_meta_tensor_key(MetaTensorType.FORWARD)
    to_copy = []
    _append_meta(to_copy, fwd_key)

    if is_hybrid_mode(fp8_meta):
        hybrid_key = get_meta_tensor_key(MetaTensorType.HYBRID)
        _append_meta(to_copy, hybrid_key)

    if buffer_position_key in fp8_meta:
        _fp8_tensors_recompute_buffer[fp8_meta[buffer_position_key]].append(to_copy)
    else:
        if len(_fp8_tensors_recompute_buffer) == 0:
            _fp8_tensors_recompute_buffer = [deque()]
        else:
            _fp8_tensors_recompute_buffer.append(deque())
        _fp8_tensors_recompute_buffer[-1].append(to_copy)
        fp8_meta[buffer_position_key] = len(_fp8_tensors_recompute_buffer) - 1


def get_old_fp8_meta_tensors_for_recompute(fp8_meta: Dict[str, Any]) -> None:
    """Switch to the copied scaling factors and amaxes from phase
    1 forward for indentical numerical outputs.
    """

    # Store updated amaxes and scales from phase 1 post forward.
    def _store_updated_meta(t: MetaTensorType):
        key = get_meta_tensor_key(t)
        key_suffix = get_key_suffix(t)
        fp8_meta[f"updated_amax_history_{key_suffix}"] = fp8_meta[key].amax_history
        fp8_meta[f"updated_amax_history_index_{key_suffix}"] = fp8_meta[key].amax_history_index
        fp8_meta[f"updated_scale_{key_suffix}"] = fp8_meta[key].scale
        fp8_meta[f"updated_scale_inv_{key_suffix}"] = fp8_meta[key].scale_inv

    _store_updated_meta(MetaTensorType.FORWARD)
    if is_hybrid_mode(fp8_meta):
        _store_updated_meta(MetaTensorType.HYBRID)

    # Retrieve stashed amaxes and scales from phase 1 pre forward.
    buffer_position_key = "global_fp8_buffer_pos_fwd_recompute"
    stashed_fp8_meta = _fp8_tensors_recompute_buffer[fp8_meta[buffer_position_key]].popleft()

    # Replace amaxes and scales with stashed values for phase 2 forward
    def _restore_meta(stashed, t: MetaTensorType):
        key = get_meta_tensor_key(t)
        fp8_meta[key].amax_history = stashed[0]
        fp8_meta[key].amax_history_index = stashed[1]
        fp8_meta[key].scale = stashed[2]
        fp8_meta[key].scale_inv = stashed[3]

    _restore_meta(stashed_fp8_meta[:4], MetaTensorType.FORWARD)
    if is_hybrid_mode(fp8_meta):
        _restore_meta(stashed_fp8_meta[4:], MetaTensorType.HYBRID)


def restore_fp8_meta_tensors(fp8_meta: Dict[str, Any]) -> None:
    """Restore latest scaling factors and amaxes after recompute forward run."""

    def _restore_updated_meta(t: MetaTensorType):
        key = get_meta_tensor_key(t)
        key_suffix = get_key_suffix(t)
        fp8_meta[key].amax_history = fp8_meta[f"updated_amax_history_{key_suffix}"]
        fp8_meta[key].amax_history_index = fp8_meta[f"updated_amax_history_index_{key_suffix}"]
        fp8_meta[key].scale = fp8_meta[f"updated_scale_{key_suffix}"]
        fp8_meta[key].scale_inv = fp8_meta[f"updated_scale_inv_{key_suffix}"]

    _restore_updated_meta(MetaTensorType.FORWARD)
    if is_hybrid_mode(fp8_meta):
        _restore_updated_meta(MetaTensorType.HYBRID)


def copy_amax_from_global_buffer(fp8_meta: Dict[str, Any], forward: bool = True) -> None:
    """Populate current amax with the correct location from buffer."""

    fp8_meta_tensor_key = get_meta_tensor_key_bool(forward=forward)
    buffer_position_key = get_buffer_position_key(forward=forward)
    if buffer_position_key not in fp8_meta:
        return

    amax_buffer_key = get_amax_buffer_key(fp8_meta, forward=forward)
    assert amax_buffer_key in _global_fp8_buffer, "TE internal error."

    fp8_meta[fp8_meta_tensor_key].amax_history[fp8_meta[fp8_meta_tensor_key].amax_history_index][0] = (
        _global_fp8_buffer[amax_buffer_key][fp8_meta[buffer_position_key]]
    )

    # NOTE: For hybrid mode amax_history is the same as for forward. To limit the number
    # of reduce operation, only fwd amax_history was reduced. Now the reduction result needs to be copied also to hybrid
    if forward and is_hybrid_mode(fp8_meta):
        hybrid_key = get_meta_tensor_key(MetaTensorType.HYBRID)
        fp8_meta[hybrid_key].amax_history[fp8_meta[hybrid_key].amax_history_index][0] = _global_fp8_buffer[
            amax_buffer_key
        ][fp8_meta[buffer_position_key]]


def set_amax_buffer_key_deletion(fp8_meta: Dict[str, Any], forward: bool = True) -> None:
    """Delete this amax key from global buffer during autocast end."""

    if get_run_id_key(forward=forward) not in fp8_meta:
        return
    global _buffer_delete_key_fwd, _buffer_delete_key_bwd
    if forward:
        _buffer_delete_key_fwd = get_amax_buffer_key(fp8_meta, forward=forward)
    else:
        _buffer_delete_key_bwd = get_amax_buffer_key(fp8_meta, forward=forward)


def get_default_fp8_recipe() -> DelayedScaling:
    """FP8 recipe if not provided by user"""
    return DelayedScaling()


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

    .. note::

        Support for FP8 in the Linear layer of Transformer Engine is currently limited to tensors
        with shapes where both dimensions are divisible by 16. In terms of the input to the full
        Transformer network, this typically requires padding sequence length to be multiple of 16.

    Parameters
    ----------
    enabled: bool, default = `False`
             whether or not to enable fp8
    fp8_recipe: recipe.DelayedScaling, default = `None`
                recipe used for FP8 training.
    fp8_group: torch._C._distributed_c10d.ProcessGroup, default = `None`
               distributed group over which amaxes for the fp8 tensors
               are reduced at the end of each training step.
    """

    global _FP8_ENABLED, _FP8_RECIPE, _FP8_DISTRIBUTED_GROUP, _FP8_AUTOCAST_DEPTH
    global _IS_FIRST_FP8_MODULE, _FP8_AUTOCAST_COUNTER
    global _global_fp8_buffer, _buffer_delete_key_fwd
    global _FP8_MANUAL_MEASUREMENT
    fp8_state = (_FP8_ENABLED, _FP8_RECIPE, _FP8_DISTRIBUTED_GROUP, _IS_FIRST_FP8_MODULE)
    try:
        _FP8_ENABLED = enabled
        _FP8_RECIPE = get_default_fp8_recipe() if fp8_recipe is None else fp8_recipe
        _FP8_DISTRIBUTED_GROUP = fp8_group

        if _FP8_AUTOCAST_DEPTH == 0:
            _IS_FIRST_FP8_MODULE = True
            _FP8_AUTOCAST_COUNTER += 1
            _FP8_MANUAL_MEASUREMENT = force_measurement
        _FP8_AUTOCAST_DEPTH += 1

        if enabled:
            fp8_available, reason_for_no_fp8 = is_fp8_available()
            assert fp8_available, reason_for_no_fp8

        yield
    finally:
        _FP8_ENABLED, _FP8_RECIPE, _FP8_DISTRIBUTED_GROUP, _IS_FIRST_FP8_MODULE = fp8_state
        _FP8_AUTOCAST_DEPTH -= 1


def get_fp8_context_id() -> int:
    """Returns an ID for the current FP8 context."""
    return _FP8_CURRENT_CONTEXT_ID


def set_fp8_context_id(ctx_id: int) -> None:
    """Sets the current FP8 context."""
    global _FP8_CURRENT_CONTEXT_ID
    _FP8_CURRENT_CONTEXT_ID = ctx_id


def new_fp8_context_id() -> int:
    """Returns global autocast counter as a proxy to be used
    as the autocast ID for FP8 modules.
    """
    return _FP8_AUTOCAST_COUNTER


def set_fp8_autocast_counter(value: int = 0):
    global _FP8_AUTOCAST_COUNTER
    _FP8_AUTOCAST_COUNTER = value


def clear_global_fp8_buffer():
    _global_fp8_buffer.clear()


def reset_global_state():
    set_fp8_autocast_counter(0)
    clear_global_fp8_buffer()


def is_fp8_enabled() -> bool:
    """Is FP8 enabled"""
    return _FP8_ENABLED


def is_first_fp8_module():
    """Returns `True` only the first time when called multiple
    times from within the same `fp8_autocast` context.
    """
    global _IS_FIRST_FP8_MODULE
    tmp = _IS_FIRST_FP8_MODULE
    _IS_FIRST_FP8_MODULE = False
    return tmp


def set_measurement_mode(manual: bool, manual_value: bool = True):
    global _FP8_MANUAL_MEASUREMENT
    if manual:
        _FP8_MANUAL_MEASUREMENT = manual_value
    else:
        _FP8_MANUAL_MEASUREMENT = None


def get_manual_measurement_mode():
    return _FP8_MANUAL_MEASUREMENT


def get_fp8_recipe() -> DelayedScaling:
    """Return the fp8 recipe"""
    return _FP8_RECIPE


def get_fp8_group() -> Union[dist_group_type, None]:
    """Return the fp8 group for scale/amax comm"""
    return _FP8_DISTRIBUTED_GROUP


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


def fused_amax_and_scale_update(
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
        fp8_meta_tensor_key = get_meta_tensor_key(meta_tensor_type)
        fp8_max_key = get_fp8_max_key(meta_tensor_type)

        if perform_scale_update:
            amax_compute = fp8_meta["recipe"].amax_compute_algo
            sf_compute = fp8_meta["recipe"].scaling_factor_compute_algo

            if not callable(amax_compute) and sf_compute is None:
                fp8_meta[fp8_meta_tensor_key].scale = fused_amax_and_scale_update(
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
        if is_hybrid_mode(fp8_meta):
            _update(MetaTensorType.HYBRID)
    else:
        _update(MetaTensorType.BACKWARD)


def get_fp8_te_dtype(fp8_recipe: DelayedScaling, fprop_tensor: bool = True) -> torch.dtype:
    """Get fp8 data type according to recipe and tensor"""
    if fp8_recipe.fp8_format == Format.E4M3 or (fp8_recipe.fp8_format == Format.HYBRID and fprop_tensor):
        return torch.float8_e4m3fn
    return torch.float8_e5m2


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


def reduce_tensor_across_group_op_max(tensor: torch.Tensor, group: dist_group_type) -> None:
    """Reduce tensor across given group."""
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(
            tensor,
            op=torch.distributed.ReduceOp.MAX,
            group=group,
            async_op=False,
        )


def global_amax_reduction(
    fp8_meta: Dict[str, Any],
    reduce_amax_across_tp_group: bool = False,
    tp_group: Optional[dist_group_type] = None,
    forward: bool = True,
) -> None:
    """Concatenate, reduce, and split amaxes in the global buffer."""
    global _global_fp8_buffer
    amax_buffer_key = get_amax_buffer_key(fp8_meta, forward=forward)

    # Key already deleted.
    if amax_buffer_key not in _global_fp8_buffer:
        return

    chunk_sizes = [x.numel() for x in _global_fp8_buffer[amax_buffer_key]]
    contiguous_amax = torch.cat(_global_fp8_buffer[amax_buffer_key])

    reduce_tensor_across_group_op_max(contiguous_amax, fp8_meta["fp8_group"])
    if reduce_amax_across_tp_group:
        reduce_tensor_across_group_op_max(contiguous_amax, tp_group)

    _global_fp8_buffer[amax_buffer_key] = list(contiguous_amax.split(chunk_sizes))


def delete_key_from_amax_buffer(forward: bool = True) -> None:
    """Delete the key from global amax buffer."""

    global _global_fp8_buffer, _buffer_delete_key_fwd, _buffer_delete_key_bwd
    if forward:
        if _buffer_delete_key_fwd is not None and _buffer_delete_key_fwd in _global_fp8_buffer:
            del _global_fp8_buffer[_buffer_delete_key_fwd]
            _buffer_delete_key_fwd = None
    else:
        if _buffer_delete_key_bwd is not None and _buffer_delete_key_bwd in _global_fp8_buffer:
            del _global_fp8_buffer[_buffer_delete_key_bwd]
            _buffer_delete_key_bwd = None
