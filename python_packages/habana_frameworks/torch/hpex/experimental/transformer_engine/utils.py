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
# - Changed device to 'hpu'
# - Removed unused functions

"""Utility functions for Transformer Engine modules"""
import math
from typing import Any, Callable, Optional, Tuple

import habana_frameworks.torch as htorch
import torch


def get_default_init_method() -> Callable:
    """Weight initialization method if not provided by user"""
    return init_method_normal(0.023)


def init_method_normal(sigma: float) -> Callable:
    """Init method based on N(0, sigma)."""

    def init_(tensor: torch.Tensor) -> Callable:
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma: float, num_layers: int) -> Callable:
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor: torch.Tensor) -> Callable:
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def all_close(a: torch.Tensor, b: torch.Tensor) -> bool:
    """torch.allclose with cpu to not run into OOMs"""
    return torch.allclose(a.cpu(), b.cpu())


def print_rank_0(*args: Any) -> None:
    """print on rank 0"""
    if htorch.hpu.current_device() == 0:
        print(*args)


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> None:
    """util function to show some tensor stats"""
    if a.shape != b.shape:
        print_rank_0("Tensors have different shape")
        return
    print_rank_0(a)
    print_rank_0(b)
    max_err = torch.max(torch.abs(a - b))
    max_a = torch.max(a)
    max_b = torch.max(b)
    print_rank_0(f"max err={max_err}, max a={max_a}, max_b={max_b}")


def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"


def divide(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def validate_ctx_manager(ctx: Callable) -> None:
    """Checks if passed in object can be used as a context manager."""
    try:
        with ctx():
            pass
    except Exception as e:
        raise ValueError("Object must be a valid ctx manager") from e


def validate_rng_states_func(get_rng_tracker: Callable) -> None:
    """Checks if passed in param function has everything
    required for tensor/model and sequence parallel.
    """
    assert callable(get_rng_tracker), "get_rng_tracker is not a valid function"

    rng_tracker = None
    try:
        rng_tracker = get_rng_tracker()
    except Exception as e:
        raise RuntimeError("Cannot call get_rng_tracker function") from e

    assert hasattr(rng_tracker, "get_states") and callable(
        rng_tracker.get_states
    ), "rng_tracker object does not have valid method get_states"
    assert hasattr(rng_tracker, "set_states") and callable(
        rng_tracker.set_states
    ), "rng_tracker object does not have valid method set_states"
    assert hasattr(rng_tracker, "fork") and callable(
        rng_tracker.fork
    ), "rng_tracker object does not have valid method fork"
    validate_ctx_manager(rng_tracker.fork)


def assert_viewless_tensor(tensor: torch.Tensor, extra_msg: Optional[str] = None) -> torch.Tensor:
    """Assert that a tensor is not a view (i.e., its '._base' field is
    not set)."""
    if isinstance(tensor, list):
        return [assert_viewless_tensor(t) for t in tensor]
    if not isinstance(tensor, torch.Tensor):
        return tensor
    assert tensor._base is None, (
        f"Ensure tensor._base is None before setting tensor.data or storing "
        f"tensor to memory buffer. Otherwise, a memory leak will occur (and "
        f"likely accumulate over iterations). {extra_msg}"
    )
    return tensor


def safely_set_viewless_tensor_data(tensor: torch.Tensor, new_data_tensor: torch.Tensor) -> None:
    """Safely set tensor's '.data' field.

    Check first that the tensor is viewless (i.e., '._base' not set). If not,
    raise an exception.
    """
    extra_msg = (
        f"FYI, tensor._base has shape "
        f"{'--' if tensor._base is None else tensor._base.shape},"
        f"and new_data_tensor has shape {new_data_tensor.shape}."
    )
    assert_viewless_tensor(tensor, extra_msg=extra_msg)
    tensor.data = new_data_tensor


def cast_if_needed(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast tensor to dtype"""
    return tensor if tensor is None or tensor.dtype == dtype else tensor.to(dtype)


# Each tensor here is shape (N, ) holding all scaling
# data for a single FP8 block, e.g. LayerNormLinear
class FP8TensorMeta:
    scale = torch.Tensor()
    scale_inv = torch.Tensor()
    amax_history = torch.Tensor()
    amax_history_index = torch.Tensor()


# NOTE: Ideally FP8FwdTensors and FP8BwdTensors classes should derive from IntEnum,
# but torch.compile doesn't support enums yet.


# Used as named indices on the `scale`, `scale_inv`,
# and `amax` tensors in the `FP8TensorMeta` class.
class FP8FwdTensors:
    GEMM1_INPUT = 0
    GEMM1_WEIGHT = 1
    GEMM2_INPUT = 2
    GEMM2_WEIGHT = 3
    GEMM3_INPUT = 4
    GEMM3_WEIGHT = 5
    GEMM4_INPUT = 6
    GEMM4_WEIGHT = 7
    GEMM5_INPUT = 8
    GEMM5_WEIGHT = 9


# Used as named indices on the `scale`, `scale_inv`,
# and `amax` tensors in the `FP8TensorMeta` class.
class FP8BwdTensors:
    GRAD_OUTPUT1 = 0
    GRAD_OUTPUT2 = 1
    GRAD_OUTPUT3 = 2
    GRAD_OUTPUT4 = 3
    GRAD_OUTPUT5 = 4


def is_gaudi3():
    return htorch.hpu.get_device_name() == "GAUDI3"
