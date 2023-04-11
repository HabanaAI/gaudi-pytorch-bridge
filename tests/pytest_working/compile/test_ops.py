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
import os
import pytest
import habana_frameworks.torch.utils.experimental as htexp

from test_utils import generic_setup_teardown_env
@pytest.fixture(autouse=True, scope="module")
def setup_teardown_env():
    def callback():
        import habana_frameworks.torch.core as htcore
        import habana_frameworks.torch.dynamo.compile_backend

    generic_setup_teardown_env(
        temp_test_env={"PT_HPU_LAZY_MODE": 0},
        callback=callback
    )


@pytest.mark.xfail(reason="Graph compile failed. synStatus 26")
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        torch.float,
        torch.half,
        torch.int,
        torch.int16,
        torch.int8,
        torch.bool,
    ],
)
@pytest.mark.parametrize(
    "memory_format", [torch.channels_last, torch.contiguous_format]
)
@pytest.mark.parametrize(
    "torch_func", [torch.empty_like, torch.zeros_like]
)
def test_empty_and_zeros_like(dtype, memory_format, torch_func):
    requires_grad = False
    layout = torch.strided
    if (
        dtype == torch.half
        and htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi
    ):
        pytest.skip("Half is not supported on Gaudi.")

    def fn(tensor, dtype, layout, requires_grad, memory_format, torch_func):
        return torch_func(
            tensor,
            dtype=dtype,
            layout=layout,
            requires_grad=requires_grad,
            memory_format=memory_format,
        )

    tensor = torch.randn(4, 3, 2, 5)

    compiled_cpu = torch.compile(fn)
    cpu_res = compiled_cpu(tensor, dtype, layout, requires_grad, memory_format, torch_func)

    compiled_hpu = torch.compile(fn, backend="aot_hpu_training_backend")
    hpu_res = compiled_hpu(
        tensor.to("hpu"), dtype, layout, requires_grad, memory_format, torch_func
    )

    assert cpu_res.size() == hpu_res.size()
    assert cpu_res.dtype == hpu_res.dtype

@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        torch.float,
        torch.half,
        torch.int,
    ],
)
def test_expand(dtype):
    if (
        dtype == torch.half
    ):
        pytest.skip("Half is not supported for expand.")
    '''
    expand is a view op. 
    For instance, if we perform inplace update on expand o/p, 
    the expand input should also reflect the change. 
    In our design, view output are eagerized.
    To test graph flow, we need to keep expand as a graph intermediate.
    '''
    def fn(tensor, sizes):
        exp_t = tensor.expand(
            sizes
        )
        return exp_t.mul(2.0)

    tensor = torch.randn(3,1)

    compiled_cpu = torch.compile(fn)
    cpu_res = compiled_cpu(tensor, (3,4))

    compiled_hpu = torch.compile(fn, backend="aot_hpu_training_backend")
    hpu_res = compiled_hpu(
        tensor.to("hpu"), (3, 4)
    )

    assert cpu_res.size() == hpu_res.size()
    assert cpu_res.dtype == hpu_res.dtype
