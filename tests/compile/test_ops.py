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


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    os.environ["PT_HPU_LAZY_MODE"] = "0"
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.dynamo.compile_backend

    yield
    del os.environ["PT_HPU_LAZY_MODE"]


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
def test_empty_like(dtype, memory_format):
    requires_grad = False
    layout = torch.strided

    def fn(tensor, dtype, layout, requires_grad, memory_format):
        return torch.empty_like(
            tensor,
            dtype=dtype,
            layout=layout,
            requires_grad=requires_grad,
            memory_format=memory_format,
        )

    tensor = torch.randn(4, 3, 2, 5)

    compiled_cpu = torch.compile(fn)
    cpu_res = compiled_cpu(tensor, dtype, layout, requires_grad, memory_format)

    compiled_hpu = torch.compile(fn, backend="aot_hpu_training_backend")
    hpu_res = compiled_hpu(
        tensor.to("hpu"), dtype, layout, requires_grad, memory_format
    )

    assert cpu_res.size() == hpu_res.size()
    assert cpu_res.dtype == hpu_res.dtype
