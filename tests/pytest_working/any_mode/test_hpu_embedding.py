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
import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    is_gaudi1,
    is_pytest_mode_compile,
)

dtypes = [torch.float32, torch.bfloat16]
if not is_gaudi1():
    dtypes += [torch.float8_e5m2, torch.float8_e4m3fn]


@pytest.mark.parametrize("shapes", [([3, 4], [3, 4]), ([3, 4], [6, 4]), ([3, 4], [2, 3, 8])])
@pytest.mark.parametrize("dtype", dtypes)
def test_hpu_embedding(shapes, dtype):
    def fn(input, indices):
        return torch.embedding(input, indices)

    input_shape, indices_shape = shapes
    cpu_input = torch.rand(input_shape).to(dtype)
    hpu_input = cpu_input.to("hpu")
    max_index = input_shape[1] - 1
    cpu_indices = torch.randint(low=0, high=max_index, size=indices_shape, dtype=torch.int)
    hpu_indices = cpu_indices.to("hpu")

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = torch.embedding(cpu_input, cpu_indices)
    hpu_output = fn(hpu_input, hpu_indices)

    print(cpu_output)
    print(hpu_output.cpu())

    compare_tensors(hpu_output, cpu_output, atol=0.0, rtol=0.0)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("embedding")


@pytest.mark.parametrize("shapes", [([3, 4], [3, 4]), ([3, 4], [6, 4]), ([3, 4], [2, 3, 8])])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hpu_embedding_bwd(shapes, dtype):
    def fn(input, indices):
        embedding = torch.embedding(input, indices)
        grad = torch.ones_like(embedding)
        embedding.backward(grad)
        return input.grad

    input_shape, indices_shape = shapes
    cpu_input = torch.rand(input_shape).to(dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_input.requires_grad = True
    hpu_input.requires_grad = True
    max_index = input_shape[1] - 1
    cpu_indices = torch.randint(low=0, high=max_index, size=indices_shape, dtype=torch.int)
    hpu_indices = cpu_indices.to("hpu")

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        hpu_fn = torch.compile(fn, backend="hpu_backend")
    else:
        hpu_fn = fn

    cpu_output = fn(cpu_input, cpu_indices)
    hpu_output = hpu_fn(hpu_input, hpu_indices)

    compare_tensors(hpu_output, cpu_output, atol=0.0, rtol=0.0)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"embedding", "embedding_dense_backward"})
