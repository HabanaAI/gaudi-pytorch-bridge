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
import pytest
import habana_frameworks.torch.dynamo.compile_backend

@pytest.mark.parametrize("shapes", [([3, 4], [3, 4]), ([3, 4], [6, 4]), ([3, 4], [2, 3, 8])])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_hpu_embedding(shapes, dtype):
    def fn(input, indices):
        return torch.embedding(input, indices)

    input_shape, indices_shape = shapes
    cpu_input = torch.rand(input_shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    max_index = input_shape[1]-1
    cpu_indices = torch.randint(low=0, high=max_index, size=indices_shape, dtype=torch.int)
    hpu_indices = cpu_indices.to("hpu")

    torch._dynamo.reset()
    cpu_compiled_fn = torch.compile(fn)
    hpu_compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

    cpu_output = cpu_compiled_fn(cpu_input, cpu_indices)
    hpu_output = hpu_compiled_fn(hpu_input, hpu_indices).cpu()
    assert torch.equal(cpu_output, hpu_output)

@pytest.mark.parametrize("shapes", [([3, 4], [3, 4]), ([3, 4], [6, 4]), ([3, 4], [2, 3, 8])])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_hpu_embedding_bwd(shapes, dtype):
    def fn(input, indices):
        embedding = torch.embedding(input, indices)
        grad = torch.ones_like(embedding)
        embedding.backward(grad)
        return input.grad

    input_shape, indices_shape = shapes
    cpu_input = torch.rand(input_shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_input.requires_grad = True
    hpu_input.requires_grad = True
    max_index = input_shape[1]-1
    cpu_indices = torch.randint(low=0, high=max_index, size=indices_shape, dtype=torch.int)
    hpu_indices = cpu_indices.to("hpu")

    torch._dynamo.reset()
    cpu_compiled_fn = torch.compile(fn)
    hpu_compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

    cpu_output = cpu_compiled_fn(cpu_input, cpu_indices)
    hpu_output = hpu_compiled_fn(hpu_input, hpu_indices).cpu()
    assert torch.equal(cpu_output, hpu_output)