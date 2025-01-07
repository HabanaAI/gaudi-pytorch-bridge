###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################
import pytest
import torch
import torch.nn
from test_utils import (
    check_ops_executed_in_jit_ir,
    compare_tensors,
    compile_function_if_compile_mode,
    is_gaudi1,
    is_pytest_mode_compile,
)

Verbose = False

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

    fn = compile_function_if_compile_mode(fn)

    cpu_output = torch.embedding(cpu_input, cpu_indices)
    hpu_output = fn(hpu_input, hpu_indices)

    if Verbose:
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

    hpu_fn = compile_function_if_compile_mode(fn)

    cpu_output = fn(cpu_input, cpu_indices)
    hpu_output = hpu_fn(hpu_input, hpu_indices)

    compare_tensors(hpu_output, cpu_output, atol=0.0, rtol=0.0)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"embedding", "embedding_dense_backward"})


@pytest.mark.parametrize("num_embeddings", [10])
@pytest.mark.parametrize("embedding_dim", [3])
@pytest.mark.parametrize("padding_idx", [0, None])
@pytest.mark.parametrize("bwd", [False, True])
def test_hpu_nn_embedding(num_embeddings, embedding_dim, padding_idx, bwd):
    if is_pytest_mode_compile():
        pytest.skip(
            "Output 0 of TracableCreateParameterBackward is a view and its base or another view of its base has been "
            "modified inplace. This view was created inside a custom Function (or because an input was returned as-is) "
            "and the autograd logic to handle view+inplace would override the custom backward associated with "
            "the custom Function, leading to incorrect gradients. This behavior is forbidden. You can fix this by "
            "cloning the output of the custom Function."
        )

    def fn(emb_input, device, init_weight=None):
        emb = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx, device=device)
        if init_weight is None:
            torch.nn.init.xavier_normal_(emb.weight.data)
        else:
            emb.weight.data = init_weight.data.clone()

        if not bwd:
            return emb(emb_input), emb.weight

        emb_fwd = emb(emb_input)
        grad = torch.ones_like(emb_fwd)
        emb_fwd.backward(grad)
        return emb.weight.grad, emb.weight

    def fn_cpu(emb_input, init_weight):
        return fn(emb_input, "cpu", init_weight)

    def fn_hpu(emb_input):
        return fn(emb_input, "hpu")

    fn_hpu = compile_function_if_compile_mode(fn_hpu)

    cpu_input = torch.LongTensor([[0, 1, 2, 4, 5], [4, 3, 2, 9, 0]])
    hpu_input = cpu_input.to("hpu")

    if Verbose:
        print(f"{cpu_input = }")
        print(f"{hpu_input = }")

    hpu_output, hpu_weight = fn_hpu(hpu_input)
    cpu_output, cpu_weight = fn_cpu(cpu_input, hpu_weight.cpu())

    if Verbose:
        print(f"{cpu_output = }")
        print(f"{hpu_output = }")

        print(f"{cpu_weight = }")
        print(f"{hpu_weight = }")

    compare_tensors(hpu_output.cpu(), cpu_output, atol=0.0, rtol=0.0)
    compare_tensors(hpu_weight.cpu(), cpu_weight, atol=0.0, rtol=0.0)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"embedding", "embedding_dense_backward"})
