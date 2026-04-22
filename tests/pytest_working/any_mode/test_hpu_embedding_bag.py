###############################################################################
# Copyright (c) 2025-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    compile_function_if_compile_mode,
    cpu,
    format_tc,
    hpu,
    is_gaudi2,
    is_gaudi3,
    is_pytest_mode_compile,
)

param_dtype = [torch.bfloat16, torch.float]
if is_gaudi2() or is_gaudi3():
    param_dtype.append(torch.float16)

atol = {torch.bfloat16: 1e-2, torch.float16: 1e-3, torch.float: 1e-8}
rtol = {torch.bfloat16: 1e-2, torch.float16: 1e-3, torch.float: 1e-5}

mode_str_to_int_map = {
    "sum": 0,
    "mean": 1,
    "max": 2,
}


@pytest.mark.parametrize("mode", ["sum", "mean", "max"])
@pytest.mark.parametrize("include_last_offset", [True, False])
@pytest.mark.parametrize("use_per_sample_weights", [True, False])
@pytest.mark.parametrize("padding_idx", [-1, 3])
@pytest.mark.parametrize("dtype", param_dtype, ids=format_tc)
def test_hpu_embedding_bag(mode, include_last_offset, use_per_sample_weights, padding_idx, dtype):
    if use_per_sample_weights and mode != "sum":
        pytest.skip("per_sample_weights are relevant only in sum mode.")

    input_shape = [12, 2]

    def fn(weight, indices, offsets, per_sample_weights, mode, include_last_offset, padding_idx):
        return torch.ops.aten._embedding_bag(
            weight=weight,
            indices=indices,
            offsets=offsets,
            scale_grad_by_freq=False,
            sparse=False,
            per_sample_weights=per_sample_weights,
            mode=mode_str_to_int_map[mode],
            include_last_offset=include_last_offset,
            padding_idx=padding_idx,
        )

    fn_cpu = fn
    fn_hpu = compile_function_if_compile_mode(fn)

    bags = [[0, 5, 8, 10], [3, 3], [1, 3, 2, 7], [3, 6, 7, 8, 9, 10]]
    indices = torch.concat([torch.tensor(bag, dtype=torch.int32) for bag in bags])
    indices_count = sum([len(bag) for bag in bags])

    bag_lengths = torch.tensor([len(x) for x in bags], dtype=torch.int32)
    offsets = torch.cumsum(bag_lengths, 0, dtype=torch.int32)
    offsets = torch.concat([torch.tensor([0], dtype=torch.int32), offsets])
    if not include_last_offset:
        offsets = offsets[:-1]

    per_sample_weights = None
    if use_per_sample_weights:
        per_sample_weights = torch.randn([indices_count], dtype=dtype)

    per_sample_weights_cpu = per_sample_weights

    weight_cpu = torch.randn(*input_shape, dtype=dtype) - 0.5
    input_cpu = indices
    offsets_cpu = offsets

    weight_hpu = weight_cpu.to(hpu)
    input_hpu = input_cpu.to(hpu)
    offsets_hpu = offsets_cpu.to(hpu)
    per_sample_weights_hpu = per_sample_weights_cpu.to(hpu) if per_sample_weights_cpu is not None else None

    res_cpu = fn_cpu(weight_cpu, input_cpu, offsets_cpu, per_sample_weights_cpu, mode, include_last_offset, padding_idx)
    res_hpu = fn_hpu(weight_hpu, input_hpu, offsets_hpu, per_sample_weights_hpu, mode, include_last_offset, padding_idx)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("_embedding_bag", verbose=False)

    for i, (r_cpu, r_hpu) in enumerate(zip(res_cpu, res_hpu, strict=False)):
        # Second output which is offset2bag is not needed when mode is sum and cpu makes it a ZST. Yet, the framework
        # requires that on other devices than CPU, that tensor isn't empty. So skip verification of the output in this
        # case as CPU doesn't provide correct reference.
        if i == 1 and mode == "sum":
            continue
        # For some unknown reason, in case of compile, the meta function for non CPU devices doesn't take into account
        # the include_last_offset param. As a result, second tensor, which is bag_sizes, has always one additional
        # element. For the sake of verification, it is trimmed to obtain tensor with correct size.
        elif i == 2 and is_pytest_mode_compile() and include_last_offset:
            r_hpu = r_hpu[:-1]
        # Fourth output which is max_indices is valid only in max mode
        elif i == 3 and mode != "max":
            continue
        assert torch.allclose(r_cpu, r_hpu.cpu(), atol=atol[dtype], rtol=rtol[dtype])


@pytest.mark.parametrize("mode", ["sum", "mean", "max"])
@pytest.mark.parametrize("scale_grad_by_freq", [True, False])
@pytest.mark.parametrize("include_last_offset", [True, False])
@pytest.mark.parametrize("use_per_sample_weights", [True, False])
@pytest.mark.parametrize("padding_idx", [-1, 3])
@pytest.mark.parametrize("dtype", param_dtype, ids=format_tc)
def test_hpu_embedding_bag_fwd_bwd(
    mode, scale_grad_by_freq, include_last_offset, use_per_sample_weights, padding_idx, dtype
):
    if scale_grad_by_freq:
        pytest.skip("scale_grad_by_freq=True is not correctly handled on CPU, so reference is incorrect")

    if use_per_sample_weights and mode != "sum":
        pytest.skip("per_sample_weights are relevant only in sum mode.")

    input_shape = [12, 5]

    def fn(shape, mode, device, _weight, include_last_offset, input, offsets, per_sample_weights):
        embedding_bag = torch.nn.EmbeddingBag(
            *shape, mode=mode, device=device, _weight=_weight, include_last_offset=include_last_offset
        )
        emb = embedding_bag(input, offsets, per_sample_weights)
        grad = torch.ones_like(emb)

        emb.backward(grad)
        return emb, embedding_bag.weight.grad

    fn_cpu = fn
    fn_hpu = compile_function_if_compile_mode(fn)

    bags = [[1, 5, 8, 10], [3, 3], [0, 2, 4], [3, 6, 7, 9, 11]]
    indices = torch.concat([torch.tensor(bag, dtype=torch.int32) for bag in bags])

    indices_count = sum([len(bag) for bag in bags])

    bag_lengths = torch.tensor([len(x) for x in bags], dtype=torch.int32)
    offsets = torch.cumsum(bag_lengths, 0, dtype=torch.int32)
    offsets = torch.concat([torch.tensor([0], dtype=torch.int32), offsets])
    if not include_last_offset:
        offsets = offsets[:-1]

    per_sample_weights = None
    if use_per_sample_weights:
        per_sample_weights = torch.randn([indices_count], dtype=dtype)

    per_sample_weights_cpu = per_sample_weights

    weight_cpu = torch.randn(*input_shape, dtype=dtype)
    input_cpu = indices
    offsets_cpu = offsets

    weight_hpu = weight_cpu.to(hpu)
    input_hpu = input_cpu.to(hpu)
    offsets_hpu = offsets_cpu.to(hpu)
    per_sample_weights_hpu = per_sample_weights_cpu.to(hpu) if per_sample_weights_cpu is not None else None

    res_cpu, grad_cpu = fn_cpu(
        input_shape, mode, cpu, weight_cpu, include_last_offset, input_cpu, offsets_cpu, per_sample_weights_cpu
    )
    res_hpu, grad_hpu = fn_hpu(
        input_shape, mode, hpu, weight_hpu, include_last_offset, input_hpu, offsets_hpu, per_sample_weights_hpu
    )

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"_embedding_bag_backward", "_embedding_bag"}, verbose=False)

    assert torch.allclose(res_cpu, res_hpu.cpu(), atol=atol[dtype], rtol=rtol[dtype])
    assert torch.allclose(grad_cpu, grad_hpu.cpu(), atol=atol[dtype], rtol=rtol[dtype])
