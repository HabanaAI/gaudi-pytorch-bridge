###############################################################################
# Copyright (c) 2025 Intel Corporation
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

import random

import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    compare_tensors,
    compile_function_if_compile_mode,
    cpu,
    hpu,
    is_pytest_mode_compile,
)


def block_softmax_const_max_ref(attn, block_bias, block_groups, batch_size, global_block_max, output_scale, out_dtype):
    output = torch.zeros_like(attn, dtype=attn.dtype, device=attn.device)
    block_sums = torch.zeros(*attn.shape[:-1], 1, dtype=attn.dtype, device=attn.device)
    global_sums = torch.zeros(batch_size + 1, *attn.shape[1:-1], 1, dtype=attn.dtype, device=attn.device)
    num_blocks = block_groups.shape[0]
    for n in range(num_blocks):
        group = block_groups[n]
        if group != -1:
            output[n] = attn[n] + block_bias[n]
            output[n].sub_(global_block_max)
            output[n] = output[n].exp()
            block_sums[n] = output[n].sum(dim=-1, keepdim=True)

    for n in range(num_blocks):
        group = block_groups[n]
        if group != -1:
            global_sums[group] += block_sums[n]

    for n in range(num_blocks):
        group = block_groups[n]
        if group != -1:
            output[n] = output[n] / (global_sums[group] + torch.finfo(output.dtype).tiny)
            if out_dtype == torch.float8_e4m3fn:
                output[n] = output[n] * output_scale
                output[n] = output[n].to(out_dtype)

    return output


@pytest.mark.skip(reason="Cguid not yet implemented https://jira.habana-labs.com/browse/SW-237040")
@pytest.mark.parametrize(
    "input_shape, batch_size",
    [([96, 8, 4, 1, 128], 32)],
)
@pytest.mark.parametrize("global_block_max", [0.0, 1.0, 10.0])
@pytest.mark.parametrize("output_scale", [2.0, 3.0])
@pytest.mark.parametrize("out_dtype", [None, torch.float8_e4m3fn])
@pytest.mark.parametrize("staged", [True, False])
def test_block_softmax_const_max(input_shape, batch_size, global_block_max, output_scale, out_dtype, staged):
    num_blocks = input_shape[0]
    block_size = input_shape[-1]
    block_bias_shape = (num_blocks, 1, 1, 1, block_size)
    input_dtype = torch.bfloat16
    groups_dtype = torch.int
    op = torch.ops.hpu.block_softmax_const_max if staged else torch.ops.hpu.block_softmax_const_max_not_staged

    attn = torch.rand(input_shape, dtype=input_dtype)
    block_bias = torch.rand(block_bias_shape, dtype=input_dtype)
    block_groups = torch.randint(0, batch_size, (num_blocks,), dtype=groups_dtype)
    padding_indices = random.sample(range(num_blocks), num_blocks // 6)
    block_groups[padding_indices] = -1

    attn_hpu = attn.to(hpu)
    block_bias_hpu = block_bias.to(hpu)
    block_groups_hpu = block_groups.to(hpu)

    ref_inputs = (attn, block_bias, block_groups, global_block_max, batch_size)

    # Reference output
    ref_output = block_softmax_const_max_ref(
        attn, block_bias, block_groups, batch_size, global_block_max, output_scale, out_dtype
    )

    hpu_fn = compile_function_if_compile_mode(op)

    hpu_kwargs = {}
    if output_scale is not None:
        hpu_kwargs["output_scale"] = output_scale
    if out_dtype is not None:
        hpu_kwargs["output_dtype"] = out_dtype

    hpu_output = hpu_fn(attn_hpu, block_bias_hpu, block_groups_hpu, batch_size, global_block_max, **hpu_kwargs)

    tol = 1e-2 if out_dtype == torch.float8_e4m3fn else 1e-3
    compare_tensors(ref_output, hpu_output.to(cpu), atol=tol, rtol=tol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir(op.__name__)
