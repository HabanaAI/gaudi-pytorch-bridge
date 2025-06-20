###############################################################################
#
#  Copyright (c) 2025 Intel Corporation
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

import math

import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    compile_function_if_compile_mode,
    cpu,
    hpu,
    is_gaudi2,
    is_pytest_mode_compile,
    is_pytest_mode_lazy,
)


def block_softmax_adjustment_ref(b_max, b_sum, groups, batch_size):
    num_blocks = b_max.shape[0]
    global_max = torch.full((batch_size, *b_max.shape[1:]), -math.inf, device=b_max.device, dtype=b_max.dtype)
    global_sum = torch.zeros((batch_size, *b_sum.shape[1:]), device=b_sum.device, dtype=b_sum.dtype)
    adjustment = torch.empty_like(b_max)

    for n in range(num_blocks):
        g = groups[n]
        new_max = torch.maximum(global_max[g], b_max[n])
        new_sum = (global_max[g] - new_max).exp() * global_sum[g] + (b_max[n] - new_max).exp() * b_sum[n]
        global_max[g] = new_max
        global_sum[g] = new_sum

    for n in range(num_blocks):
        g = groups[n]
        adjustment[n] = (b_max[n] - global_max[g]).exp() / global_sum[g]

    return adjustment


# block_maxes: 3D tensor with shape [num_blocks, kv_heads, gqa], bf16/fp32
# block_sums: 3D tensor with shape [num_blocks, kv_heads, gqa], bf16/fp32
# block_groups: 1D tensor with shape [num_blocks], int32


@pytest.mark.parametrize(
    "input_shape, batch_size",
    [([64, 8, 4, 1, 1], 32), ([512, 32, 1, 1, 1], 38), ([896, 1, 48, 1, 1], 72), ([1152, 2, 12, 1, 1], 88)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skipif(is_gaudi2(), reason="Gaudi2 not supported yet")
def test_block_softmax_adjustment(input_shape, dtype, batch_size):
    if dtype in [torch.float32] and (is_pytest_mode_compile() or is_pytest_mode_lazy()):
        pytest.skip(
            reason="https://jira.habana-labs.com/browse/SW-224619, Accuracy issues with float32 dtype in compile and lazy mode"
        )

    num_blocks = input_shape[0]
    block_maxes = torch.rand(input_shape, dtype=dtype, requires_grad=False)
    block_sums = torch.rand(input_shape, dtype=dtype, requires_grad=False)
    block_groups = torch.randint(0, batch_size, (num_blocks,), dtype=torch.long)

    block_maxes_hpu = block_maxes.to(hpu)
    block_sums_hpu = block_sums.to(hpu)
    block_groups_hpu = block_groups.to(hpu)

    # Reference output
    ref_output = block_softmax_adjustment_ref(block_maxes, block_sums, block_groups, batch_size)

    def hpu_fn(block_maxes_hpu, block_sums_hpu, block_groups_hpu, batch_size):
        return torch.ops.hpu.block_softmax_adjustment(block_maxes_hpu, block_sums_hpu, block_groups_hpu, batch_size)

    # Compile mode
    hpu_fn = compile_function_if_compile_mode(hpu_fn)
    hpu_output = hpu_fn(block_maxes_hpu, block_sums_hpu, block_groups_hpu, batch_size)
    assert torch.allclose(ref_output, hpu_output.to(cpu), atol=0.01, rtol=0.01)

    # Check ops executed in JIT IR
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("block_softmax_adjustment")
