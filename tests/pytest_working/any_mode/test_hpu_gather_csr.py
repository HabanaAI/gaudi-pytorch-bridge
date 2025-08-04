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

import pytest
import torch
from test_utils import check_ops_executed_in_jit_ir, compile_function_if_compile_mode, format_tc, is_pytest_mode_compile


def gather_csr_ref(src, indptr):
    counts = indptr[1:] - indptr[:-1]
    output = torch.repeat_interleave(src, counts)
    return output


def create_inputs(dtype):
    input_size = 10
    M = 20

    src_cpu = (torch.randn(input_size) * 10).to(dtype)
    src_hpu = src_cpu.to("hpu")

    indptr = torch.randperm(M)[: input_size + 1].sort().values
    indptr[-1] = M
    indptr[0] = 0

    indptr_cpu = indptr
    indptr_hpu = indptr_cpu.to("hpu")

    return src_cpu, src_hpu, indptr_cpu, indptr_hpu, M


@pytest.mark.parametrize("dtype", [torch.float, torch.long, torch.bfloat16, torch.int32], ids=format_tc)
def test_hpu_gather_csr(dtype):
    src_cpu, src_hpu, indptr_cpu, indptr_hpu, M = create_inputs(dtype)

    fn = compile_function_if_compile_mode(torch.ops.hpu.gather_csr)

    output_cpu = gather_csr_ref(src_cpu, indptr_cpu)
    output_hpu = fn(src_hpu, indptr_hpu, M)

    torch.testing.assert_close(output_cpu, output_hpu.cpu(), rtol=0, atol=0)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("gather_csr")


@pytest.mark.skipif(is_pytest_mode_compile(), reason="Op not supported in compile mode")
@pytest.mark.parametrize("out", [True, False])
def test_torch_scatter_gather_csr(out):
    try:
        import torch_scatter  # noqa: F401
    except ImportError:
        pytest.skip("torch_scatter is not available")

    dtype = torch.long
    src_cpu, src_hpu, indptr_cpu, indptr_hpu, M = create_inputs(dtype)

    optional_out_cpu = torch.zeros(M, dtype=dtype) if out else None
    optional_out_hpu = torch.zeros(M, dtype=dtype).to("hpu") if out else None

    output_cpu = torch_scatter.gather_csr(src_cpu, indptr_cpu, optional_out_cpu)
    output_hpu = torch_scatter.gather_csr(src_hpu, indptr_hpu, optional_out_hpu)

    torch.testing.assert_close(output_cpu, output_hpu.cpu(), rtol=0, atol=0)

    if out:
        assert optional_out_hpu.data_ptr() == output_hpu.data_ptr(), (
            "Output and out tensor should share the same storage"
        )
        torch.testing.assert_close(optional_out_cpu, optional_out_hpu.cpu(), rtol=0, atol=0)
