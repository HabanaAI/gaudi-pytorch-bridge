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


import pytest
import torch
import torch.nn.functional as F
from habana_frameworks.torch.hpex.kernels import FusedSDPA
from test_utils import compile_function_if_compile_mode, is_pytest_mode_lazy


@pytest.mark.skipif(
    is_pytest_mode_lazy(),
    reason="we will skip the lazy mode test.",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_aten_SDPA_fwd_only(dtype):
    torch.manual_seed(1234)

    def fn(query, key, value):
        x = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
        )
        return x

    def gn(query, key, value):
        x = FusedSDPA.apply(query, key, value)
        return x

    # CPU
    cpu_query = torch.rand(2, 8, 128, 64, dtype=torch.bfloat16)
    cpu_key = torch.rand(2, 8, 128, 64, dtype=torch.bfloat16)
    cpu_value = torch.rand(2, 8, 128, 64, dtype=torch.bfloat16)

    fn = compile_function_if_compile_mode(fn)
    gn = compile_function_if_compile_mode(gn)

    # HPU with Fused SDPA python autograd interface
    hpu_query_g = cpu_query.to("hpu")
    hpu_key_g = cpu_key.to("hpu")
    hpu_value_g = cpu_value.to("hpu")
    hpu_result_g = gn(hpu_query_g, hpu_key_g, hpu_value_g)

    # HPU with Fused SDPA cpp autograd interface
    hpu_query_f = cpu_query.to("hpu")
    hpu_key_f = cpu_key.to("hpu")
    hpu_value_f = cpu_value.to("hpu")
    hpu_result_f = fn(hpu_query_f, hpu_key_f, hpu_value_f)

    tolerance = 5e-2 if dtype == torch.bfloat16 else 1e-4
    assert torch.allclose(hpu_result_g.cpu(), hpu_result_f.cpu(), atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(
    is_pytest_mode_lazy(),
    reason="we will skip the lazy mode test.",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_aten_SDPA_fwd_bwd(dtype):
    torch.manual_seed(1234)

    def fn(query, key, value):
        x = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
        )
        return x

    def gn(query, key, value):
        x = FusedSDPA.apply(query, key, value)
        return x

    # CPU
    cpu_query = torch.rand(2, 8, 128, 64, dtype=torch.bfloat16)
    cpu_key = torch.rand(2, 8, 128, 64, dtype=torch.bfloat16)
    cpu_value = torch.rand(2, 8, 128, 64, dtype=torch.bfloat16)

    if pytest.mode == "compile":
        fn = torch.compile(fn, backend="hpu_backend")
        gn = torch.compile(gn, backend="hpu_backend")

    # HPU with Fused SDPA python autograd interface
    hpu_query_g = cpu_query.detach().to("hpu")
    hpu_key_g = cpu_key.to("hpu")
    hpu_value_g = cpu_value.to("hpu")

    hpu_query_g.requires_grad_(True)
    hpu_key_g.requires_grad_(True)
    hpu_value_g.requires_grad_(True)

    hpu_result_g = gn(hpu_query_g, hpu_key_g, hpu_value_g)

    # HPU with Fused SDPA cpp autograd interface
    hpu_query_f = cpu_query.detach().to("hpu")
    hpu_key_f = cpu_key.to("hpu")
    hpu_value_f = cpu_value.to("hpu")

    hpu_query_f.requires_grad_(True)
    hpu_key_f.requires_grad_(True)
    hpu_value_f.requires_grad_(True)

    hpu_result_f = fn(hpu_query_f, hpu_key_f, hpu_value_f)

    tolerance = 5e-2 if dtype == torch.bfloat16 else 1e-4
    assert torch.allclose(hpu_result_g.detach().cpu(), hpu_result_f.detach().cpu(), atol=tolerance, rtol=tolerance)

    hpu_result_g.sum().backward()
    hpu_result_f.sum().backward()

    assert torch.allclose(
        hpu_query_g.grad.detach().cpu(), hpu_query_f.grad.detach().cpu(), atol=tolerance, rtol=tolerance
    )
    assert torch.allclose(hpu_key_g.grad.detach().cpu(), hpu_key_f.grad.detach().cpu(), atol=tolerance, rtol=tolerance)
    assert torch.allclose(
        hpu_value_g.grad.detach().cpu(), hpu_value_f.grad.detach().cpu(), atol=tolerance, rtol=tolerance
    )
