###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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
from test_utils import (
    is_pytest_mode_compile,
    is_pytest_mode_lazy,
)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.skipif(is_pytest_mode_lazy() or is_pytest_mode_compile(), reason="supported only in eager mode")
def test_amp_foreach(dtype):
    # Case 1
    # When passing lists with same dtypes including inf/nan
    inv_scale = torch.full((1,), 0.25, dtype=dtype)
    found_inf = torch.full((1,), 0.0, dtype=dtype)

    size = 5
    g = torch.full((size, size), 4.0, dtype=dtype)
    ginf = g.clone()
    ginf[2, 2] = float("inf")
    gnan = g.clone()
    gnan[2, 2] = float("nan")

    cases = (
        ([g.clone(), g.clone()], False),
        ([g.clone(), g.clone().t()], False),
        ([g.clone(), g.clone()[:, :5]], False),
        ([g.clone()[:, :5], g.clone()[:, :5]], False),
        ([ginf.clone(), ginf.clone()], True),
        ([g.clone(), gnan.clone()], True),
        ([g.clone(), ginf.clone()[:, :5]], True),
        ([g.clone(), gnan.clone()[:, :5]], True),
        ([ginf.clone(), g.clone()[:, :5]], True),
        ([ginf.clone()[:, :5], g.clone()[:, :5]], True),
    )

    def fn(grads, found_inf, inv_scale):
        torch._amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv_scale)
        return grads, found_inf

    def convert_to_hpu_list(cpu_list):
        hpu_list = []
        for tensor in cpu_list:
            hpu_list.append(tensor.to("hpu"))
        return hpu_list

    for grads, _ in cases:
        found_inf.zero_()
        grads_hpu = convert_to_hpu_list(grads)
        found_inf_hpu = found_inf.to("hpu")
        inv_scale_hpu = inv_scale.to("hpu")

        # CPU
        grads_out_cpu, found_inf_out_cpu = fn(grads, found_inf, inv_scale)
        # HPU
        grads_out_hpu, found_inf_out_hpu = fn(grads_hpu, found_inf_hpu, inv_scale_hpu)
        # compare the results between cpu and hpu
        assert torch.allclose(found_inf_out_cpu, found_inf_out_hpu.cpu())
        for grad_c, grad_h in zip(grads_out_cpu, grads_out_hpu, strict=False):
            assert torch.allclose(grad_c, grad_h.cpu(), rtol=1e-5, atol=1e-7, equal_nan=True)

    # Case 2
    # When passing lists with mismatched dtypes
    grads = [g.clone(), g.to(dtype=torch.float16)]
    found_inf.zero_()
    found_inf_hpu = found_inf.to("hpu")
    grads_hpu = convert_to_hpu_list(grads)
    # CPU
    grads_out_cpu, found_inf_out_cpu = fn(grads, found_inf, inv_scale)
    # HPU
    grads_out_hpu, found_inf_out_hpu = fn(grads_hpu, found_inf_hpu, inv_scale_hpu)
    # compare the results between cpu and hpu
    assert torch.allclose(found_inf_out_cpu, found_inf_out_hpu.cpu())
    for grad_c, grad_h in zip(grads_out_cpu, grads_out_hpu, strict=False):
        assert torch.allclose(grad_c, grad_h.cpu(), rtol=1e-5, atol=1e-7)
