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

import habana_frameworks.torch.core as htcore
import numpy as np
import pytest
import torch
from test_utils import compare_tensors, hpu

matmul_lazy_list = [
    # size1, size2
    ((2), (2)),
    ((3), (3, 2)),
    ((4), (3, 4, 2)),
    ((5), (2, 3, 5, 1)),
    ((6), (2, 3, 2, 6, 2)),
    ((2, 3), (3)),
    ((3, 2, 4), (4)),
    ((2, 3, 1, 5), (5)),
    ((2, 3, 2, 2, 6), (6)),
    ((2, 1, 3, 4, 2), (1, 4, 1, 2, 3)),
    ((2, 3), (3, 4)),
    ((2, 3, 4), (4, 5)),
    ((2, 3, 4), (2, 4, 5)),
    ((2, 3, 4), (4)),
    ((2, 2, 3, 4), (2, 4, 3)),
    ((10, 8, 16), (1, 16, 12)),
    ((2, 10, 8, 16), (2, 1, 16, 12)),
]


@pytest.mark.skip(reason="Results mismatch")
@pytest.mark.parametrize("size1, size2", matmul_lazy_list)
def test_hpu_lazy_matmul_fwd_bwd(size1, size2):
    t1 = torch.randn(size1, requires_grad=True)
    t2 = torch.randn(size2, requires_grad=True)

    hpu = torch.device("hpu")

    t1_h = t1.to(hpu)
    t1_h.retain_grad()
    t2_h = t2.to(hpu)
    t2_h.retain_grad()

    out = torch.matmul(t1, t2)
    loss = out.sum()
    loss.backward()
    grad_t1_cpu = t1.grad.clone().detach()
    grad_t2_cpu = t2.grad.clone().detach()

    out_h = torch.matmul(t1_h, t2_h)
    loss_h = out_h.sum()
    loss_h.backward()

    htcore.mark_step()

    grad_t1_h = t1_h.grad.cpu()
    grad_t2_h = t2_h.grad.cpu()

    assert np.allclose(grad_t1_cpu, grad_t1_h, atol=0.001, rtol=1.0e-3), f"Data mismatch"
    assert np.allclose(grad_t2_cpu, grad_t2_h, atol=0.001, rtol=1.0e-3), f"Data mismatch"


@pytest.mark.parametrize("size1, size2", matmul_lazy_list)
def test_hpu_matmul_fwd(size1, size2):
    t1 = torch.randn(size1)
    t2 = torch.randn(size2)
    t1_h = t1.to(hpu)
    t2_h = t2.to(hpu)

    out = torch.matmul(t1, t2)
    out_h = torch.matmul(t1_h, t2_h)

    compare_tensors([out_h], [out], atol=1.0e-3, rtol=1.0e-3, assert_enable=True)


@pytest.mark.parametrize("size1, size2", matmul_lazy_list)
def test_hpu_matmul_inference(size1, size2):
    t1 = torch.randn(size1)
    t2 = torch.randn(size2)
    t1_h = t1.to(hpu)
    t2_h = t2.to(hpu)

    out_training = torch.matmul(t1_h, t2_h)
    with torch.inference_mode():
        out_inference = torch.matmul(t1_h, t2_h)

    assert torch.equal(out_training, out_inference)
