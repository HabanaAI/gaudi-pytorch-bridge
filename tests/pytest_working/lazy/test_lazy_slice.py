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


import numpy as np
import torch


def test_hpu_lazy_slice_fwd_bwd():
    t1 = torch.randn((5, 5), requires_grad=True)

    hpu = torch.device("hpu")

    t1_h = t1.detach().to(hpu)
    t1_h.requires_grad = True
    t1_h.retain_grad()

    out = t1[0:5:2, 0:2]
    out.sum().backward()
    grad_t1_cpu = t1.grad.clone().detach()
    out_h = t1_h[0:5:2, 0:2]
    out_h.sum().backward()
    grad_t1_h = t1_h.grad.cpu()

    assert np.allclose(grad_t1_cpu, grad_t1_h, atol=0, rtol=0), "Data mismatch"
