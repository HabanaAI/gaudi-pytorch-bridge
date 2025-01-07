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

habana = torch.device("hpu")
cpu = torch.device("cpu")


def test_norm():
    d1, d2, num, norm_type = 2, 1024, 5, 2.0
    max_norm_val = 1.0
    vec_cpu, vec_n_cpu, vec_hpu = [], [], []
    for _ in range(num):
        u = torch.rand(d1, d2)
        vec_cpu.append(u)
        vec_n_cpu.append(torch.norm(u))
        v = u.detach().to(habana)
        vec_hpu.append(v)
    n_cpu = torch.norm(torch.stack(vec_n_cpu), norm_type)

    from habana_frameworks.torch import _hpex_C

    max_norm_t = (torch.ones((1)) * max_norm_val).to(habana)
    n_hpu = _hpex_C.fused_norm(vec_hpu, max_norm_t, norm_type)

    max_norm_cpu = float(max_norm_val)
    clip_coef = max_norm_cpu / (n_cpu + 1e-6)
    if clip_coef < 1:
        for p in vec_cpu:
            p.mul_(clip_coef)
    comp = np.allclose(
        n_hpu.to(cpu).detach().numpy(),
        n_cpu.detach().numpy(),
        atol=0.001,
        rtol=0.001,
        equal_nan=True,
    )
    print("FusedNorm output match :: {}".format(comp))
    for p, q in zip(vec_hpu, vec_cpu):
        comp = np.allclose(
            p.to(cpu).detach().numpy(),
            q.detach().numpy(),
            atol=0.001,
            rtol=0.001,
            equal_nan=True,
        )
        print("FusedNorm grad param match :: {}".format(comp))
