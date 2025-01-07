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

import os

import numpy as np
import pytest
import torch
from test_utils import _kernel_copy_to_device, compare_tensors

try:
    import habana_frameworks.torch as ht
    import habana_frameworks.torch.core as htcore
except ImportError:
    raise AssertionError("Could Not import habana_frameworks.torch.core")


class Model(torch.nn.Module):
    def __init__(self, inp_size, out_size, inner_size):
        super(Model, self).__init__()
        self.Linear1 = torch.nn.Linear(inp_size, inner_size)
        self.Linear2 = torch.nn.Linear(inner_size, out_size)
        self.h = torch.nn.ModuleList([torch.nn.Linear(inp_size, inp_size) for i in range(20)])

    def forward(self, inp):
        for i, (block) in enumerate(self.h):
            if i % 5 == 0:
                ht.core.mark_step()
            inp = block(inp)
        res = self.Linear1(inp)
        ht.core.mark_step()
        return self.Linear2(res)


@pytest.mark.skip(reason="Results mismatch")
def test_graph_training():
    # N, D_in, H, D_out = 640, 4096, 2048, 1024
    N, D_in, H, D_out = 2, 2, 2, 2
    module1_cpu = torch.nn.Linear(D_in, H).to("cpu")
    module1_hpu = _kernel_copy_to_device(module1_cpu, "hpu")
    loss_fn = torch.nn.MSELoss()
    optimizer_cpu = torch.optim.SGD(module1_cpu.parameters(), lr=0.1)
    optimizer_hpu = torch.optim.SGD(module1_hpu.parameters(), lr=0.1)
    x_cpu = torch.randn(N, D_in, device="cpu")
    x_hpu = x_cpu.to("hpu")
    module1_hpu = ht.hpu.make_graphed_callables(module1_hpu, (x_hpu,))
    real_inputs_cpu = [torch.rand_like(x_cpu) for _ in range(100)]
    real_inputs_hpu = [input.to("hpu") for input in real_inputs_cpu]
    real_targets_cpu = [torch.randn(N, D_out, device="cpu") for _ in range(100)]
    real_targets_hpu = [target.to("hpu") for target in real_targets_cpu]

    for data, target in zip(real_inputs_hpu, real_targets_hpu):
        optimizer_hpu.zero_grad(set_to_none=True)
        tmp = module1_hpu(data)
        loss_hpu = loss_fn(tmp, target)
        loss_hpu.backward()
        optimizer_hpu.step()

    for data, target in zip(real_inputs_cpu, real_targets_cpu):
        optimizer_cpu.zero_grad(set_to_none=True)
        tmp = module1_cpu(data)
        loss_cpu = loss_fn(tmp, target)
        loss_cpu.backward()
        optimizer_cpu.step()
    for _, (p, q) in enumerate(zip(module1_hpu.parameters(), module1_cpu.parameters())):
        if p.requires_grad and q.requires_grad:
            compare_tensors(p, q, atol=0.001, rtol=1.0e-3)
    compare_tensors(loss_hpu, loss_cpu, atol=0.001, rtol=1.0e-3)


input_shapes = [
    (3, 6, 4),
    (3, 8, 4),
    (3, 10, 4),
    (3, 12, 5),
    (3, 10, 6),
    (3, 10, 7),
    (3, 10, 8),
]

from test_utils import setup_teardown_env_fixture


@pytest.mark.skip(reason="Tests in this file are chaning env variables")
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_LAZY_MODE": 1, "PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 1}],
    indirect=True,
)
@pytest.mark.parametrize("shapes", input_shapes)
def test_hpu_lazy_dynamic_shape(shapes, setup_teardown_env_fixture):
    hpu = torch.device("hpu")
    for s in shapes:
        t1 = torch.randn(s, requires_grad=False)
        t2 = torch.randn(s, requires_grad=False)

        t3 = torch.add(t1, t2)
        t4 = torch.mul(t1, t2)
        t5 = torch.mul(t3, t4)
        t6 = torch.relu(t5)

        t1_h = t1.to(hpu)
        t2_h = t2.to(hpu)
        t3_h = torch.add(t1_h, t2_h)
        t4_h = torch.mul(t1_h, t2_h)
        t5_h = torch.mul(t3_h, t4_h)
        t6_h = torch.relu(t5_h)

        htcore.mark_step()
        test_graph_training()
        assert os.environ["PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES"] == "1"
        t6_h_cpu = t6_h.cpu()
        assert np.allclose(t6, t6_h_cpu, atol=0.001, rtol=1.0e-3), "Data mismatch"
