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


import os

import habana_frameworks.torch as ht
import torch
from habana_frameworks.torch.utils import split_tensor_batch


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Lin = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.Lin(x)


class SimpleMultiInputModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Lin = torch.nn.Linear(4, 4)
        self.Lin2 = torch.nn.Linear(4, 2)

    def forward(self, x, y, z):
        lin1 = self.Lin(x)
        lin2 = self.Lin2(z)
        lin2 += y
        lin3 = lin1[:, 2:] + lin2
        return lin3


class ModelWithDecorater(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Lin = torch.nn.Linear(4, 4)

    @split_tensor_batch(num_splits=4, split_index_list=[1])
    def forward(self, x):
        return self.Lin(x)


class SimpleMultiInputModelWithDecorater(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Lin = torch.nn.Linear(4, 4)
        self.Lin2 = torch.nn.Linear(4, 2)

    @split_tensor_batch(num_splits=4, split_index_list=[1, 3])
    def forward(self, x, y, z):
        lin1 = self.Lin(x)
        lin2 = self.Lin2(z)
        lin2 += y
        lin3 = lin1[:, 2:] + lin2
        return lin3


def test_split_model_input_tensor():
    torch.manual_seed(0)
    i = torch.rand(7, 4)
    torch.manual_seed(0)
    M = SimpleModel()
    cpu_out = M(i)

    MH = M.to("hpu")
    iH = i.to("hpu")
    hpu_out1 = MH(iH)

    SH = split_tensor_batch(MH, num_splits=2)
    hpu_out2 = SH(iH)

    torch.manual_seed(0)
    MH2 = ModelWithDecorater().to("hpu")
    hpu_out3 = MH2(iH)

    lazy_mode = os.getenv("PT_HPU_LAZY_MODE")
    if lazy_mode == "1":
        hpu_graph_model = ht.hpu.wrap_in_hpu_graph(MH2)
        with torch.inference_mode():
            hpu_out4 = hpu_graph_model(iH)
            hpu_out4 = hpu_graph_model(iH)
    elif lazy_mode == "0":
        hpu_compile_model = torch.compile(MH2, backend="hpu_backend")
        with torch.inference_mode():
            hpu_compile_out = hpu_compile_model(iH)

    assert torch.allclose(cpu_out, hpu_out1.to("cpu"))
    assert torch.allclose(cpu_out, hpu_out2.to("cpu"))
    assert torch.allclose(cpu_out, hpu_out3.to("cpu"))

    if lazy_mode == "1":
        assert torch.allclose(cpu_out, hpu_out4.to("cpu"))
    elif lazy_mode == "0":
        assert torch.allclose(cpu_out, hpu_compile_out.to("cpu"))


def test_split_model_multiple_input_tensor():
    torch.manual_seed(0)
    i = torch.rand(7, 4)
    j = torch.rand(7, 4)
    torch.manual_seed(0)
    M = SimpleMultiInputModel()
    cpu_out = M(i, 1, j)

    MH = M.to("hpu")
    iH = i.to("hpu")
    jH = j.to("hpu")
    hpu_out1 = MH(iH, 1, jH)

    SH = split_tensor_batch(MH, num_splits=2, split_index_list=[2, 0])
    with torch.no_grad():
        hpu_out2 = SH(iH, 1, jH)

    torch.manual_seed(0)
    MH2 = SimpleMultiInputModelWithDecorater().to("hpu")
    hpu_out3 = MH2(iH, 1, jH)

    lazy_mode = os.getenv("PT_HPU_LAZY_MODE")
    if lazy_mode == "1":
        hpu_graph_model = ht.hpu.wrap_in_hpu_graph(MH2)
        with torch.inference_mode():
            hpu_out4 = hpu_graph_model(iH, 1, jH)
            hpu_out4 = hpu_graph_model(iH, 1, jH)
    elif lazy_mode == "0":
        hpu_compile_model = torch.compile(MH2, backend="hpu_backend")
        with torch.inference_mode():
            hpu_compile_out = hpu_compile_model(iH, 1, jH)

    assert torch.allclose(cpu_out, hpu_out1.to("cpu"))
    assert torch.allclose(cpu_out, hpu_out2.to("cpu"))
    assert torch.allclose(cpu_out, hpu_out3.to("cpu"))

    if lazy_mode == "1":
        assert torch.allclose(cpu_out, hpu_out4.to("cpu"))
    elif lazy_mode == "0":
        assert torch.allclose(cpu_out, hpu_compile_out.to("cpu"))
