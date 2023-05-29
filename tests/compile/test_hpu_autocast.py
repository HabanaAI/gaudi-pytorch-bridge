###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import torch
import pytest
import numpy as np
from torch._dynamo.backends.common import aot_autograd

import habana_frameworks.torch.core

# Initial autocast test in PT2.0
# TODO: Add more cases when https://jira.habana-labs.com/browse/SW-125974 is fixed

def test_autocast():
    def inner_compiler(fx_module: torch.fx.GraphModule, example_inputs):
        graph = str(fx_module.code)
        assert "_to_copy = torch.ops.aten._to_copy.default(arg1_1, dtype = torch.bfloat16)" in graph
        assert "_to_copy_1 = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.bfloat16)" in graph

        return fx_module

    training_backend = aot_autograd(fw_compiler=inner_compiler, bw_compiler=inner_compiler)

    def fn(f_float32, g_float32):
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16):
            out = torch.mm(f_float32,g_float32)
        return out

    f_float32 = torch.rand((4, 8), device="cpu").to("hpu")
    g_float32 = torch.rand((8, 7), device="cpu").to("hpu")

    compiled_fn = torch.compile(fn, backend=training_backend)
    result = compiled_fn(f_float32, g_float32)

def test_convolution_autocast():
    torch.manual_seed(2562825)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Sequential(
                torch.nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=0),
            )

        def forward(self, x):
            with torch.autocast("hpu", dtype=torch.bfloat16):
                out = self.layer(x)
            return out

    torch.manual_seed(2562825)
    model = Net().to("hpu")
    compiled_model = torch.compile(model, backend="aot_hpu_inference_backend")

    torch.manual_seed(2562825)
    raw_model = Net().to("hpu")

    tensor = torch.rand(8, 1, 32, 32).to("hpu")

    res_graph = compiled_model(tensor)
    res_eager = raw_model(tensor)
    assert torch.allclose(res_eager, res_graph, rtol=1e-03)