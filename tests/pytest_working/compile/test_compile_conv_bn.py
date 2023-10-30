import torch
import torch.nn.functional as F
import pytest
import os
import numpy
import copy
torch.manual_seed(0)

from contextlib import contextmanager
from typing import Type, Dict, Any, Tuple, Iterable, Optional, List, cast

from torch.fx import symbolic_trace

import torch._dynamo as dynamo
from torch._functorch.aot_autograd import aot_module_simplified

batch_norm_test_case_list_2d = [
    # N, H, W, C
    (16, 224, 224, 3),
]

from torch import _dynamo as torchdynamo

@pytest.mark.parametrize("N, H, W, C", batch_norm_test_case_list_2d)
def test_hpu_conv_and_batch_norm_2d_fwd_compile_only(N, H, W, C):
    hpu = torch.device("hpu")
    cpu = torch.device("cpu")
    class bn(torch.nn.Module):
        def __init__(self):
            super(bn, self).__init__()
            self.conv2 = torch.nn.Conv2d(C, C, kernel_size=3, stride=1, bias=True)
            self.conv2.weight = torch.nn.Parameter(0.2 * torch.ones_like(self.conv2.weight))
            self.conv2.bias = torch.nn.Parameter(0.5 * torch.ones_like(self.conv2.bias))
            self.bn2 = torch.nn.BatchNorm2d(C)
            self.bn2.weight = torch.nn.Parameter(0.12 * torch.ones_like(self.bn2.weight))
            self.bn2.bias = torch.nn.Parameter(0.15 * torch.ones_like(self.bn2.bias))
            self.bn2.running_mean = torch.nn.Parameter(0.01 * torch.ones_like(self.bn2.running_mean))
            self.bn2.running_var = torch.nn.Parameter(0.9 * torch.ones_like(self.bn2.running_var))
            self.train(False)
            self.eval()
        def _forward_impl(self, x):
            y = self.conv2(x)
            z = self.bn2(y)
            #a = self.conv2(z)
            return z
        def forward(self, x):
            return self._forward_impl(x)

    model = bn()
    model.eval()

    x = torch.randn(N, C, H, W, dtype=torch.float32, requires_grad=False)
    torch.manual_seed(3874)
    x2 = torch.randn(N, C, H, W, dtype=torch.float32, requires_grad=False)
    print("Infer on CPU....................................", flush=True)

    with torch.no_grad():
            output = model(x)
            output2 = model(x2)
    import habana_frameworks.torch.core as htcore
    model = htcore.hpu_set_env(model)
    symbolic_traced : torch.fx.GraphModule = symbolic_trace(model)
    # High-level intermediate representation (IR) - Graph representation
    print(symbolic_traced.forward)
    model_hpu = model.to(hpu)
    x_hpu = x.to(hpu)
    x2_hpu = x2.to(hpu)
    htcore.hpu_initialize(model_hpu)
    print("Infer on HPU....................................", flush=True)
    def raw_function(tensor):
        return model_hpu(tensor)

    compiled_function = torch.compile(raw_function, backend="aot_hpu_inference_backend")
    with torch.no_grad():
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True):
            x_hpu = x_hpu.to(torch.bfloat16)
            output_hpu = compiled_function(x_hpu)
            output_hpu = output_hpu.to(torch.float32)

    with torch.no_grad():
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True):
            x2_hpu = x2_hpu.to(torch.bfloat16)
            output2_hpu = compiled_function(x2_hpu)
            output2_hpu = output2_hpu.to(torch.float32)
    output_hpu_cpu = output_hpu.to(cpu)
    output2_hpu_cpu = output2_hpu.to(cpu)
    numpy.testing.assert_allclose(
        output_hpu_cpu.detach().numpy(), output.detach().numpy(), atol=0.1, rtol=0.1
    )
    numpy.testing.assert_allclose(
        output2_hpu_cpu.detach().numpy(), output2.detach().numpy(), atol=0.1, rtol=0.1
    )
    htcore.hpu_reset_env()