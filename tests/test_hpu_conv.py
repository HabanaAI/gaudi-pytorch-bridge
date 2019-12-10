import torch
import torch.nn as nn
import numpy as np
torch.ops.load_library("libhabana_pytorch_plugin.so")
# @torch.jit.script


def test_hpu_device():
    hpu = torch.device('habana')
    cpu = torch.device('cpu')

    in_C, out_C, filter, stride = 1, 1, 1, 1
    conv1 = nn.Conv2d(in_C, out_C, filter, stride)
    in_tensor = torch.randn(1, in_C, 1, 1)

    hpu_result = conv1.to(hpu)(in_tensor.to(hpu)).to(cpu)
    cpu_result = conv1.to(cpu)(in_tensor.to(cpu))

    print("input", in_tensor)
    print("weight", conv1.weight)
    print("bias", conv1.bias)
    print("result cpu", cpu_result)
    print("result hpu", hpu_result)
    np.testing.assert_allclose(hpu_result.detach().numpy(), cpu_result.detach().numpy())

test_hpu_device()
