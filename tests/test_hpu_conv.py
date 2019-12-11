import torch
import torch.nn as nn
import numpy as np
import pytest
torch.ops.load_library("libhabana_pytorch_plugin.so")

# N - batch
# H - input height
# W - input width
# C - input channels
# R - filter height
# S - filter width
# K - output channels
# str - stride
conv_test_case_list = [
    # N,   H,   W,   C, R, S,   K, str
    ( 1,   1,   1,   1, 1, 1,   1, 1),
    pytest.param( 2,   3,   4,   5, 2, 2,   6, 1, marks=pytest.mark.xfail(reason="SW-8097")),
    pytest.param( 8,  28,  28,   3, 2, 2,  16, 1, marks=pytest.mark.xfail(reason="SW-8097")),
]

# @torch.jit.script
@pytest.mark.parametrize("N, H, W, C, R, S, K, stride", conv_test_case_list)
def test_hpu_conv(N, H, W, C, R, S, K, stride):
    hpu = torch.device('habana')
    cpu = torch.device('cpu')

    assert R == S, "filter is not square"
    conv1 = nn.Conv2d(C, K, R, stride)

    in_tensor = torch.randn(N, C, H, W)
    hpu_result = conv1.to(hpu)(in_tensor.to(hpu)).to(cpu)
    cpu_result = conv1.to(cpu)(in_tensor.to(cpu))

    # print("input", in_tensor)
    # print("weight", conv1.weight)
    # print("bias", conv1.bias)
    # print("result cpu", cpu_result)
    # print("result hpu", hpu_result)
    np.testing.assert_allclose(hpu_result.detach().numpy(), cpu_result.detach().numpy(), atol=0.001, rtol=1.e-3)

if __name__ == '__main__':
    test_hpu_conv(*conv_test_case_list[0])
