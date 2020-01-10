import torch
import torch.nn as nn
import numpy as np
import pytest
torch.ops.load_library("libhabana_pytorch_plugin.so")

# Multiply matrices NxC * CxK = NxK
test_case_list = [
    # N, C, K
     pytest.param( 9, 8, 1, marks=pytest.mark.xfail(reason="SW-8820")),
    ( 1, 1, 3),
    pytest.param( 1, 2, 3, marks=pytest.mark.xfail(reason="SW-8560")),
    pytest.param( 8, 2, 3, marks=pytest.mark.xfail(reason="SW-8560")),
]

@pytest.mark.parametrize("N, C, K", test_case_list)
def test_hpu_linear(N, C, K):
    hpu = torch.device('habana')
    cpu = torch.device('cpu')

    kernel = nn.Linear(in_features = C, out_features = K, bias = True)

    in_tensor = torch.randn(N, C)
    hpu_result = kernel.to(hpu)(in_tensor.to(hpu)).to(cpu)
    cpu_result = kernel.to(cpu)(in_tensor.to(cpu))

    # print("input", in_tensor)
    # print("weight", kernel.weight)
    # print("bias", kernel.bias)
    # print("result cpu", cpu_result)
    # print("result hpu", hpu_result)
    np.testing.assert_allclose(hpu_result.detach().numpy(), cpu_result.detach().numpy(), atol=0.001, rtol=1.e-3)

@pytest.mark.skip(reason="SW-8819")
@pytest.mark.parametrize("N, C, K", test_case_list)
def test_hpu_linear_no_bias(N, C, K):
    hpu = torch.device('habana')
    cpu = torch.device('cpu')

    kernel = nn.Linear(in_features = C, out_features = K, bias = False)

    in_tensor = torch.randn(N, C)
    hpu_result = kernel.to(hpu)(in_tensor.to(hpu)).to(cpu)
    cpu_result = kernel.to(cpu)(in_tensor.to(cpu))

    # print("input", in_tensor)
    # print("weight", kernel.weight)
    # print("result cpu", cpu_result)
    # print("result hpu", hpu_result)
    np.testing.assert_allclose(hpu_result.detach().numpy(), cpu_result.detach().numpy(), atol=0.001, rtol=1.e-3)

if __name__ == '__main__':
    test_hpu_linear_no_bias(*test_case_list[-1])
