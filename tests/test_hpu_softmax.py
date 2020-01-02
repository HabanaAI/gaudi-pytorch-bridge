import torch
import torch.nn.functional as F
import numpy as np
import pytest
torch.ops.load_library("libhabana_pytorch_plugin.so")

test_case_list = [
    # N,   H,   W,   C, dim
    pytest.param( 1,   1,   1,   1,  1, marks=pytest.mark.xfail(reason="SW-8303")),
    pytest.param( 2,   3,   4,   5,  1, marks=pytest.mark.xfail(reason="SW-8560")),
]


@pytest.mark.parametrize("N, H, W, C, dim", test_case_list)
def test_hpu_log_softmax(N, H, W, C, dim):
    hpu = torch.device('habana')
    cpu = torch.device('cpu')

    kernel = F.log_softmax
    in_tensor = torch.randn(N, C, H, W)
    hpu_result = kernel(in_tensor.to(hpu), dim=dim)
    cpu_result = kernel(in_tensor.to(cpu), dim=dim)

    # print("input", in_tensor)
    # print("result cpu", cpu_result)
    # print("result hpu", hpu_result.to(cpu))
    np.testing.assert_allclose(hpu_result.to(cpu).detach().numpy(), cpu_result.detach().numpy(), atol=0.001, rtol=1.e-3)

if __name__ == '__main__':
    test_hpu_log_softmax(*test_case_list[0])
