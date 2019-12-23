import torch
import torch.nn.functional as F
import numpy as np
import pytest
torch.ops.load_library("libhabana_pytorch_plugin.so")

# N - batch
# H - input height
# W - input width
# C - input channels
# R - filter height
# S - filter width
# str - stride
pool_test_case_list = [
    # N,   H,   W,   C, R, S, str_H, str_W
    pytest.param( 1,   1,   1,   1, 1, 1,     1,     1, marks=pytest.mark.xfail(reason="SW-8303")),
    ( 1,   1,   1,   1, 1, 1,    1,    1),
    ( 2,   3,   4,   5, 2, 2,    1,    1),
    ( 8,  28,  28,   3, 2, 2,    1,    1),
]

@pytest.mark.parametrize("N, H, W, C, R, S, str_H, str_W", pool_test_case_list)
def test_hpu_pool(N, H, W, C, R, S, str_H, str_W):
    hpu = torch.device('habana')
    cpu = torch.device('cpu')

    pool_op = F.max_pool2d
    in_tensor = torch.randn(N, C, H, W)
    hpu_result = pool_op(in_tensor.to(hpu), kernel_size=[R,S],
        stride=[str_H, str_W], return_indices=True)
    cpu_result = pool_op(in_tensor.to(cpu), kernel_size=[R,S],
        stride=[str_H, str_W], return_indices=True)

    # print("input", in_tensor)
    # print("result cpu[0]", cpu_result[0])
    # print("result hpu[0]", hpu_result[0].to(cpu))
    # print("result cpu[1]", cpu_result[1])
    # print("result hpu[1]", hpu_result[1].to(cpu))

    np.testing.assert_allclose(hpu_result[0].to(cpu).detach().numpy(), cpu_result[0].detach().numpy(), atol=0.001, rtol=1.e-3)
    # format of indices are implementation dependant
    # np.testing.assert_array_equal(hpu_result[1].to(cpu).detach().numpy(), cpu_result[1].detach().numpy(), verbose=True)

if __name__ == '__main__':
    test_hpu_pool(*pool_test_case_list[0])
