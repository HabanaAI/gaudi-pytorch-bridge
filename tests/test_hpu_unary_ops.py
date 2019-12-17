import torch
import torch.nn.functional as F
import numpy as np
import pytest
torch.ops.load_library("libhabana_pytorch_plugin.so")

# N - batch
# H - input height
# W - input width
# C - input channels
test_case_list = [
    #  N,   H,   W,   C,
    (  8,  24,  24,   3,),
    ( 12,  32,  32,  16,),
]

unary_op_list = [
    F.relu,
]

@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("unary_op", unary_op_list)
def test_hpu_unary_op(N, H, W, C, unary_op):
    hpu = torch.device('habana')
    cpu = torch.device('cpu')

    in_tensor = torch.randn(N, C, H, W)
    hpu_result = unary_op(in_tensor.to(hpu)).to(cpu)
    cpu_result = unary_op(in_tensor.to(cpu))

    # print("input", in_tensor)
    # print("result cpu", cpu_result)
    # print("result hpu", hpu_result)
    np.testing.assert_allclose(hpu_result.detach().numpy(), cpu_result.detach().numpy(), atol=0.001, rtol=1.e-3)

if __name__ == '__main__':
    test_hpu_unary_op(*test_case_list[0], unary_op_list[0])
