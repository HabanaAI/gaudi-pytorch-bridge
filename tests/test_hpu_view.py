import torch
import numpy as np
import pytest
from test_utils import reset_seed, compare_tensors

# N - batch
# H - input height
# W - input width
# C - input channels
test_case_list = [
    # N,   H,   W,   C
    ( 8,  28,  28,   3),
]

# @torch.jit.script
@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_view(N, H, W, C):
    hpu = torch.device('habana')
    cpu = torch.device('cpu')

    in_tensor = torch.randn(N, C, H, W)

    hpu_result = in_tensor.to(hpu).view(-1, C*H*W)
    cpu_result = in_tensor.to(cpu).view(-1, C*H*W)
    compare_tensors(hpu_result, cpu_result, atol=0.001, rtol=1.e-3)

if __name__ == '__main__':
    test_case_list(*test_case_list[0])
