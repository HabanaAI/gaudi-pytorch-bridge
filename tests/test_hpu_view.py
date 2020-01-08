import torch
import numpy as np
import pytest
torch.ops.load_library("libhabana_pytorch_plugin.so")

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

    hpu_result = in_tensor.to(hpu).view(-1, C*H*W).to(cpu)
    cpu_result = in_tensor.to(cpu).view(-1, C*H*W)
    np.testing.assert_array_equal(hpu_result.detach().numpy(), cpu_result.detach().numpy())

if __name__ == '__main__':
    test_case_list(*test_case_list[1])
