import os
import torch
import numpy as np
import pytest
torch.ops.load_library(os.path.join(os.environ['BUILD_ROOT_LATEST'], "libhabana_pytorch_plugin.so"))

# N - batch
# H - input height
# W - input width
# C - input channels
test_case_list = [
    # N, H, W, C
    (8, 28, 28, 3),
]

# @torch.jit.script
@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_conv(N, H, W, C):
    hpu = torch.device('habana')
    cpu = torch.device('cpu')

    in_tensor = torch.randn(N, C, H, W)
    hpu_result = in_tensor.to(hpu).permute((0, 2, 3, 1)).to(cpu)
    cpu_result = in_tensor.to(cpu).permute((0, 2, 3, 1))
    np.testing.assert_allclose(hpu_result.detach().numpy(), cpu_result.detach().numpy(), atol=0.001, rtol=1.e-3)

    hpu_result = hpu_result.to(hpu).permute((0, 3, 1, 2)).to(cpu)
    cpu_result = cpu_result.to(cpu).permute((0, 3, 1, 2))
    np.testing.assert_allclose(hpu_result.detach().numpy(), cpu_result.detach().numpy(), atol=0.001, rtol=1.e-3)

    np.testing.assert_allclose(hpu_result.detach().numpy(), in_tensor.detach().numpy(), atol=0.001, rtol=1.e-3)


if __name__ == '__main__':
    test_case_list(*test_case_list[1])
