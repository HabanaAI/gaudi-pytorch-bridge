import torch
import numpy as np
import pytest
torch.ops.load_library("libhabana_pytorch_plugin.so")

test_case_list = [
   # N,  C,
   ( 2, 10,),
]

@pytest.mark.xfail(reason="SW-8819")
@pytest.mark.parametrize("N, C", test_case_list)
def test_hpu_fill(N, C):
    hpu = torch.device('habana')
    cpu = torch.device('cpu')

    cpu_tensor = torch.randn(N, C)
    hpu_tensor = cpu_tensor.to(hpu)

    cpu_tensor.fill_(2.2)
    hpu_tensor.fill_(2.2)
    hpu_tensor = hpu_tensor.to(cpu)

    print('cpu_tensor', cpu_tensor)
    print('hpu_tensor', hpu_tensor)
    np.testing.assert_allclose(hpu_result.to(cpu).detach().numpy(), cpu_result.detach().numpy(), atol=0.001, rtol=1.e-3)

if __name__ == '__main__':
    test_hpu_fill(*test_case_list[0])
