import torch
import pytest
from test_utils import reset_seed, compare_tensors

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
    compare_tensors(hpu_result, cpu_result, atol=0.001, rtol=1.e-3)

if __name__ == '__main__':
    test_hpu_fill(*test_case_list[0])
