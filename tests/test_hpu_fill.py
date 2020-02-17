import torch
import pytest
from test_utils import reset_seed, compare_tensors

test_case_list = [
    # N, C, fill_val,
    (2, 10, 2.2),
    (2, 10, 0.0),
]


# @pytest.mark.xfail(reason="SW-8819")
@pytest.mark.parametrize("N, C, fill_val", test_case_list)
def test_hpu_fill(N, C, fill_val):
    hpu = torch.device('habana')
    cpu = torch.device('cpu')

    cpu_tensor = torch.randn(N, C)
    hpu_tensor = cpu_tensor.to(hpu)

    cpu_tensor.fill_(fill_val)
    hpu_tensor.fill_(fill_val)
    compare_tensors(hpu_tensor, cpu_tensor, atol=0.001, rtol=1.e-3)


if __name__ == '__main__':
    test_hpu_fill(*test_case_list[0])
