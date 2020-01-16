import torch
import torch.nn.functional as F
import pytest
from test_utils import evaluate_fwd_kernel, evaluate_fwd_bwd_kernel, reset_seed


test_case_list = [
    # N,   H,   W,   C, dim
    pytest.param( 2,   3,   4,   5,  1, marks=pytest.mark.xfail(reason="SW-8560")),
]


@pytest.mark.parametrize("N, H, W, C, dim", test_case_list)
def test_hpu_log_softmax(N, H, W, C, dim):
    in_tensors = [torch.randn(N, C, H, W)]
    evaluate_fwd_kernel(F.log_softmax, in_tensors)

if __name__ == '__main__':
    test_hpu_log_softmax(*test_case_list[0])
