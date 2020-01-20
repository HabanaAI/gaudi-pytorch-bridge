import torch
import torch.nn.functional as F
import pytest
from test_utils import evaluate_fwd_kernel, evaluate_fwd_bwd_kernel, reset_seed


test_case_list = [
    # N, H, W, C, dim
    pytest.param(2, 3, 4, 5, 1, marks=pytest.mark.xfail(reason="SW-8560")),
]


@pytest.mark.parametrize("N, H, W, C, dim", test_case_list)
def test_hpu_log_softmax(N, H, W, C, dim):
    kernel_params = {'input': torch.randn(N, C, H, W),
                     'dim': dim}
    evaluate_fwd_kernel(kernel=F.log_softmax, kernel_params=kernel_params)


@pytest.mark.parametrize("N, H, W, C, dim", test_case_list)
def test_hpu_log_softmax_fwd_bwd(N, H, W, C, dim):
    kernel_params = {'input': torch.randn(N, C, H, W, requires_grad=True),
                     'dim': dim}
    bwd_tensors = [torch.randn(N, C, H, W)]
    # TODO: after fixing fwd we can enable checking fwd results
    evaluate_fwd_bwd_kernel(kernel=F.log_softmax, tensor_list_bwd=bwd_tensors,
                            kernel_params_fwd=kernel_params, check_results_fwd=0)


if __name__ == '__main__':
    test_hpu_log_softmax(*test_case_list[0])
