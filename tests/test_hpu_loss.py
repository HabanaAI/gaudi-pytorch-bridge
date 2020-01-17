import torch
import torch.nn.functional as F
import pytest
from test_utils import evaluate_fwd_kernel, evaluate_fwd_bwd_kernel, reset_seed

test_case_list = [
    # N, C,
    (500, 10,),
]


@pytest.mark.parametrize("N, C", test_case_list)
def test_hpu_nllloss(N, C):
    # TODO: extend that test to all features
    kernel_params = {'input': torch.randn(N, C),
                     'target': torch.randint(low=0, high=C - 1, size=(N,))}
    evaluate_fwd_kernel(kernel=F.nll_loss, kernel_params=kernel_params)


@pytest.mark.parametrize("N, C", test_case_list)
def test_hpu_nllloss_fwd_bwd(N, C):
    # TODO: extend that test to all features
    kernel = F.nll_loss
    kernel_params_fwd = {'input': torch.randn(N, C, requires_grad=True),
                         'target': torch.randint(low=0, high=C - 1, size=(N,))}
    bwd_tensors = [torch.randn(1)]
    evaluate_fwd_bwd_kernel(kernel=kernel, tensor_list_bwd=bwd_tensors, kernel_params_fwd=kernel_params_fwd)


if __name__ == '__main__':
    test_hpu_nllloss(*test_case_list[0])
