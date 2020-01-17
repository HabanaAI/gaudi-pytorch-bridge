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
    kernel = F.nll_loss
    in_tensors = [torch.randn(N, C), torch.randint(low=0, high=C - 1, size=(N,))]
    evaluate_fwd_kernel(kernel, in_tensors)


@pytest.mark.parametrize("N, C", test_case_list)
def test_hpu_nllloss_fwd_bwd(N, C):
    # TODO: extend that test to all features
    kernel = F.nll_loss
    fwd_tensors = [torch.randn(N, C, requires_grad=True), torch.randint(low=0, high=C - 1, size=(N,))]
    bwd_tensors = [torch.randn(1, requires_grad=True)]
    evaluate_fwd_bwd_kernel(kernel, fwd_tensors, bwd_tensors)


if __name__ == '__main__':
    test_hpu_nllloss(*test_case_list[0])
