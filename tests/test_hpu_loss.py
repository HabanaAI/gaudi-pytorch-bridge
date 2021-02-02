import torch
import torch.nn.functional as F
import pytest
from test_utils import evaluate_fwd_kernel, evaluate_fwd_bwd_kernel, reset_seed

test_case_list = [
    # N, C,
    (500, 10,),
]

@pytest.mark.parametrize("N, C", test_case_list)
@pytest.mark.parametrize("ignore_index", [-1, -100])
def test_hpu_nllloss(N, C, ignore_index):
    # TODO: extend that test to all features
    kernel = F.nll_loss
    ignore_index_val = ignore_index
    t1 = torch.randint(low=0, high=C - 1, size=(N,))
    t_in = torch.empty_like(t1)
    if ignore_index_val == -100:
        t_in = t1
    else:
        #Fill a percentage of target tensor wih ignore_index_val
        fill_thresh = round(0.3*C)
        t2 = t1 < torch.tensor(fill_thresh)
        t_in = t1.masked_fill(t2, ignore_index_val)

    kernel_params = {
        "input": torch.randn(N, C),
        "target": t_in,
        "ignore_index": ignore_index_val,
    }
    evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params)

@pytest.mark.parametrize("N, C", test_case_list)
@pytest.mark.parametrize("ignore_index", [-1, -100])
def test_hpu_nllloss_fwd_bwd(N, C, ignore_index):
    # TODO: extend that test to all features
    kernel = F.nll_loss
    ignore_index_val = ignore_index
    t1 = torch.randint(low=0, high=C - 1, size=(N,))
    t_in = torch.empty_like(t1)
    if ignore_index_val == -100:
        t_in = t1
    else:
        #Fill a percentage of target tensor wih ignore_index_val
        fill_thresh = round(0.3*C)
        t2 = t1 < torch.tensor(fill_thresh)
        t_in = t1.masked_fill(t2, ignore_index_val)

    kernel_params_fwd = {
        "input": torch.randn(N, C, requires_grad=True),
        "target": t_in,
        "ignore_index": ignore_index_val,
    }
    bwd_tensors = [torch.randn(1)]
    evaluate_fwd_bwd_kernel(
        kernel=kernel, tensor_list_bwd=bwd_tensors, kernel_params_fwd=kernel_params_fwd
    )

@pytest.mark.parametrize("N, C", test_case_list)
@pytest.mark.parametrize("mode", ["mean", "sum", "none"])
def test_hpu_mseloss(N, C, mode):
    # TODO: extend that test to all features
    kernel = F.mse_loss
    kernel_params = {
        "input": torch.randn(N, C),
        "target": torch.randn(N, C),
        "reduction": mode,
    }
    evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params)


@pytest.mark.parametrize("N, C", test_case_list)
@pytest.mark.parametrize("mode", ["mean", "sum", "none"])
def test_hpu_mseloss_fwd_bwd(N, C, mode):
    kernel = F.mse_loss
    kernel_params_fwd = {
        "input": torch.randn(N, C, requires_grad=True),
        "target": torch.randn(N, C),
        "reduction": mode,
    }
    if mode == "none":
        bwd_tensors = [torch.randn(N, C)]
    else:
        bwd_tensors = [torch.randn(1)]
    evaluate_fwd_bwd_kernel(
        kernel=kernel, tensor_list_bwd=bwd_tensors, kernel_params_fwd=kernel_params_fwd
    )


@pytest.mark.parametrize("N, C", test_case_list)
def test_hpu_bceloss(N, C):
    kernel = torch.nn.functional.binary_cross_entropy
    kernel_params = {
        "input": torch.sigmoid(torch.randn(N, 1)),
        "target": torch.randn(N, 1),
    }
    evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params)


@pytest.mark.parametrize("N, C", test_case_list)
def test_hpu_bceloss_fwd_bwd(N, C):
    # TODO: extend that test to all features
    kernel = torch.nn.functional.binary_cross_entropy
    kernel_params_fwd = {
        "input": torch.sigmoid(torch.randn(N, 1, requires_grad=True)),
        "target": torch.randn(N, 1),
    }
    bwd_tensors = [torch.randn(1)]
    evaluate_fwd_bwd_kernel(
        kernel=kernel, tensor_list_bwd=bwd_tensors, kernel_params_fwd=kernel_params_fwd
    )

if __name__ == "__main__":
    test_hpu_nllloss_fwd_bwd(*test_case_list[0])
    test_hpu_mseloss_fwd_bwd(*test_case_list[0], "none")

