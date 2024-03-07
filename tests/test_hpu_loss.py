import pytest
import torch
import torch.nn.functional as F
from test_utils import evaluate_fwd_bwd_kernel, evaluate_fwd_kernel

test_case_list = [
    # N, C,
    (
        500,
        10,
    ),
]

test_case_list1 = [
    # N, C, beta
    (500, 10, 1),
]

test_case_list_4d = [
    # N, C, H, W
    (5, 3, 2, 4),
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
        # Fill a percentage of target tensor wih ignore_index_val
        fill_thresh = round(0.3 * C)
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
        # Fill a percentage of target tensor wih ignore_index_val
        fill_thresh = round(0.3 * C)
        t2 = t1 < torch.tensor(fill_thresh)
        t_in = t1.masked_fill(t2, ignore_index_val)

    kernel_params_fwd = {
        "input": torch.randn(N, C, requires_grad=True),
        "target": t_in,
        "ignore_index": ignore_index_val,
    }
    bwd_tensors = [torch.randn(1)]
    evaluate_fwd_bwd_kernel(kernel=kernel, tensor_list_bwd=bwd_tensors, kernel_params_fwd=kernel_params_fwd)


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
    evaluate_fwd_bwd_kernel(kernel=kernel, tensor_list_bwd=bwd_tensors, kernel_params_fwd=kernel_params_fwd)


@pytest.mark.parametrize("N, C, beta", test_case_list1)
@pytest.mark.parametrize("mode", ["mean", "sum", "none"])
def test_hpu_smooth_l1_loss(N, C, beta, mode):
    # TODO: extend that test to all features
    kernel = F.smooth_l1_loss
    kernel_params = {
        "input": torch.randn(N, C),
        "target": torch.randn(N, C),
        "reduction": mode,
        "beta": beta,
    }
    evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params)


@pytest.mark.parametrize("N, C, beta", test_case_list1)
@pytest.mark.parametrize("mode", ["mean", "sum", "none"])
def test_hpu_smooth_l1_loss_fwd_bwd(N, C, beta, mode):
    # TODO: extend that test to all features
    kernel = F.smooth_l1_loss
    kernel_params_fwd = {
        "input": torch.randn(N, C, requires_grad=True),
        "target": torch.randn(N, C),
        "reduction": mode,
        "beta": beta,
    }
    if mode == "none":
        bwd_tensors = [torch.randn(N, C)]
    else:
        bwd_tensors = [torch.ones(1)]
    evaluate_fwd_bwd_kernel(kernel=kernel, tensor_list_bwd=bwd_tensors, kernel_params_fwd=kernel_params_fwd)


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
    evaluate_fwd_bwd_kernel(kernel=kernel, tensor_list_bwd=bwd_tensors, kernel_params_fwd=kernel_params_fwd)


@pytest.mark.parametrize("N, C, H, W", test_case_list_4d)
def test_hpu_bceloss4d(N, C, H, W):
    kernel = torch.nn.functional.binary_cross_entropy
    kernel_params = {
        "input": torch.sigmoid(torch.randn(N, C, H, W)),
        "target": torch.randn(N, C, H, W),
    }
    evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params)


@pytest.mark.parametrize("N, C, H, W", test_case_list_4d)
def test_hpu_bceloss_fwd_bwd4d(N, C, H, W):
    kernel = torch.nn.functional.binary_cross_entropy
    kernel_params_fwd = {
        "input": torch.sigmoid(torch.randn(N, C, H, W, requires_grad=True)),
        "target": torch.randn(N, C, H, W),
    }
    bwd_tensors = [torch.randn(1)]
    evaluate_fwd_bwd_kernel(kernel=kernel, tensor_list_bwd=bwd_tensors, kernel_params_fwd=kernel_params_fwd)


@pytest.mark.parametrize("N, C, H, W", test_case_list_4d)
@pytest.mark.parametrize("mode", ("sum", "mean"))
def test_hpu_bcelogitsloss_fwd_bwd(N, C, H, W, mode):
    kernel = torch.nn.functional.binary_cross_entropy_with_logits
    kernel_params_fwd = {
        "input": torch.randn(N, C, H, W, requires_grad=True),
        "target": torch.randn(N, C, H, W, requires_grad=True),
        "reduction": mode,
    }
    if mode == "none":
        bwd_tensors = [torch.randn(N, C, H, W)]
    else:
        bwd_tensors = [torch.randn(1)]
    evaluate_fwd_bwd_kernel(kernel=kernel, tensor_list_bwd=bwd_tensors, kernel_params_fwd=kernel_params_fwd)


# test for channelslast & reduction=none will be added in next patch
@pytest.mark.parametrize("N, C, H, W", [(14, 4, 192, 160)])
@pytest.mark.parametrize("ignore_index", [-100, 0])
def test_hpu_nllloss2d(N, C, H, W, ignore_index):
    # TODO: extend that test to all features
    kernel = F.nll_loss
    ignore_index_val = ignore_index
    t1 = torch.randint(low=0, high=C - 1, size=(N, H, W))
    t_in = torch.empty_like(t1)
    if ignore_index_val == -100:
        t_in = t1
    else:
        # Fill a percentage of target tensor wih ignore_index_val
        fill_thresh = round(0.3 * C)
        t2 = t1 < torch.tensor(fill_thresh)
        t_in = t1.masked_fill(t2, ignore_index_val)

    kernel_params = {
        "input": torch.randn(N, C, H, W),
        "target": t_in,
        "ignore_index": ignore_index_val,
        "reduction": "mean",
    }
    evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params)


@pytest.mark.parametrize("N, C, H, W", [(14, 4, 192, 160)])
@pytest.mark.parametrize("ignore_index", [-100, 0])
def test_hpu_nllloss2d_fwd_bwd(N, C, H, W, ignore_index):
    # TODO: extend that test to all features
    kernel = F.nll_loss
    ignore_index_val = ignore_index
    t1 = torch.randint(low=0, high=C - 1, size=(N, H, W))
    t_in = torch.empty_like(t1)
    if ignore_index_val == -100:
        t_in = t1
    else:
        # Fill a percentage of target tensor wih ignore_index_val
        fill_thresh = round(0.3 * C)
        t2 = t1 < torch.tensor(fill_thresh)
        t_in = t1.masked_fill(t2, ignore_index_val)

    kernel_params_fwd = {
        "input": torch.randn(N, C, H, W, requires_grad=True),
        "target": t_in,
        "ignore_index": ignore_index_val,
    }
    bwd_tensors = [torch.randn(1)]
    evaluate_fwd_bwd_kernel(kernel=kernel, tensor_list_bwd=bwd_tensors, kernel_params_fwd=kernel_params_fwd)


@pytest.mark.parametrize("N, C, H, W", [(32, 81, 8, 1091)])
def test_hpu_crossentropyloss_4d_fwd(N, C, H, W):
    kernel = torch.nn.CrossEntropyLoss(reduction="none")
    input = torch.randn(N, C, H, W, requires_grad=True)
    target = torch.randint(low=0, high=C - 1, size=(N, H, W))
    kernel_params = {"input": input, "target": target}

    evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params)


@pytest.mark.parametrize("N, C, H", [(32, 81, 8732)])
def test_hpu_crossentropyloss_fwd(N, C, H):
    kernel = torch.nn.CrossEntropyLoss(reduction="none")
    input = torch.randn(N, C, H, requires_grad=True)
    target = torch.randint(low=0, high=C - 1, size=(N, H))
    kernel_params = {"input": input, "target": target}

    evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params)


@pytest.mark.parametrize("N, C", [(32, 81)])
def test_hpu_crossentropyloss_1d_fwd(N, C):
    kernel = torch.nn.CrossEntropyLoss(reduction="none")
    input = torch.randn(N, C, requires_grad=True)
    target = torch.randint(low=0, high=C - 1, size=(N,))
    kernel_params = {"input": input, "target": target}

    evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params)


if __name__ == "__main__":
    test_hpu_nllloss_fwd_bwd(*test_case_list[0])
    test_hpu_mseloss_fwd_bwd(*test_case_list[0], "none")
