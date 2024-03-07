from typing import Tuple

import numpy as np
import pytest
import torch
from test_utils import cpu, hpu, is_gaudi1


def util_calc_l2_error(t1, t2) -> torch.tensor:
    assert t1.numel() == t2.numel()
    err = torch.sqrt(torch.abs((t1 - t2) * (t1 - t2)).sum())
    return err / t1.numel()


def batch_norm_stats_ref(inp, eps):
    mean = torch.mean(inp, (0, 2, 3))
    var = torch.var(inp, (0, 2, 3), unbiased=True)
    invstd = 1 / torch.sqrt(var + eps)
    return (mean, invstd)


def batch_norm_elemt_ref(inp, weight, bias, mean, invstd, eps):
    C = inp.shape[1]
    gamma = torch.ones(C, dtype=torch.float) if weight is None else weight
    beta = torch.zeros(C, dtype=torch.float) if bias is None else bias

    mean_reshaped = mean.view(1, mean.shape[0], 1, 1)
    invstd_reshaped = invstd.view(1, invstd.shape[0], 1, 1)
    beta_reshaped = beta.view(1, beta.shape[0], 1, 1)
    gamma_reshaped = gamma.view(1, beta.shape[0], 1, 1)

    out = (inp - mean_reshaped) * invstd_reshaped * gamma_reshaped + beta_reshaped

    return out


def batch_norm_backward_elemt_ref(grad_out, inp, mean, invstd, weight, sum_dy, sum_dy_xmu, counts) -> torch.tensor:
    mean_reshaped = mean.reshape(1, mean.shape[0], 1, 1)
    invstd_reshaped = invstd.reshape(1, invstd.shape[0], 1, 1)
    sum_dy_reshaped = sum_dy.reshape(1, sum_dy.shape[0], 1, 1)
    sum_dy_xmu_reshaped = sum_dy_xmu.reshape(1, sum_dy_xmu.shape[0], 1, 1)

    total_count = torch.sum(counts)
    factor_2_c = (1 / invstd_reshaped) if weight is None else (weight * invstd_reshaped)
    factor_1_c = sum_dy_xmu_reshaped * invstd_reshaped * invstd_reshaped / total_count

    grad_in = (grad_out - (sum_dy_reshaped / total_count) - (inp - mean_reshaped) * factor_1_c) * factor_2_c

    return grad_in


def batch_norm_backward_reduce_ref(
    grad_out, inp, mean, invstd, weight, b_inp_grad, b_wei_grad, b_bias_grad
) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    grad_out_reshaped = grad_out.reshape(
        grad_out.shape[0], grad_out.shape[1], -1
    )  # collapse H,W or H,W,K dims into one
    inp_reshaped = inp.reshape(inp.shape[0], inp.shape[1], -1)
    mean_reshaped = mean.reshape(1, mean.shape[0], 1)
    invstd_reshaped = invstd.reshape(1, invstd.shape[0], 1)

    sum_dy = torch.sum(grad_out_reshaped, (0, 2)) if b_inp_grad else None

    dy_xmu = grad_out_reshaped * (inp_reshaped - mean_reshaped)
    sum_dy_xmu = torch.sum(dy_xmu, (0, 2)) if b_inp_grad else None

    wei_term = dy_xmu * invstd_reshaped
    grad_wei = torch.sum(wei_term, (0, 2)) if b_wei_grad else None

    grad_bias = torch.sum(grad_out_reshaped, (0, 2)) if b_bias_grad else None

    return (sum_dy, sum_dy_xmu, grad_wei, grad_bias)


def batch_norm_gather_stats_with_counts_ref(
    mean_tensor, invstds_tensor, count_tensor, running_mean, running_var, momentum, eps
) -> torch.tensor:
    # TPC kernel reshape
    counts_reshaped = count_tensor.view(-1, 1)

    # TPC kernel cumsum
    counts_accum_inclusive = torch.cumsum(counts_reshaped, dim=0)

    # TPC kernel cumsum with exclusive flag = 1
    counts_accum_exclusive = counts_accum_inclusive - counts_reshaped

    # TPC kernel mult
    mean_times_counts = counts_reshaped * mean_tensor

    # TPC kernel reciprocal
    one_div_counts_accum_inclusive = 1.0 / counts_accum_inclusive

    # TPC kernel mult
    partial_mean = torch.cumsum(mean_times_counts, dim=0) * one_div_counts_accum_inclusive

    # TPC kernel cumsum with exclusive flag = 1
    tmp_partial_mean = partial_mean.roll(1, 0)
    tmp_partial_mean[0] = 0

    # TPC kernels sub, mult
    second_term = (
        (tmp_partial_mean - mean_tensor)
        * (tmp_partial_mean - mean_tensor)
        * counts_accum_exclusive
        * counts_reshaped
        * one_div_counts_accum_inclusive
    )
    # TPC kernel reciprocal
    v = 1 / invstds_tensor

    # TPC kernels mult, sub
    v = (v * v - eps) * counts_reshaped

    # TPC kernel cumsum
    first_term = torch.cumsum(v, dim=0)

    # TPC kernel add
    partial_var = first_term + second_term
    # TPC kernel rsqrt
    g_invstd = 1 / torch.sqrt(partial_var[-1] / counts_accum_inclusive[-1] + eps)
    # Update running mean
    if running_mean is not None:
        running_mean = ((1 - momentum) * running_mean) + (momentum * partial_mean[-1])
    # Update running var
    unbiasedVar = partial_var[-1] / (torch.sum(count_tensor) - 1)
    if running_var is not None:
        running_var = ((1 - momentum) * running_var) + (momentum * unbiasedVar)


@pytest.mark.xfail
@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported dtype")
def test_all():
    num_devices = 4
    N = 8
    C = 3
    H = 5
    W = 5
    atol = 1e-4
    rtol = 1e-4

    # Input tensor
    inp = torch.rand(N, C, H, W)
    inp_hpu = inp.to(hpu)
    print(f"Input shape: " + str(inp.shape))

    running_mean = torch.rand(C)
    running_mean_hpu = running_mean.to(hpu)
    running_var = torch.rand(C)
    running_var_hpu = running_var.to(hpu)
    momentum = 0.1

    print("Forward pass")
    # Input tensor split across num_devices
    inp_split_list = torch.split(inp, N // num_devices, 0)
    inp_split_list_hpu = torch.split(inp_hpu, N // num_devices, 0)

    # Compute mean and invstd for each device
    mean_list = []
    invstd_list = []
    mean_list_hpu = []
    invstd_list_hpu = []

    for i in range(len(inp_split_list)):
        # CPU
        dev_mean, dev_invstd = batch_norm_stats_ref(inp_split_list[i], 1e-5)
        mean_list.append(dev_mean)
        invstd_list.append(dev_invstd)

        # hpu
        dev_mean_hpu, dev_invstd_hpu = torch.batch_norm_stats(inp_split_list_hpu[i], 1e-5)
        mean_list_hpu.append(dev_mean_hpu)
        invstd_list_hpu.append(dev_invstd_hpu)

    # Verify CPU and hpu results
    print("-----\nComparing outputs for batch_norm_stats stage")
    for i in range(len(inp_split_list)):
        cpu_mean_tensor = mean_list[i]
        hpu_mean_tensor = mean_list_hpu[i]
        err_mean = np.allclose(
            hpu_mean_tensor.to(cpu).detach().numpy(),
            cpu_mean_tensor.detach().numpy(),
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        )

        cpu_invstd_tensor = invstd_list[i]
        hpu_invstd_tensor = invstd_list_hpu[i]
        err_invstd = np.allclose(
            hpu_invstd_tensor.to(cpu).detach().numpy(),
            cpu_invstd_tensor.detach().numpy(),
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        )

        print(f"device {i}: mean allclose = {err_mean}, invstd allclose = {err_invstd}")

    print("-----")
    # Gather statistics from across devices
    # CPU
    counts = torch.full([num_devices], inp.numel() // num_devices, dtype=torch.float)
    g_mean, g_invstd, running_mean, running_var = batch_norm_gather_stats_with_counts_ref(
        torch.stack(mean_list), torch.stack(invstd_list), counts, running_mean, running_var, momentum, 1e-5
    )

    # HPU
    counts_hpu = counts.to(hpu)
    g_mean_hpu, g_invstd_hpu = torch.batch_norm_gather_stats_with_counts(
        inp_hpu,
        torch.stack(mean_list_hpu),
        torch.stack(invstd_list_hpu),
        running_mean_hpu,
        running_var_hpu,
        momentum,
        1e-5,
        counts_hpu,
    )
    # Verify runnig mean and variance
    err_mean = np.allclose(
        running_mean_hpu.to(cpu).detach().numpy(), running_mean.detach().numpy(), atol=atol, rtol=rtol, equal_nan=True
    )
    err_invstd = np.allclose(
        running_var_hpu.to(cpu).detach().numpy(), running_var.detach().numpy(), atol=atol, rtol=rtol, equal_nan=True
    )
    print("Comparing results for batch_norm_gather_stats stage - Running mean and Running Variance")
    print(f"Running Mean allclose = {err_mean}, Running Variance allclose = {err_invstd}")

    # Verify CPU and HPU results
    err_mean = np.allclose(
        g_mean_hpu.to(cpu).detach().numpy(), g_mean.detach().numpy(), atol=atol, rtol=rtol, equal_nan=True
    )
    err_invstd = np.allclose(
        g_invstd_hpu.to(cpu).detach().numpy(), g_invstd.detach().numpy(), atol=atol, rtol=rtol, equal_nan=True
    )
    print("Comparing results for batch_norm_gather_stats stage")
    print(f"g_mean error allclose = {err_mean}, g_invstd error allclose = {err_invstd}")
    print("-----")

    # Apply elementwise normalization
    out_list = []
    out_list_hpu = []
    for i in range(len(inp_split_list)):
        out = batch_norm_elemt_ref(inp_split_list[i], None, None, g_mean, g_invstd, 1e-5)
        out_hpu = torch.batch_norm_elemt(inp_split_list_hpu[i], None, None, g_mean_hpu, g_invstd_hpu, 1e-5)
        out_list.append(out)
        out_list_hpu.append(out_hpu)

    # Veriy results between CPU and GPU
    for i in range(len(out_list)):
        err = np.allclose(
            out_list_hpu[i].to(cpu).detach().numpy(), out_list[i].detach().numpy(), atol=atol, rtol=rtol, equal_nan=True
        )
        print(f"Device: {i}, output allclose after batch_norm_elemt = {err}")

    # SANITY CHECK
    out = batch_norm_elemt_ref(inp, None, None, g_mean, g_invstd, 1e-5)
    out_hpu = torch.batch_norm_elemt(inp_hpu, None, None, g_mean_hpu, g_invstd_hpu, 1e-5)
    err = np.allclose(out_hpu.to(cpu).detach().numpy(), out.detach().numpy(), atol=atol, rtol=rtol, equal_nan=True)
    print(f"SANITY: Output allclose = {err}")
    print("=====")

    ######## BACKWARD PASS #########
    print("Backward pass\n-----")
    grad_out = torch.rand(N, C, H, W)
    grad_out_hpu = grad_out.to(hpu)

    grad_out_list = torch.split(grad_out, N // num_devices, 0)
    grad_out_hpu_list = torch.split(grad_out_hpu, N // num_devices, 0)

    sum_dy_list = []
    sum_dy_hpu_list = []
    sum_dy_xmu_list = []
    sum_dy_xmu_hpu_list = []
    grad_wei_list = []
    grad_wei_hpu_list = []
    grad_bias_list = []
    grad_bias_hpu_list = []

    # Compute local statistics on each device
    for i in range(len(grad_out_list)):
        sum_dy, sum_dy_xmu, g_w, g_b = batch_norm_backward_reduce_ref(
            grad_out_list[i], inp_split_list[i], g_mean, g_invstd, None, True, False, False
        )
        sum_dy_hpu, sum_dy_xmu_hpu, g_w_hpu, g_b_hpu = torch.batch_norm_backward_reduce(
            grad_out_hpu_list[i], inp_split_list_hpu[i], g_mean_hpu, g_invstd_hpu, None, True, False, False
        )

        sum_dy_list.append(sum_dy)
        sum_dy_hpu_list.append(sum_dy_hpu)
        sum_dy_xmu_list.append(sum_dy_xmu)
        sum_dy_xmu_hpu_list.append(sum_dy_xmu_hpu)
        grad_wei_list.append(g_w)
        grad_wei_hpu_list.append(g_w_hpu)
        grad_bias_list.append(g_b)
        grad_bias_hpu_list.append(g_b_hpu)

    # Verify CPU and GPU results
    print("Comparing results for batch_norm_backward_reduce")
    for i in range(len(grad_out_list)):
        err_sum_dy = np.allclose(
            sum_dy_hpu_list[i].to(cpu).detach().numpy(),
            sum_dy_list[i].detach().numpy(),
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        )
        err_sum_dy_xmu = np.allclose(
            sum_dy_xmu_hpu_list[i].to(cpu).detach().numpy(),
            sum_dy_xmu_list[i].detach().numpy(),
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        )
        print(f"Device: {i}: rms sum_dy allclose = {err_sum_dy}, rms sum_dy_xmu allclose = {err_sum_dy_xmu}")
    print("-----")
    # All reduce
    sum_dy_tensor = torch.stack(sum_dy_list)
    sum_dy_xmu_tensor = torch.stack(sum_dy_xmu_list)
    sum_dy_tensor_hpu = torch.stack(sum_dy_hpu_list)
    sum_dy_xmu_tensor_hpu = torch.stack(sum_dy_xmu_hpu_list)

    g_sum_dy = torch.sum(sum_dy_tensor, 0)
    g_sum_dy_xmu = torch.sum(sum_dy_xmu_tensor, 0)
    g_sum_dy_hpu = torch.sum(sum_dy_tensor_hpu, 0)
    g_sum_dy_xmu_hpu = torch.sum(sum_dy_xmu_tensor_hpu, 0)

    print("Comparing results of all reduce")
    print(
        f"global sum_dy rms error allclose = {np.allclose(g_sum_dy_hpu.to(cpu).detach().numpy(), g_sum_dy.detach().numpy(), atol=atol, rtol=rtol, equal_nan=True)}"
    )
    print(
        f"global sum_dy_xmu rms error allclose = {np.allclose(g_sum_dy_xmu_hpu.to(cpu).detach().numpy(), g_sum_dy_xmu.detach().numpy(), atol=atol, rtol=rtol, equal_nan=True)}"
    )
    print("-----")
    # Batch norm backward elemt
    grad_in_list = []
    grad_in_hpu_list = []
    for i in range(len(inp_split_list)):
        grad_in = batch_norm_backward_elemt_ref(
            grad_out_list[i], inp_split_list[i], g_mean, g_invstd, None, g_sum_dy, g_sum_dy_xmu, counts
        )
        grad_in_hpu = torch.batch_norm_backward_elemt(
            grad_out_hpu_list[i],
            inp_split_list_hpu[i],
            g_mean_hpu,
            g_invstd_hpu,
            None,
            g_sum_dy_hpu,
            g_sum_dy_xmu_hpu,
            counts_hpu.to(torch.int32),
        )

        grad_in_list.append(grad_in)
        grad_in_hpu_list.append(grad_in_hpu)

    # Verify CPU and GPU results
    print("Comparing results of batch_norm_backward_elemt")
    for i in range(len(inp_split_list)):
        np.testing.assert_allclose(
            grad_in_hpu_list[i].to(cpu).detach().numpy(),
            grad_in_list[i].detach().numpy(),
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        )

    # Sanity check
    grad_in_sanity = batch_norm_backward_elemt_ref(
        grad_out, inp, g_mean, g_invstd, None, g_sum_dy, g_sum_dy_xmu, counts
    )
    grad_in_sanity_hpu = torch.batch_norm_backward_elemt(
        grad_out_hpu,
        inp_hpu,
        g_mean_hpu,
        g_invstd_hpu,
        None,
        g_sum_dy_hpu,
        g_sum_dy_xmu_hpu,
        counts_hpu.to(torch.int32),
    )
    np.testing.assert_allclose(
        grad_in_sanity_hpu.to(cpu).detach().numpy(),
        grad_in_sanity.detach().numpy(),
        atol=atol,
        rtol=rtol,
        equal_nan=True,
    )
