# ******************************************************************************
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************

import copy

import habana_frameworks.torch.core as htcore
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from test_utils import compare_tensors, cpu, hpu


def reference_lamb_fused_norm(grads, max_grad_norm):
    global_grad_norm = torch.zeros(1, dtype=grads[0].dtype, device=grads[0].device)
    for grad in grads:
        global_grad_norm.add_(grad.pow(2).sum())

    global_grad_norm = global_grad_norm.sqrt()

    if global_grad_norm > max_grad_norm:
        clip_global_grad_norm = global_grad_norm / max_grad_norm
    else:
        clip_global_grad_norm = torch.tensor([1.0], dtype=grads[0].dtype)
    return clip_global_grad_norm


def create_grads(dtypes, shapes):
    cpu_grads, hpu_grads = [], []
    for dtype, shape in zip(dtypes, shapes):
        cpu_grads.append(torch.randn(shape, device=cpu).to(dtype))
        hpu_grads.append(cpu_grads[-1].to(hpu))
    return cpu_grads, hpu_grads


@pytest.mark.parametrize("max_grad_norm", (0.2, 1.0, 4.0, 8))
@pytest.mark.parametrize(
    "shapes, dtypes",
    (
        ([(3, 4), (5, 6)], [torch.float, torch.float]),
        ([(3, 4), (5, 6)], [torch.bfloat16, torch.bfloat16]),
    ),
)
def test_optimizer_lamb_fused_norm(dtypes, shapes, max_grad_norm):
    torch.manual_seed(0)
    cpu_grads, hpu_grads = create_grads(dtypes, shapes)

    result = torch.ops.hpu.optimizer_lamb_fused_norm(hpu_grads, max_grad_norm)
    reference = reference_lamb_fused_norm(cpu_grads, max_grad_norm)

    compare_tensors(result, reference, atol=1e-08, rtol=1e-05)


def test_optimizer_lamb_fused_norm_slice_insert():
    tensor_hpu = torch.zeros(4).to("hpu")
    tensor_hpu[:2] = 2.0
    tensor_hpu[2:] = 1.0
    tensor_cpu = torch.tensor([2.0, 2.0, 1.0, 1.0])

    grad_denom_hpu = torch.ops.hpu.optimizer_lamb_fused_norm([tensor_hpu], 1.0)
    grad_denom_cpu = reference_lamb_fused_norm([tensor_cpu], 1.0)

    compare_tensors(grad_denom_hpu, grad_denom_cpu, atol=1e-08, rtol=1e-05)


def test_optimizer_lamb_fused_norm_views():
    tensor_cpu, tensor_hpu = create_grads(
        [torch.float32],
        [(2, 2)],
    )
    tensor_hpu = tensor_hpu[0].view(-1)
    tensor_cpu = tensor_cpu[0].view(-1)

    grad_denom_hpu = torch.ops.hpu.optimizer_lamb_fused_norm([tensor_hpu], 1.0)
    grad_denom_cpu = reference_lamb_fused_norm([tensor_cpu], 1.0)

    compare_tensors(grad_denom_hpu, grad_denom_cpu, atol=1e-08, rtol=1e-05)


def reference_optimizer_lamb_fused_phase2(
    weights, adam_norms, weight_norms, adam_steps, step, weight_decay, use_lamb
):
    for weight, adam_norm, weight_norm, adam_step in zip(
        weights, adam_norms, weight_norms, adam_steps
    ):
        if (weight_decay != 0 or use_lamb) and adam_norm > 0 and weight_norm > 0:
            trust_ratio = weight_norm / adam_norm
        else:
            trust_ratio = 1
        adam_step = adam_step * -step * trust_ratio
        weight.add_(adam_step)


@pytest.mark.parametrize(
    "weight_dtype",
    [torch.float, torch.bfloat16],
)
@pytest.mark.parametrize("weight_shapes", [[(5, 4)], [(2, 3, 3), (4, 2)]])
@pytest.mark.parametrize("use_lamb", [True, False])
@pytest.mark.parametrize("weight_decay", [0, 0.1])
def test_optimizer_lamb_fused_phase2(
    weight_dtype, weight_shapes, weight_decay, use_lamb
):
    torch.manual_seed(0)
    n = len(weight_shapes)
    cpu_weights, hpu_weights = create_grads([weight_dtype] * n, weight_shapes)
    cpu_adam_norm, hpu_adam_norm = create_grads([weight_dtype] * n, [(1,)] * n)
    cpu_weight_norm, hpu_weight_norm = create_grads([weight_dtype] * n, [(1,)] * n)
    cpu_adam_step, hpu_adam_step = create_grads([weight_dtype] * n, weight_shapes)

    torch.ops.hpu.optimizer_lamb_fused_phase2(
        hpu_weights,
        hpu_adam_norm,
        hpu_weight_norm,
        hpu_adam_step,
        0.1,
        weight_decay,
        use_lamb,
    )
    reference_optimizer_lamb_fused_phase2(
        cpu_weights,
        cpu_adam_norm,
        cpu_weight_norm,
        cpu_adam_step,
        0.1,
        weight_decay,
        use_lamb,
    )
    compare_tensors(hpu_weights, cpu_weights, atol=1e-08, rtol=1e-05)


def test_lamb0():
    from habana_frameworks.torch.hpex.optimizers import FusedLamb

    class TinyModel(nn.Module):
        def __init__(self):
            super(TinyModel, self).__init__()
            self.l0 = nn.Linear(1, 1)

        def forward(self, x):
            x = self.l0(x)
            return x

    m_hpu = TinyModel().to(hpu)

    param_optimizer = list(m_hpu.named_parameters())
    no_decay = ["bias", "gamma", "beta", "LayerNorm"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    opt_hpu_fl = FusedLamb(optimizer_grouped_parameters, lr=0.1)
    opt_hpu_fl.zero_grad()

    i_hpu_fl = torch.rand(1, 1).to(hpu)
    exp_hpu_fl = torch.rand(1, 1).to(hpu)

    o_hpu_fl = m_hpu(i_hpu_fl)
    loss = F.mse_loss(o_hpu_fl, exp_hpu_fl)
    htcore.mark_step()

    loss.backward()
    htcore.mark_step()

    for i in range(2):
        opt_hpu_fl.step()
        htcore.mark_step()
    # The expectation is that the test case survives up to this point.


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 5, 1)
        self.conv2 = nn.Conv2d(5, 10, 5, 1)
        self.fc1 = nn.Linear(3 * 3 * 10, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = x.view(-1, 3 * 3 * 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


test_case_list = [
    # iterations, lr,
    (1, 0.001),
    (1, 0.01),
]


@pytest.mark.parametrize("count, lr", test_case_list)
def test_lamb(count, lr):
    from habana_frameworks.torch.hpex.optimizers import FusedLamb
    from fused_ops.lamb_ut import TorchNVLAMB

    torch.manual_seed(0)

    m_hpu_nv = MNISTNet().to(hpu)
    m_clone = copy.deepcopy(m_hpu_nv)

    i_clone_list, t_clone_list = [], []

    opt_hpu_nv = TorchNVLAMB(m_hpu_nv.parameters(), lr=lr)

    for i in range(count):
        i_hpu_nv = torch.rand((1, 1, 28, 28))
        t_hpu_nv = torch.randint(10, (1,))
        i_clone_list.append(i_hpu_nv.detach().clone())
        t_clone_list.append(t_hpu_nv.detach().clone())

        # train one iteration on cpu
        opt_hpu_nv.zero_grad()
        out_hpu_nv = m_hpu_nv(i_hpu_nv.to(hpu))
        l_hpu_nv = F.nll_loss(out_hpu_nv, t_hpu_nv.to(hpu))
        l_hpu_nv.backward()
        opt_hpu_nv.step()

    # same model for training with FusedLamb
    opt_hpu_fl = FusedLamb(m_clone.parameters(), lr=lr)

    for i in range(count):
        i_hpu_fl, t_hpu_fl = i_clone_list[i], t_clone_list[i]

        # train one iteration on hpu
        opt_hpu_fl.zero_grad()
        out_hpu_fl = m_clone(i_hpu_fl.to(hpu))
        l_hpu_fl = F.nll_loss(out_hpu_fl, t_hpu_fl.to(hpu))
        l_hpu_fl.backward()
        opt_hpu_fl.step()

    # compare NVLamb and FusedLamb results
    for p, q in zip(m_hpu_nv.parameters(), m_clone.parameters()):
        if p.requires_grad and q.requires_grad:
            compare_tensors(p.data.to(cpu), q.data.to(cpu), atol=1.0e-3, rtol=1.0e-3)


if __name__ == "__main__":
    test_lamb0()
    test_lamb(*test_case_list[0])
