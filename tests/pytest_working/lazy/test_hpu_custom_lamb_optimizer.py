###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

import copy

import habana_frameworks.torch.core as htcore
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from habana_frameworks.torch.hpex.optimizers import FusedLamb
from test_utils import compare_tensors, cpu, hpu


def reference_lamb_norm(grads, max_grad_norm):
    global_grad_norm = torch.zeros(1, dtype=grads[0].dtype, device=grads[0].device)
    for grad in grads:
        global_grad_norm.add_(grad.pow(2).sum())

    global_grad_norm = global_grad_norm.sqrt()

    if global_grad_norm > max_grad_norm:
        clip_global_grad_norm = global_grad_norm / max_grad_norm
    else:
        clip_global_grad_norm = torch.tensor([1.0], dtype=grads[0].dtype)
    return clip_global_grad_norm


def create_args(dtypes, shapes, fn):
    cpu_grads, hpu_grads = [], []
    for dtype, shape in zip(dtypes, shapes):
        cpu_grads.append(fn(shape, device=cpu).to(dtype))
        hpu_grads.append(cpu_grads[-1].to(hpu))
    return cpu_grads, hpu_grads


# norms should be non-negative
def create_norms(dtypes, shapes):
    return create_args(dtypes, shapes, torch.rand)


def create_grads(dtypes, shapes):
    return create_args(dtypes, shapes, torch.randn)


@pytest.mark.parametrize("max_grad_norm", (0.2, 1.0, 4.0, 8))
@pytest.mark.parametrize(
    "shapes, dtypes",
    (
        ([(3, 4), (5, 6)], [torch.float, torch.float]),
        ([(3, 4), (5, 6)], [torch.bfloat16, torch.bfloat16]),
    ),
)
def test_optimizer_lamb_norm(dtypes, shapes, max_grad_norm):

    cpu_grads, hpu_grads = create_grads(dtypes, shapes)

    result = torch.ops.hpu.optimizer_lamb_fused_norm(hpu_grads, max_grad_norm)
    reference = reference_lamb_norm(cpu_grads, max_grad_norm)

    compare_tensors(result, reference, atol=1e-08, rtol=1e-05)


def test_optimizer_lamb_norm_slice_insert():
    tensor_hpu = torch.zeros(4).to(hpu)
    tensor_hpu[:2] = 2.0
    tensor_hpu[2:] = 1.0
    tensor_cpu = torch.tensor([2.0, 2.0, 1.0, 1.0])

    grad_denom_hpu = torch.ops.hpu.optimizer_lamb_fused_norm([tensor_hpu], 1.0)
    grad_denom_cpu = reference_lamb_norm([tensor_cpu], 1.0)

    compare_tensors(grad_denom_hpu, grad_denom_cpu, atol=1e-08, rtol=1e-05)


def test_optimizer_lamb_norm_views():
    tensor_cpu, tensor_hpu = create_grads(
        [torch.float32],
        [(2, 2)],
    )
    tensor_hpu = tensor_hpu[0].view(-1)
    tensor_cpu = tensor_cpu[0].view(-1)

    grad_denom_hpu = torch.ops.hpu.optimizer_lamb_fused_norm([tensor_hpu], 1.0)
    grad_denom_cpu = reference_lamb_norm([tensor_cpu], 1.0)

    compare_tensors(grad_denom_hpu, grad_denom_cpu, atol=1e-08, rtol=1e-05)


def reference_optimizer_lamb_phase2(weights, adam_norms, weight_norms, adam_steps, neg_step, weight_decay, use_lamb):
    for weight, adam_norm, weight_norm, adam_step in zip(weights, adam_norms, weight_norms, adam_steps):
        if (weight_decay != 0 or use_lamb) and adam_norm > 0 and weight_norm > 0:
            trust_ratio = weight_norm / adam_norm
        else:
            trust_ratio = 1.0
        adam_step = adam_step * neg_step * trust_ratio
        weight.add_(adam_step)


@pytest.mark.parametrize(
    "weight_dtype",
    [torch.float, torch.bfloat16],
)
@pytest.mark.parametrize("weight_shapes", [[(5, 4)], [(2, 3, 3), (4, 2)]])
@pytest.mark.parametrize("use_lamb", [True, False])
@pytest.mark.parametrize("weight_decay", [0, 0.1])
def test_optimizer_lamb_phase2(weight_dtype, weight_shapes, weight_decay, use_lamb):
    lr = 0.1
    n = len(weight_shapes)
    cpu_weights, hpu_weights = create_grads([weight_dtype] * n, weight_shapes)
    cpu_adam_norm, hpu_adam_norm = create_norms([weight_dtype] * n, [(1,)] * n)
    cpu_weight_norm, hpu_weight_norm = create_norms([weight_dtype] * n, [(1,)] * n)
    cpu_adam_step, hpu_adam_step = create_grads([weight_dtype] * n, weight_shapes)

    torch.ops.hpu.optimizer_lamb_phase2(
        hpu_weights,
        hpu_adam_norm,
        hpu_weight_norm,
        hpu_adam_step,
        torch.tensor(-lr, device=hpu),
        weight_decay,
        use_lamb,
    )
    reference_optimizer_lamb_phase2(
        cpu_weights,
        cpu_adam_norm,
        cpu_weight_norm,
        cpu_adam_step,
        -lr,
        weight_decay,
        use_lamb,
    )

    atol = 1e-08
    rtol = 1e-05
    if weight_dtype == torch.bfloat16:
        atol = 1e-02
        rtol = 1e-02
    compare_tensors(hpu_weights, cpu_weights, atol, rtol)


def reference_optimizer_lamb_phase1(
    grad_list,
    wt_list,
    exp_avg_list,
    exp_avg_sq_list,
    wt_norm_list,
    adam_norm_list,
    adam_step_list,
    clip_global_grad_norm,
    averaging,
    beta1,
    beta2,
    eps,
    step,
    bias_correction,
    weight_decay,
):
    if averaging:
        beta3 = 1.0 - beta1
    else:
        beta3 = 1.0
    for i in range(len(wt_list)):
        grad = grad_list[i].div_(clip_global_grad_norm)
        # Decay the first and second moment running average coefficient
        # m_t
        exp_avg_list[i].mul_(beta1).add_(grad, alpha=beta3)
        # v_t
        exp_avg_sq_list[i].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        if bias_correction:
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
        else:
            bias_correction1, bias_correction2 = 1.0, 1.0
        # create clones to avoid modifying runner stats
        exp_avg = exp_avg_list[i].div(bias_correction1)
        exp_avg_sq = exp_avg_sq_list[i].div(bias_correction2)
        # || w_t ||
        wt_norm_list[i] = wt_list[i].norm()
        # u_t
        adam_step_list[i] = exp_avg.div(exp_avg_sq.sqrt().add_(eps))

        if weight_decay != 0:
            adam_step_list[i].add_(wt_list[i], alpha=weight_decay)
        # || u_t ||
        adam_norm_list[i] = adam_step_list[i].norm()


@pytest.mark.parametrize("weight_decay", [0, 0.01])
@pytest.mark.parametrize("bias_correction", [0, 1])
@pytest.mark.parametrize("step", [1, 4])
@pytest.mark.parametrize("grad_averaging", [0, 1])
def test_optimizer_lamb_phase1(weight_decay, bias_correction, step, grad_averaging):

    dtype = torch.float
    shape = (5, 4)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-6

    cpu_grad_list = [torch.randn(shape, dtype=dtype)]
    cpu_wt_list = [torch.randn(shape, dtype=dtype)]
    cpu_exp_avg_list = [torch.randn(shape, dtype=dtype)]
    cpu_exp_avg_sq_list = [torch.abs(torch.randn(shape, dtype=dtype))]
    cpu_wt_norm_list = [torch.empty((1,), dtype=dtype)]
    cpu_adam_norm_list = [torch.empty((1,), dtype=dtype)]
    cpu_adam_step_list = [torch.empty(shape, dtype=dtype)]
    cpu_clip_global_grad_norm = reference_lamb_norm(cpu_grad_list, 1.0)

    hpu_grad_list = [cpu_grad_list[0].to(hpu)]
    hpu_wt_list = [cpu_wt_list[0].to(hpu)]
    hpu_exp_avg_list = [cpu_exp_avg_list[0].to(hpu)]
    hpu_exp_avg_sq_list = [cpu_exp_avg_sq_list[0].to(hpu)]
    hpu_wt_norm_list = [torch.empty((1,), dtype=dtype).to(hpu)]
    hpu_adam_norm_list = [torch.empty((1,), dtype=dtype).to(hpu)]
    hpu_adam_step_list = [torch.empty(shape, dtype=dtype).to(hpu)]
    hpu_clip_global_grad_norm = cpu_clip_global_grad_norm.to(hpu)

    if bias_correction:
        bias_correction1 = torch.tensor(1.0 - pow(beta1, step), device=hpu)
        bias_correction2 = torch.tensor(1.0 - pow(beta2, step), device=hpu)
    else:
        bias_correction1 = torch.tensor(1.0, device=hpu)
        bias_correction2 = torch.tensor(1.0, device=hpu)

    torch.ops.hpu.optimizer_lamb_phase1(
        hpu_grad_list,
        hpu_wt_list,
        hpu_exp_avg_list,
        hpu_exp_avg_sq_list,
        hpu_wt_norm_list,
        hpu_adam_norm_list,
        hpu_adam_step_list,
        hpu_clip_global_grad_norm,
        grad_averaging,
        beta1,
        beta2,
        eps,
        bias_correction1,
        bias_correction2,
        weight_decay,
    )
    reference_optimizer_lamb_phase1(
        cpu_grad_list,
        cpu_wt_list,
        cpu_exp_avg_list,
        cpu_exp_avg_sq_list,
        cpu_wt_norm_list,
        cpu_adam_norm_list,
        cpu_adam_step_list,
        cpu_clip_global_grad_norm,
        grad_averaging,
        beta1,
        beta2,
        eps,
        step,
        bias_correction,
        weight_decay,
    )

    compare_tensors(hpu_exp_avg_list, cpu_exp_avg_list, atol=1.0e-6, rtol=1.0e-6)
    compare_tensors(hpu_exp_avg_sq_list, cpu_exp_avg_sq_list, atol=1.0e-6, rtol=1.0e-6)
    compare_tensors(hpu_wt_norm_list, cpu_wt_norm_list, atol=1.0e-6, rtol=1.0e-6)
    compare_tensors(hpu_adam_norm_list, cpu_adam_norm_list, atol=1.0e-6, rtol=1.0e-6)
    compare_tensors(hpu_adam_step_list, cpu_adam_step_list, atol=1.0e-6, rtol=1.0e-6)


def test_lamb0():
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
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
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

    for _ in range(2):
        opt_hpu_fl.step()
        htcore.mark_step()
    # The expectation is that the test case survives up to this point.


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5, 1)
        self.conv2 = nn.Conv2d(3, 6, 5, 1)
        self.fc1 = nn.Linear(3 * 3 * 6, 30)
        self.fc2 = nn.Linear(30, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = x.view(-1, 3 * 3 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


@pytest.mark.skip(reason="Results mismatch")
@pytest.mark.parametrize("count, lr", [(3, 0.001), (2, 0.01)])
def test_lamb(count, lr):
    from habana_frameworks.torch.hpex.optimizers import FusedLamb
    from pytest_working.fused_ops.test_lamb import TorchNVLAMB

    m_hpu_nv = MNISTNet().to(hpu)
    m_clone = copy.deepcopy(m_hpu_nv)

    i_clone_list, t_clone_list = [], []

    opt_hpu_nv = TorchNVLAMB(m_hpu_nv.parameters(), lr=lr)

    for _ in range(count):
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
            compare_tensors(p.data.to(cpu), q.data.to(cpu), atol=1.0e-4, rtol=1.0e-4)
