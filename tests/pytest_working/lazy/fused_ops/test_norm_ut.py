###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(3 * 3 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = x.view(-1, 3 * 3 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


@pytest.mark.skip
def test_mnist():
    m_cpu = MNISTNet()
    m_clone = copy.deepcopy(m_cpu)
    max_norm = 1.0

    i_clone_list, t_clone_list = [], []
    l_cpu_list, n_cpu_list = [], []

    opt_cpu = optim.SGD(m_cpu.parameters(), lr=0.003)
    opt_cpu.zero_grad()

    count = 5
    for _ in range(count):
        i_cpu = torch.rand(1, 1, 28, 28)
        t_cpu = torch.from_numpy(np.random.choice(10, 1))
        i_clone_list.append(i_cpu.clone())
        t_clone_list.append(t_cpu.clone())

        # train one iteration on cpu
        out_cpu = m_cpu(i_cpu)
        l_cpu = F.nll_loss(out_cpu, t_cpu)
        l_cpu.backward()
        opt_cpu.step()
        l_cpu_list.append(l_cpu)

        n_cpu = torch.nn.utils.clip_grad_norm_(m_cpu.parameters(), max_norm)
        n_cpu_list.append(n_cpu)

    habana = torch.device("hpu")
    cpu = torch.device("cpu")

    # same model in hpu
    m_hpu = m_clone.to(habana)

    opt_hpu = optim.SGD(m_hpu.parameters(), lr=0.003)
    opt_hpu.zero_grad()

    with torch.no_grad():
        for _, param in m_hpu.named_parameters():
            if param.ndim == 4:
                param.data = param.data.permute((2, 3, 1, 0))
    try:
        from habana_frameworks.torch.hpex.normalization import FusedClipNorm
    except ImportError:
        raise ImportError("Please install habana_torch.")

    for i in range(count):
        i_hpu, t_hpu = i_clone_list[i].to(habana), t_clone_list[i].to(habana)

        # train one iteration on hpu
        out_hpu = m_hpu(i_hpu)
        l_hpu = F.nll_loss(out_hpu, t_hpu)
        l_hpu.backward()
        opt_hpu.step()

        FusedNorm = FusedClipNorm(m_hpu.parameters(), max_norm)
        n_hpu = FusedNorm.clip_norm()

        comp = np.allclose(
            n_cpu_list[i].detach().numpy(),
            n_hpu.detach().to(cpu).numpy(),
            atol=0.001,
            rtol=1.0e-3,
            equal_nan=True,
        )

        print("Iteraion {} norm output match :: {}".format(i, comp))
