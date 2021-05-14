import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F
from test_utils import compare_tensors
import habana_frameworks.torch.core as htcore
import os
import sys
import itertools
import copy
import numpy as np
from test_utils import *

sys.path.append( os.environ["MODEL_GARDEN_PYTORCH_PATH"] + "/nlp/bert/pretraining/")
from lamb import NVLAMB as TorchNVLAMB

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

test_case_list = [
    # iterations, lr,
    (3, 0.001),
    (2, 0.01)
    ]

@pytest.mark.parametrize("count, lr", test_case_list)
def test_lamb(count, lr):
    m_hpu_nv = MNISTNet().to(hpu)
    m_clone =  copy.deepcopy(m_hpu_nv)

    i_clone_list, t_clone_list = [], []

    opt_hpu_nv = TorchNVLAMB(m_hpu_nv.parameters(), lr = lr)
    opt_hpu_nv.zero_grad()

    with torch.no_grad():
        for name, param in m_hpu_nv.named_parameters():
            if(param.ndim == 4):
                param.data = param.data.permute((2,3,1,0))

    for i in range(count) :
        i_hpu_nv = torch.rand(1, 1, 28, 28)
        t_hpu_nv = torch.randint(10, (1,))
        i_clone_list.append(i_hpu_nv.detach().clone())
        t_clone_list.append(t_hpu_nv.detach().clone())

        # train one iteration on cpu
        out_hpu_nv = m_hpu_nv(i_hpu_nv.to(hpu))
        l_hpu_nv = F.nll_loss(out_hpu_nv, t_hpu_nv.to(hpu))
        l_hpu_nv.backward()
        opt_hpu_nv.step()

    try:
        from hb_custom import FusedLamb
    except ImportError:
        raise ImportError("Please install hb_custom.")

    # same model for training with FusedLamb
    m_hpu_fl = m_clone
    opt_hpu_fl = FusedLamb(m_hpu_fl.parameters(), lr = lr)
    opt_hpu_fl.zero_grad()

    with torch.no_grad():
        for name, param in m_hpu_fl.named_parameters():
            if(param.ndim == 4):
                param.data = param.data.permute((2,3,1,0))

    for i in range(count) :
        i_hpu_fl, t_hpu_fl = i_clone_list[i], t_clone_list[i]
        # train one iteration on hpu
        out_hpu_fl = m_clone(i_hpu_fl.to(hpu))
        l_hpu_fl = F.nll_loss(out_hpu_fl, t_hpu_fl.to(hpu))
        l_hpu_fl.backward()
        opt_hpu_fl.step()

    #compare NVLamb and FusedLamb results
    for j, (p, q) in enumerate(zip(m_hpu_nv.parameters(), m_hpu_fl.parameters())):
        if p.requires_grad and q.requires_grad:
            compare_tensors(p.data.to(cpu), q.data.to(cpu), atol=0.001, rtol=1.0e-3)

if __name__ == "__main__":
    test_lamb(*test_case_list[0])
