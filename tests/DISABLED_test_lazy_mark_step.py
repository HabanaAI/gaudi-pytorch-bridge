import torch
import torch.nn as nn
from test_utils import compare_tensors
import hb_torch
import pytest
import os
import hblazy.core.hb_model as hm


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

    def forward(self, x, y):
        z = torch.add(x, y)
        a = torch.relu(z)
        return a


def test_lazy_mark_step_basic():
    hpu = torch.device("habana")
    cpu = torch.device("cpu")
    m = model()

    os.environ["PT_HPU_LAZY_MODE"] = "1"
    a = torch.randn(2, 2)
    b = torch.randn(2, 2)
    out_cpu = m(a, b)

    a_hpu = a.to(hpu)
    b_hpu = b.to(hpu)
    d_hpu = m(a_hpu, b_hpu)
    hm.mark_step()
    d_hpu = m(a_hpu, b_hpu)
    hm.mark_step()
    out_hpu = d_hpu.to(cpu)
    compare_tensors(out_hpu, out_cpu, atol=0, rtol=0)

    os.environ.pop("PT_HPU_LAZY_MODE")

if __name__ == "__main__":
    test_lazy_mark_step_basic()