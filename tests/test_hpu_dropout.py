import torch
import numpy as np
import os
import pytest
from test_utils import compare_tensors


def test_hpu_dropout():
    # torch.manual_seed(0)
    a_cpu = torch.randn(2, 2, requires_grad=True).detach()
    a = a_cpu.to("hpu")
    a.requires_grad = True
    dropoutmod = torch.nn.Dropout(p=1.0).to("hpu")
    out = dropoutmod(a)
    grad_out_cpu = torch.randn((2, 2)).detach()
    grad_out = grad_out_cpu.to("hpu")
    grad_out.requires_grad = False
    out.backward(grad_out)
    grad_in = a.grad
    # print(a.to('cpu'), out.to('cpu'), grad_out.to('cpu'), grad_in.to('cpu'))


def test_hpu_dropout_lazy():
    # torch.manual_seed(0)
    os.environ["PT_HPU_LAZY_MODE"] = "1"
    a = torch.randn(2, 2, requires_grad=True).to("hpu")
    dropoutmod = torch.nn.Dropout(p=0.3)
    out = dropoutmod(a)
    grad_out = torch.randn((2, 2), requires_grad=False)
    grad_in = out.grad_fn(grad_out.to("hpu"))
    out.to("cpu")
    grad_in.to("cpu")
    del os.environ["PT_HPU_LAZY_MODE"]
    # print(a.to('cpu'), out.to('cpu'), grad_out.to('cpu'), grad_in.to('cpu'))


if __name__ == "__main__":
    test_hpu_dropout()
