import torch
import sys
import os
import numpy as np
from test_utils import compare_tensors, evaluate_fwd_bwd_kernel

def test_hpu_lazy_slice_fwd_bwd():
    t1 = torch.randn((5, 5), requires_grad=True)
    grad_out = torch.randn(3, 2, requires_grad=False)

    hpu = torch.device("hpu")

    t1_h = t1.detach().to(hpu)
    t1_h.requires_grad = True
    t1_h.retain_grad()

    out = t1[0:5:2, 0:2]
    out.sum().backward()
    grad_t1_cpu = t1.grad.clone().detach()
    out_h = t1_h[0:5:2, 0:2]
    out_h.sum().backward()
    grad_t1_h = t1_h.grad.cpu()

    assert np.allclose(grad_t1_cpu, grad_t1_h, atol=0, rtol=0), f"Data mismatch"


if __name__ == '__main__':
    run_lazy_mode = os.environ["PT_HPU_LAZY_MODE"]
    if not run_lazy_mode:
        assert False, "Set PT_HPU_LAZY_MODE=1 to run in Lazy mode"

    test_hpu_lazy_slice_fwd_bwd()
