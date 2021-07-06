
import torch
import sys
import os
import numpy as np
from test_utils import compare_tensors, evaluate_fwd_bwd_kernel
import pytest
sys.path.insert(0, os.path.join(os.environ['PYTORCH_MODULES_RELEASE_BUILD']))
try:
    import habana_frameworks.torch.core as htcore
except ImportError:
    assert False, "Could Not import habana_frameworks.torch.core"

matmul_lazy_list = [
    # size1, size2
    ((2, 3), (3, 4)),
    ((2, 3, 4), (4, 5)),
    ((2, 3, 4), (2, 4, 5)),
    ((2, 3, 4), (4)),
    ((2, 2, 3, 4), (2, 4, 3))

]

@pytest.mark.parametrize("size1, size2", matmul_lazy_list)
def test_hpu_lazy_matmul_fwd_bwd(size1, size2):
    t1 = torch.randn(size1, requires_grad = True)
    t2 = torch.randn(size2, requires_grad = True)

    hpu = torch.device("hpu")

    t1_h = t1.to(hpu)
    t1_h.retain_grad()
    t2_h = t2.to(hpu)
    t2_h.retain_grad()

    out = torch.matmul(t1, t2)
    loss = out.sum()
    loss.backward()
    grad_t1_cpu = t1.grad.clone().detach()
    grad_t2_cpu = t2.grad.clone().detach()

    out_h = torch.matmul(t1_h, t2_h)
    loss_h = out_h.sum()
    loss_h.backward()

    htcore.mark_step()

    grad_t1_h = t1_h.grad.cpu()
    grad_t2_h = t2_h.grad.cpu()

    assert np.allclose(grad_t1_cpu, grad_t1_h, atol=0.001, rtol=1.e-3), f"Data mismatch"
    assert np.allclose(grad_t2_cpu, grad_t2_h, atol=0.001, rtol=1.e-3), f"Data mismatch"


if __name__ == '__main__':
    run_lazy_mode = os.environ["PT_HPU_LAZY_MODE"]
    if not run_lazy_mode:
        assert False, "Set PT_HPU_LAZY_MODE=1 to run in Lazy mode"
    
    test_hpu_lazy_matmul_fwd_bwd(*matmul_lazy_list[0])
