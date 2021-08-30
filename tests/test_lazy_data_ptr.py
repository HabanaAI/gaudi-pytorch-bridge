import torch
import sys
import os
import numpy as np
from test_utils import compare_tensors

try:
    import habana_frameworks.torch.core as htcore
except ImportError:
    assert False, "Could Not import habana_frameworks.torch.core"


def test_hpu_lazy_data_ptr(input_tensor):
    t1 = torch.randn(input_tensor, requires_grad=True)
    grad_out = torch.randn(3, 2, requires_grad=False)

    hpu = torch.device("hpu")

    os.environ["PT_HPU_LAZY_MODE"] = "1"
    os.environ["PT_HPU_ENABLE_DATAPTR_ACCESS"] = "1"
    t1_h = t1.detach().to(hpu)
    t1_h.requires_grad = True
    t1_h.retain_grad()
    print("t1_h data_ptr ", t1_h.data_ptr())

    t2_h = torch.abs(t1_h)
    t3_h = t2_h.mul_(t1_h)
    out_h = torch.add(t2_h, t3_h)

    print("out_h data_ptr ", out_h.data_ptr())

    t2 = torch.abs(t1)
    t3 = t2.mul_(t1)
    out = torch.add(t2, t3)

    out.sum().backward()
    grad_t1_cpu = t1.grad.clone().detach()

    out_h.sum().backward()
    # out_h.backward(grad_out.detach().to(hpu))
    htcore.mark_step()

    print("t1_h.grad data_ptr ", t1_h.grad.data_ptr())
    grad_t1_h = t1_h.grad.cpu()

    out_cpu_to_compare = out.clone().detach()
    out_h_cpu_to_compare = out_h.cpu().clone().detach()

    del os.environ["PT_HPU_LAZY_MODE"]
    del os.environ["PT_HPU_ENABLE_DATAPTR_ACCESS"]
    assert np.allclose(out_cpu_to_compare, out_h_cpu_to_compare, atol=0, rtol=0), f"Data mismatch"

if __name__ == '__main__':
    test_hpu_lazy_data_ptr((5, 5))
