import torch
import sys
import os
import numpy as np
from test_utils import compare_tensors
import pytest
try:
    import habana_frameworks.torch.core as htcore
except ImportError:
    assert False, "Could Not import habana_frameworks.torch.core"

sizeList = [
    # size
    (4, 5),
    (2, 3, 4),
    (2, 3, 2, 3)
]

@pytest.mark.parametrize("size ", sizeList)
def test_hpu_lazy_floor_divide_size_variation(size):
    t1 = torch.randn(size, requires_grad = False)
    t2 = torch.randn(size, requires_grad = False)

    hpu = torch.device("hpu")

    t1_h = t1.to(hpu)
    t2_h = t2.to(hpu)

    out_h = t1_h// t2_h
    out = t1// t2

    compare_tensors(out_h, out, atol=0.001, rtol=0.001)

def test_hpu_lazy_floor_divide_broadcast():
    size1 = (3,5,4)
    size2 = (1,5,4)
    t1 = torch.randn(size1, requires_grad = False)
    t2 = torch.randn(size2, requires_grad = False)

    hpu = torch.device("hpu")

    t1_h = t1.to(hpu)
    t2_h = t2.to(hpu)

    out_h = t1_h// t2_h
    out = t1// t2

    compare_tensors(out_h, out, atol=0.001, rtol=0.001)

def test_hpu_lazy_floor_divid_int_int():
    size = (5,4)
    max_int_tested = 300

    t1 = torch.randint(max_int_tested, size, requires_grad = False)
    t2 = torch.randint(max_int_tested, size, requires_grad = False)

    hpu = torch.device("hpu")

    t1_h = t1.to(hpu)
    t2_h = t2.to(hpu)

    out_h = t1_h// t2_h
    out = t1// t2
    compare_tensors(out_h, out, atol=0.001, rtol=0.001)

if __name__ == '__main__':
    run_lazy_mode = os.environ["PT_HPU_LAZY_MODE"]
    if not run_lazy_mode:
        assert False, "Set PT_HPU_LAZY_MODE=1 to run in Lazy mode"
    test_hpu_lazy_floor_divide_size_variation()
    test_hpu_lazy_floor_divide_broadcast()
    test_hpu_lazy_floor_divid_int_int()
