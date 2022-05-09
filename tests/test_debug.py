import os
import torch
import numpy as np
import pytest
from test_utils import compare_tensors


def test_hpu_ds(t1_shape, t2_shape):
    t1 = torch.randint(0, 2, t1_shape)
    t1_hpu = t1.to("hpu")
    t2 = torch.ones(t2_shape)
    t2_hpu = t2.to("hpu")
    t3 = t1.eq(t2)
    t3_hpu = t1_hpu.eq(t2_hpu)
    t4 = t3.nonzero()
    t4_hpu = t3_hpu.nonzero()
    print(t1_shape, t2_shape)
    compare_tensors(t4_hpu, t4, atol=0.0, rtol=0.0)

if __name__ == "__main__":
    os.environ["PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES"] = "1"
    os.environ["PT_HPU_DYNAMIC_MIN_POLICY_ORDER"] = "4,3,1"
    os.environ["PT_HPU_DYNAMIC_MAX_POLICY_ORDER"] = "4,2,3,1"
    torch.manual_seed(0)
    test_hpu_ds([1023], [1])
    test_hpu_ds([1007], [1])
    test_hpu_ds([2048], [2048])
    test_hpu_ds([1030], [1])
