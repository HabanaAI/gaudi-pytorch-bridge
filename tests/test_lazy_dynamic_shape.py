import torch
import sys
import os
import numpy as np
import pytest
import habana_frameworks.torch.utils.debug as htdebug

try:
    import habana_frameworks.torch.core as htcore
except ImportError:
    assert False, "Could Not import habana_frameworks.torch.core"

input_shapes = [
    (3, 6, 4),
    (3, 8, 4),
    (3, 10, 4)
]

@pytest.mark.parametrize("shapes", input_shapes)
def test_hpu_lazy_dynamic_shape(shapes):
    hpu = torch.device("hpu")
    for s in shapes:
        t1 = torch.randn(s, requires_grad = False)
        t2 = torch.randn(s, requires_grad = False)

        t3 = torch.add(t1, t2)
        t4 = torch.mul(t1, t2)
        t5 = torch.mul(t3, t4)
        t6 = torch.relu(t5)

        t1_h = t1.to(hpu)
        t2_h = t2.to(hpu)
        t3_h = torch.add(t1_h, t2_h)
        t4_h = torch.mul(t1_h, t2_h)
        t5_h = torch.mul(t3_h, t4_h)
        t6_h = torch.relu(t5_h)

        htcore.mark_step()

        t6_h_cpu = t6_h.cpu()
        assert np.allclose(t6, t6_h_cpu, atol=0.001, rtol=1.e-3), f"Data mismatch"

@pytest.mark.parametrize("shapes", input_shapes)
def test_hpu_lazy_dynamic_shape_cache_clear(shapes):
    hpu = torch.device("hpu")
    for s in shapes:
        t1 = torch.randn(s, requires_grad = False)
        t2 = torch.randn(s, requires_grad = False)

        t3 = torch.add(t1, t2)
        t4 = torch.mul(t1, t2)
        t5 = torch.mul(t3, t4)
        t6 = torch.relu(t5)

        t1_h = t1.to(hpu)
        t2_h = t2.to(hpu)
        t3_h = torch.add(t1_h, t2_h)
        t4_h = torch.mul(t1_h, t2_h)
        t5_h = torch.mul(t3_h, t4_h)
        t6_h = torch.relu(t5_h)

        htcore.mark_step()

        t6_h_cpu = t6_h.cpu()
        assert np.allclose(t6, t6_h_cpu, atol=0.001, rtol=1.e-3), f"Data mismatch"
        htdebug.clear_dynamic_bucket_recipe_info()


if __name__ == '__main__':
    run_lazy_mode = os.environ["PT_HPU_LAZY_MODE"]
    if not run_lazy_mode:
        assert False, "Set PT_HPU_LAZY_MODE=1 to run in Lazy mode"

    run_ds = os.environ["PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES"]
    if not run_ds:
        assert False, "Set PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=1 to enable dynamic shape"

    test_hpu_lazy_dynamic_shape(input_shapes)
    test_hpu_lazy_dynamic_shape_cache_clear(input_shapes)
