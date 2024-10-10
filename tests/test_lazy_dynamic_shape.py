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

import habana_frameworks.torch.utils.debug as htdebug
import numpy as np
import pytest
import torch

try:
    import habana_frameworks.torch.core as htcore
except ImportError:
    raise AssertionError("Could Not import habana_frameworks.torch.core")

input_shapes = [(3, 6, 4), (3, 8, 4), (3, 10, 4)]


from test_utils import setup_teardown_env_fixture


@pytest.mark.skip(reason="Tests in this file are chaning env variables")
@pytest.mark.parametrize("setup_teardown_env_fixture", [{"PT_HPU_LAZY_MODE": "1"}], indirect=True)
@pytest.mark.parametrize("shapes", input_shapes)
def test_hpu_lazy_dynamic_shape(shapes, setup_teardown_env_fixture):
    hpu = torch.device("hpu")
    for s in shapes:
        t1 = torch.randn(s, requires_grad=False)
        t2 = torch.randn(s, requires_grad=False)

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
        assert np.allclose(t6, t6_h_cpu, atol=0.001, rtol=1.0e-3), "Data mismatch"


@pytest.mark.parametrize("shapes", input_shapes)
@pytest.mark.skip(reason="Tests in this file are chaning env variables")
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": "1"}],
    indirect=True,
)
def test_hpu_lazy_dynamic_shape_cache_clear(shapes, setup_teardown_env_fixture):
    hpu = torch.device("hpu")
    for s in shapes:
        t1 = torch.randn(s, requires_grad=False)
        t2 = torch.randn(s, requires_grad=False)

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
        assert np.allclose(t6, t6_h_cpu, atol=0.001, rtol=1.0e-3), f"Data mismatch"
        htdebug.clear_dynamic_bucket_recipe_info()


@pytest.mark.parametrize(
    "setup_teardown_env_fixture", [{"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": "1", "PT_HPU_LAZY_MODE": "1"}], indirect=True
)
def test_hpu_lazy_dynamic_shape_simple(setup_teardown_env_fixture):
    def raw_function(t1, t2):
        t3 = torch.mul(t1, t2)
        tmp1 = t3 - 1
        return torch.relu(tmp1)

    for s in input_shapes:
        t1 = torch.randn(s, requires_grad=False)
        t2 = torch.randn(s, requires_grad=False)
        t1_h = t1.to("hpu")
        t2_h = t2.to("hpu")
        out_c = raw_function(t1, t2)
        out_h = raw_function(t1_h, t2_h)
        assert np.allclose(out_h.to("cpu"), out_c, atol=0.001, rtol=1.0e-3), f"Data mismatch"
