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

import pytest
import torch
from test_utils import compare_tensors, generic_setup_teardown_env

pytestmark = pytest.mark.skip(reason="Tests in this file are chaning env variables")


@pytest.fixture(autouse=True, scope="module")
def setup_teardown_env():
    yield from generic_setup_teardown_env(
        {
            "PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": "1",
            "PT_HPU_DYNAMIC_MIN_POLICY_ORDER": "4,3,1",
            "PT_HPU_DYNAMIC_MAX_POLICY_ORDER": "4,2,3,1",
        }
    )


@pytest.mark.parametrize(
    "t1_shape, t2_shape",
    [
        ([1023], [1]),
        ([1007], [1]),
        ([2048], [2048]),
        ([1030], [1]),
    ],
)
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
