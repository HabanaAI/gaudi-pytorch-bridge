###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

import pytest
import torch
from binary_cross_entropy_utils import (
    binary_cross_entropy_bwd_test,
    binary_cross_entropy_fwd_test,
    binary_cross_entropy_with_logits_fwd_test,
)
from compile.test_dynamo_utils import use_eager_fallback
from test_utils import setup_teardown_env_fixture  # noqa F401
from test_utils import format_tc, is_gaudi1, is_pytest_mode_compile, is_pytest_mode_eager

size = [
    (6,),
    (3, 1),
    (2, 3, 4),
    (2, 3, 4, 1),
    (2, 3, 2, 3, 2),
]

reduction = ["none", "mean", "sum"]

dtype = [torch.float32, torch.bfloat16]
if not is_gaudi1():
    dtype.append(torch.float16)

use_weight_broadcastable_weight = [(False, False), (True, False), (True, True)]


# <--- Forward --->


@pytest.mark.parametrize("size", size, ids=format_tc)
@pytest.mark.parametrize("reduction", reduction, ids=format_tc)
@pytest.mark.parametrize("dtype", dtype, ids=format_tc)
@pytest.mark.parametrize("use_weight, broadcastable_weight", use_weight_broadcastable_weight, ids=format_tc)
def test_hpu_binary_cross_entropy_fwd(size, reduction, dtype, use_weight, broadcastable_weight):

    if not is_pytest_mode_eager() and reduction == "none" and use_weight and broadcastable_weight:
        pytest.skip("Handling under SW-204408, bug already in regression")

    with use_eager_fallback():

        binary_cross_entropy_fwd_test(
            size,
            reduction,
            dtype,
            use_weight,
            broadcastable_weight=broadcastable_weight,
            is_compile=is_pytest_mode_compile(),
        )


@pytest.mark.skipif(is_pytest_mode_eager(), reason="DS are not supported in eager mode")
@pytest.mark.parametrize("size", size, ids=format_tc)
@pytest.mark.parametrize("reduction", reduction, ids=format_tc)
@pytest.mark.parametrize("dtype", dtype, ids=format_tc)
@pytest.mark.parametrize("use_weight, broadcastable_weight", use_weight_broadcastable_weight, ids=format_tc)
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 1}],
    indirect=True,
)
def test_hpu_binary_cross_entropy_fwd_dynamic(
    size, reduction, dtype, use_weight, broadcastable_weight, setup_teardown_env_fixture
):

    if is_pytest_mode_compile() and use_weight:
        pytest.skip(
            "Due to improper handling of SymInts in PT 2.1, test fails on cpu when weights are used. Used to work on PT 2.0 - [SW-165520]"
        )
    if not is_pytest_mode_eager() and reduction == "none" and use_weight and broadcastable_weight:
        pytest.skip("Handling under SW-204408, bug already in regression")

    with use_eager_fallback():

        binary_cross_entropy_fwd_test(
            size,
            reduction,
            dtype,
            use_weight,
            broadcastable_weight=broadcastable_weight,
            is_dynamic=True,
            is_compile=is_pytest_mode_compile(),
        )


@pytest.mark.parametrize("size", size, ids=format_tc)
@pytest.mark.parametrize("reduction", reduction, ids=format_tc)
@pytest.mark.parametrize("dtype", dtype, ids=format_tc)
@pytest.mark.parametrize("use_weight, broadcastable_weight", use_weight_broadcastable_weight, ids=format_tc)
def test_hpu_binary_cross_entropy_with_logits_fwd(size, reduction, dtype, use_weight, broadcastable_weight):

    if not is_pytest_mode_eager() and reduction == "none" and use_weight and broadcastable_weight:
        pytest.skip("Handling under SW-204408, bug already in regression")

    with use_eager_fallback():

        binary_cross_entropy_with_logits_fwd_test(
            size,
            reduction,
            dtype,
            use_weight,
            broadcastable_weight=broadcastable_weight,
            is_compile=is_pytest_mode_compile(),
        )


@pytest.mark.skipif(is_pytest_mode_eager(), reason="DS are not supported in eager mode")
@pytest.mark.parametrize("size", size, ids=format_tc)
@pytest.mark.parametrize("reduction", reduction, ids=format_tc)
@pytest.mark.parametrize("dtype", dtype, ids=format_tc)
@pytest.mark.parametrize("use_weight, broadcastable_weight", use_weight_broadcastable_weight, ids=format_tc)
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 1}],
    indirect=True,
)
def test_hpu_binary_cross_entropy_with_logits_fwd_dynamic(
    size, reduction, dtype, use_weight, broadcastable_weight, setup_teardown_env_fixture
):

    if not is_pytest_mode_eager() and reduction == "none" and use_weight and broadcastable_weight:
        pytest.skip("Handling under SW-204408, bug already in regression")

    with use_eager_fallback():

        binary_cross_entropy_with_logits_fwd_test(
            size,
            reduction,
            dtype,
            use_weight,
            broadcastable_weight=broadcastable_weight,
            is_dynamic=True,
            is_compile=is_pytest_mode_compile(),
        )


# <--- Backward --->


@pytest.mark.parametrize("size", size, ids=format_tc)
@pytest.mark.parametrize("reduction", reduction, ids=format_tc)
@pytest.mark.parametrize("dtype", dtype, ids=format_tc)
@pytest.mark.parametrize("use_weight", [False, True], ids=format_tc)
def test_hpu_binary_cross_entropy_bwd(size, reduction, dtype, use_weight):

    binary_cross_entropy_bwd_test(size, reduction, dtype, use_weight, is_compile=is_pytest_mode_compile())


@pytest.mark.skipif(is_pytest_mode_eager(), reason="DS are not supported in eager mode")
@pytest.mark.parametrize("size", size, ids=format_tc)
@pytest.mark.parametrize("reduction", reduction, ids=format_tc)
@pytest.mark.parametrize("dtype", dtype, ids=format_tc)
@pytest.mark.parametrize("use_weight", [False, True], ids=format_tc)
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 1}],
    indirect=True,
)
@pytest.mark.skipif(True, reason="Will be resolved under SW-159211")
def test_hpu_binary_cross_entropy_bwd_dynamic(size, reduction, dtype, use_weight, setup_teardown_env_fixture):

    if is_pytest_mode_compile() and use_weight:
        pytest.skip(
            "Due to improper handling of SymInts in PT 2.1, test fails on cpu when weights are used. Used to work on PT 2.0 - [SW-165520]"
        )

    with use_eager_fallback():

        binary_cross_entropy_bwd_test(
            size, reduction, dtype, use_weight, is_dynamic=True, is_compile=is_pytest_mode_compile()
        )
