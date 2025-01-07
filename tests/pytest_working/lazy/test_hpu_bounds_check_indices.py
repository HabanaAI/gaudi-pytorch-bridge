###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
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

import enum
import unittest

import numpy as np
import pytest
import torch
from habana_frameworks.torch.hpex.kernels.fbgemm import bounds_check_indices
from numpy.testing import assert_array_equal, assert_raises
from test_utils import cpu, generic_setup_teardown_env, hpu

pytestmark = pytest.mark.skip(reason="Tests in this file are chaning env variables")


@pytest.fixture(autouse=True, scope="module")
def setup_teardown_env():
    yield from generic_setup_teardown_env({"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 0})


class BoundsCheckMode(enum.IntEnum):
    # Raise an exception (CPU) or device-side assert (HPU)
    FATAL = 0
    # Log the first out-of-bounds instance per kernel, and set to zero.
    WARNING = 1
    # Set to zero.
    IGNORE = 2
    # No bounds checks.
    NONE = 3


class TestCase(enum.IntEnum):
    __test__ = False
    # Check that indexes are the same.
    CHECK_ALL_INDICES_THE_SAME = 0
    # Check that the indexes are not the same.
    CHECK_INDICES_NOT_THE_SAME = 1
    # Check that all indexes are zero.
    CHECK_ALL_INDICES_ZERO = 2
    # Test offsets bound errors.
    TEST_OFFSETS_BOUND_ERRORS = 3


bounds_check_test_case_list = [
    # T, B, max_L, bounds_check_mode, weighted, test_case, dtype
    (
        4,
        8,
        8,
        BoundsCheckMode.NONE,
        True,
        TestCase.CHECK_ALL_INDICES_THE_SAME,
        torch.int32,
    ),
    (
        2,
        4,
        8,
        BoundsCheckMode.NONE,
        False,
        TestCase.CHECK_ALL_INDICES_THE_SAME,
        torch.int32,
    ),
    (
        4,
        8,
        8,
        BoundsCheckMode.WARNING,
        True,
        TestCase.CHECK_INDICES_NOT_THE_SAME,
        torch.int32,
    ),
    (
        4,
        8,
        8,
        BoundsCheckMode.FATAL,
        True,
        TestCase.CHECK_INDICES_NOT_THE_SAME,
        torch.int32,
    ),
    (
        2,
        4,
        8,
        BoundsCheckMode.WARNING,
        True,
        TestCase.CHECK_ALL_INDICES_ZERO,
        torch.int32,
    ),
    (
        4,
        8,
        8,
        BoundsCheckMode.FATAL,
        True,
        TestCase.CHECK_ALL_INDICES_ZERO,
        torch.int32,
    ),
    (
        2,
        4,
        4,
        BoundsCheckMode.WARNING,
        True,
        TestCase.TEST_OFFSETS_BOUND_ERRORS,
        torch.int32,
    ),
    (
        4,
        8,
        8,
        BoundsCheckMode.FATAL,
        True,
        TestCase.TEST_OFFSETS_BOUND_ERRORS,
        torch.int32,
    ),
    (
        4,
        8,
        8,
        BoundsCheckMode.NONE,
        True,
        TestCase.CHECK_ALL_INDICES_THE_SAME,
        torch.int64,
    ),
    (
        2,
        4,
        8,
        BoundsCheckMode.NONE,
        False,
        TestCase.CHECK_ALL_INDICES_THE_SAME,
        torch.int64,
    ),
    (
        4,
        8,
        8,
        BoundsCheckMode.WARNING,
        True,
        TestCase.CHECK_INDICES_NOT_THE_SAME,
        torch.int64,
    ),
    (
        4,
        8,
        8,
        BoundsCheckMode.FATAL,
        True,
        TestCase.CHECK_INDICES_NOT_THE_SAME,
        torch.int64,
    ),
    (
        2,
        4,
        4,
        BoundsCheckMode.WARNING,
        True,
        TestCase.TEST_OFFSETS_BOUND_ERRORS,
        torch.int64,
    ),
    (
        4,
        8,
        8,
        BoundsCheckMode.FATAL,
        True,
        TestCase.TEST_OFFSETS_BOUND_ERRORS,
        torch.int64,
    ),
]


def assert_array_not_equal(x, y):
    return assert_raises(AssertionError, assert_array_equal, x, y)


@pytest.mark.skip
@pytest.mark.parametrize(
    "T, B, max_L, bounds_check_mode, weighted, test_case, dtype",
    bounds_check_test_case_list,
)
def test_bounds_check(T, B, max_L, bounds_check_mode, weighted, test_case, dtype):
    if test_case != test_case.CHECK_INDICES_NOT_THE_SAME:
        rows_per_table = torch.tensor(np.random.randint(low=1, high=1000, size=(T,))).long()
        Ls = np.random.randint(low=0, high=max_L, size=(T, B))
        indices = [np.random.randint(low=0, high=rows_per_table[t], size=Ls[t, b]) for t in range(T) for b in range(B)]
        indices = torch.tensor(np.concatenate(indices, axis=0)).to(dtype)
        weights = torch.rand(indices.shape, dtype=torch.float, device=indices.device) if weighted else None
        offsets = torch.tensor([0] + np.cumsum(Ls.flatten()).tolist()).to(dtype)

        rows_per_table = rows_per_table.to(hpu)
        offsets = offsets.to(hpu)
    else:
        rows_per_table = torch.tensor([45, 344]).to(hpu)
        indices = torch.tensor(
            [
                44,
                3,
                41,
                12,
                45,
                13,
                32,
                29,
                7,
                34,
                21,
                20,
                43,
                30,
                32,
                21,
                29,
                198,
                309,
                55,
                237,
                196,
                128,
                122,
                28,
                246,
                170,
                252,
                243,
                11,
                230,
                35,
                41,
                111,
                142,
                147,
                11,
                170,
            ]
        )
        offsets = torch.tensor([0, 4, 6, 12, 16, 23, 30, 38, 38]).to(hpu)
        warning_expected = 1

    if weighted:
        weights = indices.float().clone().to(hpu)

    warning = torch.tensor([0]).to(hpu)

    if test_case == TestCase.CHECK_ALL_INDICES_ZERO:
        indices[:] = torch.iinfo(dtype).max
    else:
        if test_case == TestCase.TEST_OFFSETS_BOUND_ERRORS:
            if offsets.numel() > 0:
                offsets[0] = -100
            if offsets.numel() > 1:
                offsets[-1] += 100

    indices_copy = indices.clone()

    indices = indices.to(hpu)

    bounds_check_indices(
        rows_per_table,
        indices,
        offsets,
        bounds_check_mode,
        warning,
        weights if weighted else None,
    )

    if test_case == TestCase.CHECK_ALL_INDICES_THE_SAME:
        torch.testing.assert_close(indices_copy, indices.to(cpu))
    elif test_case == TestCase.CHECK_INDICES_NOT_THE_SAME:
        assert_array_not_equal(indices_copy, indices.to(cpu))

        if bounds_check_mode == BoundsCheckMode.WARNING:
            torch.testing.assert_close(warning.to(cpu).numpy()[0], warning_expected)
        elif bounds_check_mode == BoundsCheckMode.FATAL:
            np.testing.assert_array_less(torch.min(indices).to(cpu), 0)
    elif test_case == TestCase.CHECK_ALL_INDICES_ZERO:
        if bounds_check_mode != BoundsCheckMode.FATAL:
            torch.testing.assert_close(indices, torch.zeros_like(indices))

            if bounds_check_mode == BoundsCheckMode.WARNING:
                torch.testing.assert_close(warning.to(cpu).numpy()[0], indices.numel())
        elif indices.numel():
            np.testing.assert_array_less(torch.min(indices).to(cpu), 0)
    elif test_case == TestCase.TEST_OFFSETS_BOUND_ERRORS:
        if bounds_check_mode != BoundsCheckMode.FATAL:
            if offsets.numel() > 0:
                torch.testing.assert_close(offsets[0].to(cpu).item(), 0)

            if offsets.numel() > 1:
                torch.testing.assert_close(offsets[-1].item(), indices.numel())

            if bounds_check_mode == BoundsCheckMode.WARNING:
                unittest.TestCase().assertGreaterEqual(warning.item(), min(2, offsets.numel() - 1))
        else:
            np.testing.assert_array_less(torch.min(indices).to(cpu), 0)
