###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import os

import pytest
import torch


@pytest.mark.parametrize(
    "env_value, should_fail",
    [("False", False), ("True", True)],  # Expect the test to fail when the environment variable is "False"
)
def test_pass_fix_arange_device(env_value, should_fail):
    # Set the environment variable
    os.environ["PT_HPU_DISABLE_PASS_FIX_ARANGE_DEVICE"] = env_value

    @torch.compile(backend="hpu_backend")
    def func(x: torch.Tensor):
        return x[torch.arange(32)]

    x = torch.randn([64, 64], device="hpu")

    # Use xfail to mark the test as expected to fail when should_fail is True
    if should_fail:
        pytest.xfail("Expected failure when environment variable is set to 'False'")

    # Run the function and check for exceptions
    _ = func(x)

    # Clean up by deleting the environment variable
    del os.environ["PT_HPU_DISABLE_PASS_FIX_ARANGE_DEVICE"]
