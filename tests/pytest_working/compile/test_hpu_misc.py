###############################################################################
# Copyright (c) 2025 Intel Corporation
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

import habana_frameworks.torch.internal.bridge_config as bc
import torch
from test_utils import compile_function_if_compile_mode, is_pytest_mode_compile


def test_hpu_disallow_torch_compile():
    def fn(x):
        return x + 1

    x_hpu = torch.randn(3).to("hpu")

    # Set flag not allowing compile
    with bc.env_setting("PT_HPU_DISALLOW_TORCH_COMPILE", True):
        assert bc.get_pt_hpu_disallow_torch_compile() is True

        # No assert since this is pure eager
        exception_raised = False
        try:
            hpu_res = fn(x_hpu)
        except RuntimeError as e:
            exception_raised = True
        assert not exception_raised, "No exception is expected for pure eager"

        # Assert if compile is used for eager
        exception_raised = False
        try:
            hpu_res = compile_function_if_compile_mode(fn)(x_hpu)
        except RuntimeError as e:
            exception_raised = True

        if is_pytest_mode_compile():
            assert exception_raised, "Expected exception is not raised for eager with compile"
        else:
            assert not exception_raised, "No exception is expected for pure eager"

    # Set flag allowing compile
    with bc.env_setting("PT_HPU_DISALLOW_TORCH_COMPILE", False):
        assert bc.get_pt_hpu_disallow_torch_compile() is False

        # No assert if compile is used for eager allowing compile
        exception_raised = False
        try:
            hpu_res = compile_function_if_compile_mode(fn)(x_hpu)
        except RuntimeError as e:
            exception_raised = True

        assert not exception_raised, "No exception is expected for eager allowing compile"


def test_hpu_fallback_random_unused_input(monkeypatch):
    monkeypatch.setattr(torch._inductor.config, "fallback_random", True)
    device = "hpu"

    def fn(t1, t2):
        t3 = torch.empty(1, device=device)
        t3.uniform_(0, 1)
        return t1.add(t2)

    t1 = torch.ones(1, device=device)
    t2 = torch.ones(1, device=device) * 2

    compiled_fn = compile_function_if_compile_mode(fn)
    result = compiled_fn(t1, t2)

    assert torch.allclose(result, t1 + t2)
