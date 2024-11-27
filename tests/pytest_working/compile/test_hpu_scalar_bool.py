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

import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
from habana_frameworks.torch.dynamo.compile_backend.config import configuration_flags


class TestHpuScalarBool:
    @classmethod
    def setup_class(self):
        # For mul op there is expected fallback to eager
        self.original_configuration = configuration_flags["use_eager_fallback"]
        configuration_flags["use_eager_fallback"] = True

    @classmethod
    def teardown_class(self):
        configuration_flags["use_eager_fallback"] = self.original_configuration

    @staticmethod
    def test_scalar_bool():
        def fn(input, scalar):
            return torch.mul(input, scalar)

        cpu_input = torch.randint(low=0, high=2, size=(2, 2), dtype=torch.bool)
        hpu_input = cpu_input.to("hpu")
        torch._dynamo.reset()

        hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend")

        cpu_output = fn(cpu_input, True)
        hpu_output = hpu_wrapped_fn(hpu_input, True).cpu()

        assert torch.equal(cpu_output, hpu_output)
