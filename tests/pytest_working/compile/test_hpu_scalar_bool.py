###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
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
