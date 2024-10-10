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
import os
import pickle

import pytest
import torch
from habana_frameworks.torch.dynamo.compile_backend.backends import hpu_backend
from torch._dynamo.backends.registry import register_backend


@pytest.mark.skip(reason="Test is non-deterministic. TODO create deterministic one.")
def test_reordered_outputs_with_cache():
    def fn(inp):
        x = torch.max(inp, dim=0)
        out1 = inp * 2
        x1 = x[0]
        x2 = x[1]
        out2 = x1 + x2

        return out1, x2, out2

    hpu_input = torch.rand([5, 5], device="hpu")
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    hpu_output = hpu_compiled_fn(hpu_input)
    hpu_output_cached_recipe = hpu_compiled_fn(hpu_input)

    for out, out_cached in zip(hpu_output, hpu_output_cached_recipe):
        assert torch.equal(out.cpu(), out_cached.cpu())
