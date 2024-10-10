###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
from contextlib import contextmanager

import pytest
import torch
import torch.nn as nn
from habana_frameworks.torch.dynamo.compile_backend.config import configuration_flags


# Fixture to set the environment variable
@pytest.fixture
def set_env(request, arg=False):
    os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = "/tmp/cache,false,8192"
    print("Enabled serialization of recipe on disk")
    is_eager_fallback = configuration_flags["use_eager_fallback"]
    configuration_flags["use_eager_fallback"] = True
    # Yield to provide the value for the test
    yield "1"

    os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = ""
    configuration_flags["use_eager_fallback"] = is_eager_fallback


def test_recipe_cache1(set_env):
    import habana_frameworks.torch.core as htcore

    input_shapes = [
        [(3, 6, 4), (3, 24)],
        [(3, 8, 4), (3, 32)],
        [(3, 10, 4), (3, 40)],
    ]

    def raw_function(t1, x2):
        t = t1.shape
        t1 = torch.relu(t1)
        shape = (t[0], int(t[1] * t[2]))
        t2 = t1.reshape(shape)
        t3 = torch.add(t2, x2)
        return t3

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    def execute_model(input_shapes):
        for s in input_shapes:
            t1 = torch.randn(s[0], requires_grad=False)
            t2 = torch.randn(s[1], requires_grad=False)
            result = raw_function(t1, t2)
            t1_h = t1.to("hpu")
            t2_h = t2.to("hpu")
            h_result = compiled_fn(t1_h, t2_h)
            assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)

    execute_model(input_shapes)

    # Receipes are cached in the previous run, expected to see cache hits
    # for static and dynamic recipes.
    import habana_frameworks.torch.utils.debug as htdebug

    htdebug._bridge_cleanup()

    execute_model(input_shapes)
