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

import random

import pytest
import torch
from habana_frameworks.torch.utils.debug import flush_permute_cache, get_permute_cache_size


@pytest.fixture
def setup_permute_cache_test():
    flush_permute_cache()


def test_jit_shapeless_hash_include_permutations(setup_permute_cache_test):
    """
    This test checks whether permutations on inputs are taken into account when calculating
    Shapeless hash used for permute caching. Convolution operation performs permutations
    on output tensor. Second mul should not cause a PermuteCache CACHE HIT
    """
    m = torch.nn.Conv2d(2, 2, 1, device="hpu")
    inp = torch.randn((2, 2, 8, 8), device="hpu")

    compiled_mul = torch.compile(torch.mul, backend="hpu_backend")
    compiled_conv = torch.compile(m, backend="hpu_backend")

    r1 = compiled_mul(inp, 2.0)
    r2 = compiled_conv(r1)
    rf = compiled_mul(r2, 2.0)
    rf.to("cpu")
    assert get_permute_cache_size() == 3, "Cache hit should not have occured"


def test_permutations_reused_for_same_shapeless_base_graph(setup_permute_cache_test):
    random.seed(42)

    POSSIBLE_BATCH_SIZES = [1, 2, 4, 8]
    POSSIBLE_HW_SIZES = [2, 4, 8, 16, 32]
    KERNEL_SIZE = 1
    CHANNELS_IN = 2
    CHANNELS_OUT = 2
    NUMBER_OF_INPUTS_GENERATIONS = 32

    def prepare_inputs():
        inputs = []
        for _ in range(NUMBER_OF_INPUTS_GENERATIONS):
            n_idx = random.randint(0, len(POSSIBLE_BATCH_SIZES) - 1)
            h_idx = random.randint(0, len(POSSIBLE_HW_SIZES) - 1)
            w_idx = random.randint(0, len(POSSIBLE_HW_SIZES) - 1)
            size = (POSSIBLE_BATCH_SIZES[n_idx], CHANNELS_IN, POSSIBLE_HW_SIZES[h_idx], POSSIBLE_HW_SIZES[w_idx])

            inputs.append(torch.randn(size).to("hpu"))

        return set(inputs)

    m = torch.nn.Conv2d(CHANNELS_IN, CHANNELS_OUT, KERNEL_SIZE, device="hpu")
    m_compiled = torch.compile(m, backend="hpu_backend", dynamic=False)

    inputs = prepare_inputs()
    for i in inputs:
        res = m_compiled(i)
        res.to("cpu")
    assert get_permute_cache_size() == 1, "Wrong number of cached permutations"
