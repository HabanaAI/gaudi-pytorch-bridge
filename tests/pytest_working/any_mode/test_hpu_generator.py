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

import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    compile_function_if_compile_mode,
    format_tc,
    is_pytest_mode_compile,
)


@pytest.mark.parametrize("shape", [[2, 3]], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
@pytest.mark.parametrize("mean", [0], ids=format_tc)
@pytest.mark.parametrize("std", [5], ids=format_tc)
@pytest.mark.parametrize("seed", [1245], ids=format_tc)
def test_hpu_generator(shape, dtype, mean, std, seed):
    input = torch.empty(shape, dtype=dtype).to("hpu")

    def gen(**gen_kwargs):
        g = torch.Generator(**gen_kwargs)
        g.manual_seed(seed)
        return g

    g = gen(device="hpu")
    g2 = gen()

    def fn(input, g):
        output = input.normal_(mean=mean, std=std, generator=g)
        # normal_ executes eagerly in compile mode so to have anything in compiled graph
        # generate dummy addition
        return output + 0

    fn = compile_function_if_compile_mode(fn)

    output_hpu = fn(input, g).to("cpu")
    output_hpu_2 = fn(input, g2).to("cpu")

    torch.allclose(output_hpu, output_hpu_2, atol=0.001, rtol=1.0e-3)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("add")
