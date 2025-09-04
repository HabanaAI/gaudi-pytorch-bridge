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

import torch
from test_utils import compile_function_if_compile_mode


def func(input1, input2, input3):
    a = input1 + 1.0
    b = input2 + 2.0
    return a + b


def test_recipe_cache():
    input1 = torch.ones(10, 10, device="hpu")
    input3 = torch.ones(10, 10, device="hpu")
    input2 = input1[:, 9:10]
    compiled_fn = compile_function_if_compile_mode(func)
    op1 = compiled_fn(input1, input2, input1).cpu()
    op2 = compiled_fn(input3, input2, input1).cpu()
    assert torch.allclose(op1, op2)


if __name__ == "__main__":
    test_recipe_cache()
