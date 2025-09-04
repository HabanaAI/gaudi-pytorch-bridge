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


def test_hpu_cond_nested():
    def cond_fn(x):
        def outer_true_fn(x):
            def inner_true_fn(x):
                return x + 1

            def inner_false_fn(x):
                return x - 2

            return torch.cond(x.sum() > 2, inner_true_fn, inner_false_fn, (x,))

        def outer_false_fn(x):
            return x + 20

        x = torch.mul(x, 2.0)
        res = torch.cond(x.sum() > 2, outer_true_fn, outer_false_fn, (x,))
        return res

    x = torch.randn(4, 2)
    ref_res = cond_fn(x)
    aot_eager_res = compile_function_if_compile_mode(cond_fn, backend="aot_eager")(x)
    torch.allclose(ref_res, aot_eager_res)

    x_hpu = x.to("hpu")
    hpu_res = compile_function_if_compile_mode(cond_fn)(x_hpu)
    torch.allclose(hpu_res.cpu(), ref_res)
