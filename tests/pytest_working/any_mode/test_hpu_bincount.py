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


import pytest
import torch
from compile.test_dynamo_utils import use_eager_fallback
from test_utils import compare_tensors, compile_function_if_compile_mode, format_tc


@pytest.mark.parametrize("input_dtype", [torch.int32, torch.long], ids=format_tc)
@pytest.mark.parametrize("weights_dtype", [torch.int32, torch.float32, torch.double, None], ids=format_tc)
@pytest.mark.parametrize("minlength", [0, 3, 50], ids=format_tc)
@pytest.mark.parametrize("shape", [(0,), (5,), (80,)], ids=format_tc)
def test_bincount(input_dtype, weights_dtype, minlength, shape):
    with use_eager_fallback():
        input = torch.randint(low=0, high=30, size=shape, dtype=input_dtype)
        input_hpu = input.to("hpu")
        if weights_dtype is None:
            weights = None
            weights_hpu = None
        else:
            if weights_dtype in [torch.int32, torch.long]:
                weights = torch.randint(low=0, high=30, size=shape, dtype=weights_dtype)
            else:
                weights = torch.randn(size=shape, dtype=weights_dtype) * 10
            weights_hpu = weights.to("hpu")

        def fn(input, weights, minlength):
            return torch.bincount(input, weights, minlength)

        fn_hpu = compile_function_if_compile_mode(fn)

        result_hpu = fn_hpu(input_hpu, weights_hpu, minlength)

        result_ref = fn(input, weights, minlength)

        compare_tensors(result_hpu, result_ref, atol=1e-5, rtol=1e-5)
        assert result_hpu.dtype == result_ref.dtype
