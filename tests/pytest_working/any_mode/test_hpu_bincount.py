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
