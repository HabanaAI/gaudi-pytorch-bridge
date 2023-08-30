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
import torch
import pytest
import os
from test_utils import env_var_in_scope

pytestmark = pytest.mark.skip(reason="Tests in this file are chaning env variables")


@pytest.mark.xfail(
    reason="CI problem: undefined symbol: _ZN6habana5graph12GraphStorage3getEv [SW-150162]"
)
@pytest.mark.parametrize("shape", [(4, 6, 8), (8, 8, 4, 16)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_softmax_retain_fwd(shape, dtype):
    with env_var_in_scope(PT_HPU_LAZY_MODE="0"):
        import habana_frameworks.torch.dynamo._custom_op_meta_registrations

        torch.manual_seed(12345)

        def raw_function(input):
            output, max, sum_exp = torch.ops.hpu.retain_softmax_producer(input)
            result_quick = torch.ops.hpu.retain_softmax_consumer(input, max, sum_exp)
            return output, result_quick

        compiled_function = torch.compile(
            raw_function, backend="aot_hpu_training_backend"
        )

        input = torch.randn(shape, dtype=dtype).to("hpu")
        output, result_quick = compiled_function(input)

        assert torch.equal(output, result_quick)
