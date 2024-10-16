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
from test_utils import check_ops_executed_in_jit_ir, compile_function_if_compile_mode, format_tc, is_pytest_mode_compile


@pytest.mark.parametrize("shape", [(20,), (5, 4, 3)], ids=format_tc)
def test_hpu_exp_fast_math(shape):
    self_cpu = torch.randn(shape, dtype=torch.bfloat16) * 10
    self_hpu = self_cpu.to("hpu")

    fn_hpu = compile_function_if_compile_mode(torch.ops.hpu.exp_fast_math)

    result_hpu = fn_hpu(self_hpu)
    result_cpu = torch.exp(self_cpu)

    torch.testing.assert_close(result_hpu.cpu(), result_cpu, atol=0.01, rtol=0.11)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("exp_fast_math")
