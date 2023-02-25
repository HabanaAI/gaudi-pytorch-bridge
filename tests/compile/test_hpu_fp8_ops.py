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
import numpy as np
import habana_frameworks.torch.dynamo._custom_op_meta_registrations

@pytest.mark.parametrize("shape", [(64, 64)])
@pytest.mark.parametrize("scale", [0.75])
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("stochastic", [True])
def test_cast_to_fp8(shape, scale, dtype, stochastic):
    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype)*30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))
    scale = torch.tensor(scale, dtype=torch.float)

    amax = torch.empty((2, 3), dtype=torch.float).to(hpu)
    def fn(input_hpu, scale_hpu, amax_hpu, out):
        torch.ops.hpu.cast_to_fp8(input_hpu, scale_hpu, stochastic, out, amax_hpu)
        return out

    def toy_compiler(fx_module: torch.fx.GraphModule, example_inputs):
        print(fx_module.code)
        f = torch.jit.script(fx_module)
        print("JIT IR", f.graph)
        return fx_module

    compiled_fn = torch.compile(fn, backend=toy_compiler)

    input_hpu = input.to(hpu)
    scale_hpu = scale.to(hpu)
    amax_hpu = amax[1][2]
    out_hpu = torch.empty(input_hpu.shape, dtype=torch.int8, device=input_hpu.device)
    casted = compiled_fn(input_hpu, scale_hpu, amax_hpu, out_hpu)

