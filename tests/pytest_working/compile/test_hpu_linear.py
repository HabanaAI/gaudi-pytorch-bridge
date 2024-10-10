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
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch

out_features = 10
in_features = 7


@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.half])
def test_linear(dtype):
    def fn(model, input):
        return model(input)

    # CPU
    input = torch.randn((2, 3, 4, in_features))
    h_input = input.to("hpu").detach().requires_grad_()
    model = torch.nn.Linear(in_features, out_features, True)
    result = fn(model, input)
    model = model.to("hpu")

    # HPU
    compiled_fn = torch.compile(fn, backend="hpu_backend")
    hresult = compiled_fn(model, h_input)

    assert torch.allclose(result, hresult.cpu(), atol=0.001, rtol=0.001)
