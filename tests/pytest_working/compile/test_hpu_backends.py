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

import pytest
import torch
from unittest.mock import patch


@torch.compile(backend="hpu_backend")
def fn(x):
    return x + x


# testing if inner compiler is called as expected
class TestInnerCompiler:
    def test_inference_compiler_called(self):
        x = torch.tensor(2.0).to("hpu")

        with patch("habana_frameworks.torch.dynamo.compile_backend.compilers.hpu_compiler_inner") as mock_my_function:
            res = fn(x)
            mock_my_function.assert_called_once()
            # inference is called with _, _, is_training=False, is_backward=False, uses_aot=True
            assert mock_my_function.call_args.args[2:5] == (False, False, True)

    def test_fwd_compiler_called(self):
        x = torch.tensor(2.0, requires_grad=True).to("hpu")

        with patch("habana_frameworks.torch.dynamo.compile_backend.compilers.hpu_compiler_inner") as mock_my_function:
            res = fn(x)
            mock_my_function.assert_called_once()
            # fwd training is called with _, _, is_training=True, is_backward=False, uses_aot=True
            assert mock_my_function.call_args.args[2:5] == (True, False, True)

    def test_bwd_compiler_called(self):
        x = torch.tensor(2.0, requires_grad=True).to("hpu")

        def fn(x):
            return x + x

        compiled_fn = torch.compile(fn, backend="hpu_backend")
        res = compiled_fn(x)

        with patch(
            "habana_frameworks.torch.dynamo.compile_backend.compilers.hpu_compiler_inner", return_value=lambda x: x
        ) as mock_my_function:
            res.backward()
            mock_my_function.assert_called_once()
            # bwd training is called with _, _, is_training=True, is_backward=True, uses_aot=True
            assert mock_my_function.call_args.args[2:5] == (True, True, True)
