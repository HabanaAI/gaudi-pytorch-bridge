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

from unittest.mock import patch

import pytest
import torch


@torch.compile(backend="hpu_backend")
def fn(x):
    return x + x


# this test checks only that mode arg provided to torch_compile is not throwing in case of 'hpu_backend'
# there is no actual effect of this parameter in our case
def test_compile_mode_nothrow():
    def fn(x):
        return x + x

    compiled_fn = torch.compile(fn, backend="hpu_backend", mode="anything")
    compiled_fn(torch.tensor(2.0).to("hpu"))


def test_compile_config_use_compiled_recipes():
    def fn(x):
        return x + x

    input = torch.tensor(2.0).to("hpu")
    with patch(
        "habana_frameworks.torch.dynamo.compile_backend.recipe_compiler.HabanaGraphModule.__call__", return_value=None
    ) as mock_my_function:
        # when use_compiled_recipes=False, HabanaGraphModule should not be used at all
        torch.compile(fn, backend="hpu_backend", options={"use_compiled_recipes": False})(input)
        mock_my_function.assert_not_called()
        torch.compile(fn, backend="hpu_backend")(input)  # default for use_compiled_recipes is True
        mock_my_function.assert_called()


# testing if inner compiler is called as expected
class TestInnerCompiler:
    def test_inference_compiler_called(self):
        x = torch.tensor(2.0).to("hpu")

        with patch("habana_frameworks.torch.dynamo.compile_backend.compilers.hpu_compiler_inner") as mock_my_function:
            res = fn(x)
            mock_my_function.assert_called_once()
            # inference is called with _, _, is_training=False, is_backward=False
            assert mock_my_function.call_args.args[2:5] == (False, False)

    def test_fwd_compiler_called(self):
        x = torch.tensor(2.0, requires_grad=True).to("hpu")

        with patch("habana_frameworks.torch.dynamo.compile_backend.compilers.hpu_compiler_inner") as mock_my_function:
            res = fn(x)
            mock_my_function.assert_called_once()
            # fwd training is called with _, _, is_training=True, is_backward=False
            assert mock_my_function.call_args.args[2:5] == (True, False)

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
            # bwd training is called with _, _, is_training=True, is_backward=True
            assert mock_my_function.call_args.args[2:5] == (True, True)
