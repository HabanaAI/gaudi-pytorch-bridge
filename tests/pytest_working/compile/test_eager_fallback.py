###############################################################################
#
#  Copyright (c) 2025 Intel Corporation
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
from habana_frameworks.torch.dynamo.compile_backend.random_utils import (
    HABANA_RANDOM_OPS,
)
from test_utils import (
    check_eager_fallback_reason,
    check_eager_placement_reason,
    compile_function_if_compile_mode,
)


def test_fallback_ops_list():
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    fn = compile_function_if_compile_mode(torch.ops.aten._unique2, dynamic=False)

    shape = (6, 8)
    input = torch.randint(0, 10, shape, dtype=torch.int).to("hpu")

    with pytest.raises(Exception) as e_info:
        fn(input)

    torch._dynamo.config.capture_dynamic_output_shape_ops = False

    check_eager_fallback_reason("_unique2", "Default fallback based on hpu_fallback_op_list", exception=e_info)


def test_unsupported_dtype():
    fn = compile_function_if_compile_mode(torch.permute)

    shape = (2, 3, 4, 5)
    input = torch.randint(0, 10, shape, dtype=torch.long).to("hpu")

    with pytest.raises(Exception) as e_info:
        fn(input, (2, 0, 1, 3))

    check_eager_fallback_reason("permute", "Op not supported with dtype: torch.int64", exception=e_info)


def test_unsupported_random_activation_checkpoint():
    aten_op = "aten.bernoulli.default"
    checkpoint_op_bckp = HABANA_RANDOM_OPS.pop(aten_op)

    def op(input):
        return torch.ops.aten.bernoulli(input) * input

    def fn(input):
        return torch.utils.checkpoint.checkpoint(op, input, use_reentrant=False)

    fn = compile_function_if_compile_mode(fn)

    shape = (6, 8)
    input = torch.randn(shape).to("hpu").requires_grad_(True)

    with pytest.raises(Exception) as e_info:
        res = fn(input)
        res.sum().backward()

    HABANA_RANDOM_OPS[aten_op] = checkpoint_op_bckp

    check_eager_fallback_reason(
        "run_and_save_rng_state", f"Random op {aten_op} not supported in activation_checkpoint flow", exception=e_info
    )


def run_sdpa(dynamic):
    fn = compile_function_if_compile_mode(torch.ops.hpu.sdpa_recomp_fwd, dynamic=dynamic)

    q = torch.randn((3, 4, 12, 8)).to("hpu")
    k = torch.randn((3, 4, 8, 8)).to("hpu")
    v = torch.randn((3, 4, 8, 8)).to("hpu")

    fn(q, k, v, None, 0.4, 1.0, False, False, "fp32", None, "left")


def test_unsupported_dynamic_shapes():
    with pytest.raises(Exception) as e_info:
        run_sdpa(dynamic=True)

    check_eager_fallback_reason("sdpa_recomp_fwd_dropout", "Op not supported in dynamic shapes flow", exception=e_info)


def test_conditional_support():
    run_sdpa(dynamic=False)

    check_eager_fallback_reason(
        "sdpa_recomp_fwd_dropout", "Graph support based on hpu_supported_op_list", is_fallback=False
    )


def test_shared_layer_failed():
    fn = compile_function_if_compile_mode(torch.softmax)
    a = torch.randint(0, 10, (5, 5), dtype=torch.int).to("hpu")

    with pytest.raises(Exception) as e_info:
        fn(a, 0)

    check_eager_fallback_reason("_softmax", "Shared layer validation failed", exception=e_info)


def test_copy_ops():
    def fn(a, b):
        return torch.add(a, b.to("hpu"))

    fn = compile_function_if_compile_mode(fn)

    a = torch.randn((6, 8)).to("hpu")
    b = torch.randn((6, 8))

    with pytest.raises(Exception) as e_info:
        fn(a, b)

    check_eager_placement_reason(
        "_to_copy",
        "eager placement as node is a non D2D copy",
        exception=e_info,
        full_op_name="torch.ops.aten._to_copy.default",
    )


def test_host_function():
    model = torch.nn.Conv3d(3, 6, 3, stride=2).to("hpu")
    fn = compile_function_if_compile_mode(model)
    input = torch.randn(5, 3, 4, 5, 6).to("hpu")
    res = fn(input).cpu()

    check_eager_placement_reason("weight_permutation", "eager placement as node is a host call function")
