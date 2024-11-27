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

import numpy as np
import pytest
import torch
from test_utils import format_tc, is_pytest_mode_compile, is_pytest_mode_eager, setup_teardown_env_fixture


@pytest.mark.skipif(
    is_pytest_mode_compile(), reason="In compile mode, a custom backend is used istead of Bernoulli with p"
)
@pytest.mark.parametrize("shape", [(3, 4), (2, 5, 6)], ids=format_tc)
@pytest.mark.parametrize("p", [0.5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=format_tc)
def test_bernouli_with_p_float_different_seed(shape, p, dtype):
    input = torch.empty(shape, dtype=dtype).uniform_(0, 1).to("hpu")

    torch.manual_seed(12345)
    result_1 = torch.bernoulli(input, p).cpu()
    torch.manual_seed(12346)
    result_2 = torch.bernoulli(input, p).cpu()
    assert not torch.equal(result_1, result_2)


@pytest.mark.skipif(
    is_pytest_mode_compile(), reason="In compile mode, a custom backend is used istead of Bernoulli with p"
)
@pytest.mark.parametrize("shape", [(3, 4), (2, 5, 6)], ids=format_tc)
@pytest.mark.parametrize("p", [0.5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=format_tc)
def test_bernouli_with_p_float_same_seed(shape, p, dtype):
    input = torch.empty(shape, dtype=dtype).uniform_(0, 1).to("hpu")

    torch.manual_seed(12345)
    result_1 = torch.bernoulli(input, p).cpu()
    torch.manual_seed(12345)
    result_2 = torch.bernoulli(input, p).cpu()
    assert torch.equal(result_1, result_2)


@pytest.mark.skipif(
    is_pytest_mode_compile(), reason="In compile mode, a custom backend is used istead of Bernoulli with p"
)
@pytest.mark.skipif(is_pytest_mode_eager(), reason="In eager mode, DSD is not supported")
@pytest.mark.parametrize("shape", [[3, 4]], ids=format_tc)
@pytest.mark.parametrize("p", [0.5])
@pytest.mark.parametrize("dtype", [torch.float32], ids=format_tc)
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 1}],
    indirect=True,
)
def test_bernouli_with_p_float_different_seed_dynamic(shape, p, dtype, setup_teardown_env_fixture):
    shapes = [shape, np.multiply(shape, 2), np.multiply(shape, 3), np.multiply(shape, 4)]
    results_1 = []
    results_2 = []
    for i in range(len(shapes)):
        print(i, "\n\n")
        input = torch.empty(tuple(shapes[i]), dtype=dtype).uniform_(0, 1).to("hpu")
        torch.manual_seed(12345)
        result_1 = torch.bernoulli(input, p)
        results_1.append(result_1)
        torch.manual_seed(12346)
        result_2 = torch.bernoulli(input, p)
        results_2.append(result_2)
        assert not torch.equal(result_1.to("cpu"), result_2.to("cpu"))


@pytest.mark.skipif(
    is_pytest_mode_compile(), reason="In compile mode, a custom backend is used istead of Bernoulli with p"
)
@pytest.mark.skipif(is_pytest_mode_eager(), reason="In eager mode, DSD is not supported")
@pytest.mark.parametrize("shape", [[3, 4]], ids=format_tc)
@pytest.mark.parametrize("p", [0.5])
@pytest.mark.parametrize("dtype", [torch.float32], ids=format_tc)
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 1}],
    indirect=True,
)
def test_bernouli_with_p_float_p_changed_in_last_iter_dynamic(shape, p, dtype, setup_teardown_env_fixture):
    shapes = [shape, np.multiply(shape, 2), np.multiply(shape, 3), np.multiply(shape, 4)]
    results_1 = []
    results_2 = []
    for i in range(len(shapes)):
        input = torch.empty(tuple(shapes[i]), dtype=dtype).uniform_(0, 1).to("hpu")
        torch.manual_seed(12345)
        result_1 = torch.bernoulli(input, p)
        results_1.append(result_1)
        torch.manual_seed(12346 if i != 3 else 12345)
        result_2 = torch.bernoulli(input, p if i != 3 else 0)
        results_2.append(result_2)
        assert not torch.equal(result_1.to("cpu"), result_2.to("cpu"))


@pytest.mark.skipif(
    is_pytest_mode_compile(), reason="In compile mode, a custom backend is used istead of Bernoulli with p"
)
@pytest.mark.parametrize("shape", [(3, 4), (2, 5, 6)], ids=format_tc)
@pytest.mark.parametrize("p", [0.5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=format_tc)
def test_bernouli_input_as_p_different_seed(shape, p, dtype):
    input = torch.empty(shape, dtype=dtype).fill_(p).to("hpu")

    torch.manual_seed(12345)
    result_1 = torch.bernoulli(input).cpu()
    torch.manual_seed(12346)
    result_2 = torch.bernoulli(input).cpu()
    assert not torch.equal(result_1, result_2)


@pytest.mark.skipif(
    is_pytest_mode_compile(), reason="In compile mode, a custom backend is used istead of Bernoulli with p"
)
@pytest.mark.parametrize("shape", [(3, 4), (2, 5, 6)], ids=format_tc)
@pytest.mark.parametrize("p", [0.5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=format_tc)
def test_bernouli_input_as_p_same_seed(shape, p, dtype):
    input = torch.empty(shape, dtype=dtype).fill_(p).to("hpu")

    torch.manual_seed(12345)
    result_1 = torch.bernoulli(input).cpu()
    torch.manual_seed(12345)
    result_2 = torch.bernoulli(input).cpu()
    assert torch.equal(result_1, result_2)


@pytest.mark.skipif(
    is_pytest_mode_compile(), reason="In compile mode, a custom backend is used istead of Bernoulli with p"
)
@pytest.mark.skipif(is_pytest_mode_eager(), reason="In eager mode, DSD is not supported")
@pytest.mark.parametrize("shape", [[3, 4]], ids=format_tc)
@pytest.mark.parametrize("p", [0.5])
@pytest.mark.parametrize("dtype", [torch.float32], ids=format_tc)
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 1}],
    indirect=True,
)
def test_bernouli_input_as_p_different_seed_dynamic(shape, p, dtype, setup_teardown_env_fixture):
    shapes = [shape, np.multiply(shape, 2), np.multiply(shape, 3), np.multiply(shape, 4)]
    results_1 = []
    results_2 = []
    for i in range(len(shapes)):
        input = torch.empty(shape, dtype=dtype).fill_(p).to("hpu")
        torch.manual_seed(12345)
        result_1 = torch.bernoulli(input)
        results_1.append(result_1)
        torch.manual_seed(12346)
        result_2 = torch.bernoulli(input)
        results_2.append(result_2)
        assert not torch.equal(result_1.to("cpu"), result_2.to("cpu"))


@pytest.mark.skipif(
    is_pytest_mode_compile(), reason="In compile mode, a custom backend is used istead of Bernoulli with p"
)
@pytest.mark.skipif(is_pytest_mode_eager(), reason="In eager mode, DSD is not supported")
@pytest.mark.parametrize("shape", [[3, 4]], ids=format_tc)
@pytest.mark.parametrize("p", [0.5])
@pytest.mark.parametrize("dtype", [torch.float32], ids=format_tc)
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 1}],
    indirect=True,
)
def test_bernouli_input_as_p_p_changed_in_last_iter_dynamic(shape, p, dtype, setup_teardown_env_fixture):
    shapes = [shape, np.multiply(shape, 2), np.multiply(shape, 3), np.multiply(shape, 4)]
    results_1 = []
    results_2 = []
    changed_input = torch.empty(shape, dtype=dtype).fill_(0).to("hpu")
    for i in range(len(shapes)):
        input = torch.empty(shape, dtype=dtype).fill_(p).to("hpu")
        torch.manual_seed(12345)
        result_1 = torch.bernoulli(input)
        results_1.append(result_1)
        torch.manual_seed(12346 if i != 3 else 12345)
        result_2 = torch.bernoulli(input if i != 3 else changed_input)
        results_2.append(result_2)
        assert not torch.equal(result_1.to("cpu"), result_2.to("cpu"))
