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
from test_utils import hpu


@pytest.mark.skip(reason="Error when trying to cast Long to Int, Input values range exceeds Int range")
def test_long_to_int_expect_fail_positive():
    max_long = torch.iinfo(torch.long).max

    in_tensor = torch.LongTensor([1, max_long, -5])
    print(in_tensor)
    with pytest.raises(RuntimeError) as e_info:
        in_tensor.to(hpu)
    assert str(e_info.value) == "Error when trying to cast Long to Int, Input values range exceeds Int range"


@pytest.mark.skip(reason="Error when trying to cast Long to Int, Input values range exceeds Int range")
def test_long_to_int_expect_fail_negative():
    max_long = torch.iinfo(torch.long).max

    in_tensor = torch.LongTensor([1, -max_long, -5])
    print(in_tensor)
    with pytest.raises(RuntimeError) as e_info:
        in_tensor.to(hpu)
    assert str(e_info.value) == "Error when trying to cast Long to Int, Input values range exceeds Int range"


def test_long_to_int_expect_pass():
    max_int = torch.iinfo(torch.int32).max

    in_tensor = torch.LongTensor([1, max_int, -max_int, -5])
    print(in_tensor)
    in_tensor.to(hpu)


@pytest.mark.skip(reason="Error when trying to cast Double to Float, Input values range exceeds Float range")
def test_double_to_float_expect_fail_positive():
    max_double = torch.finfo(torch.double).max

    in_tensor = torch.DoubleTensor([1, max_double, -5])
    print(in_tensor)
    with pytest.raises(RuntimeError) as e_info:
        in_tensor.to(hpu)
    assert str(e_info.value) == "Error when trying to cast Double to Float, Input values range exceeds Float range"


@pytest.mark.skip(reason="Error when trying to cast Double to Float, Input values range exceeds Float range")
def test_double_to_float_expect_fail_negative():
    max_double = torch.finfo(torch.double).max

    in_tensor = torch.DoubleTensor([1, -max_double, -5])
    print(in_tensor)
    with pytest.raises(RuntimeError) as e_info:
        in_tensor.to(hpu)
    assert str(e_info.value) == "Error when trying to cast Double to Float, Input values range exceeds Float range"


def test_double_to_float_expect_pass():
    max_float = torch.finfo(torch.float32).max
    print(max_float)
    in_tensor = torch.DoubleTensor([1, max_float, -max_float, -5])
    print(in_tensor)
    in_tensor.to(hpu)


def test_inf_nan_double_to_float_expect_pass():
    in_tensor = torch.DoubleTensor([float("inf"), -float("inf"), float("nan")])
    print("in cpu : ", in_tensor)
    hpu_tensor = in_tensor.to(hpu)
    print("in hpu : ", hpu_tensor)
    print(torch.eq(in_tensor, hpu_tensor))
