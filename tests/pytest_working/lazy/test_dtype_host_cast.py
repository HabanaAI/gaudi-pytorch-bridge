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


if __name__ == "__main__":
    test_long_to_int_expect_fail_positive()
    test_long_to_int_expect_fail_negative()
    test_long_to_int_expect_pass()
    test_double_to_float_expect_fail_positive()
    test_double_to_float_expect_fail_negative()
    test_double_to_float_expect_pass()
    test_inf_nan_double_to_float_expect_pass()
