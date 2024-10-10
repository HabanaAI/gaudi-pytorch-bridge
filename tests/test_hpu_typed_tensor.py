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
from test_utils import is_gaudi1

test_values = (2, 3, 4)

test_type_list = [
    (torch.hpu.BFloat16Tensor, torch.BFloat16Tensor),
    (torch.hpu.BoolTensor, torch.BoolTensor),
    (torch.hpu.ByteTensor, torch.ByteTensor),
    (torch.hpu.CharTensor, torch.CharTensor),
    (torch.hpu.DoubleTensor, torch.DoubleTensor),
    (torch.hpu.FloatTensor, torch.FloatTensor),
    (torch.hpu.HalfTensor, torch.HalfTensor),
    (torch.hpu.IntTensor, torch.IntTensor),
    (torch.hpu.LongTensor, torch.LongTensor),
    (torch.hpu.ShortTensor, torch.ShortTensor),
]


def _annotate_type_list(testcase):
    r"""
    Should provide type name as
        Test_hpu_type_alias_tensor::test_correct_device[int64]
    instead
        Test_hpu_type_alias_tensor::test_correct_device[type8]
    """

    torch_type = testcase[1]
    torch_dtype = torch_type.dtype
    result = str(torch_dtype).rsplit(".")[-1]

    return result


@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported dtype")
@pytest.mark.parametrize("type", test_type_list, ids=_annotate_type_list)
class Test_hpu_type_alias_tensor:
    def test_dtype(self, type):
        hpu_type = type[0](test_values)
        torch_type = type[1](test_values)

        assert hpu_type.dtype == torch_type.dtype

    def test_device(self, type):
        hpu = torch.device("hpu", 0)
        hpu_type = type[0](test_values)

        assert hpu_type.device == hpu

    def test_instance(self, type):
        hpu_type = type[0](test_values)

        assert isinstance(hpu_type, type[0])
        assert isinstance(hpu_type, torch.Tensor)
        assert not isinstance(hpu_type, type[1])

    def test_shape(self, type):
        hpu_type = type[0](test_values)
        torch_type = type[1](test_values)

        assert hpu_type.shape == torch_type.shape

    def test_size(self, type):
        hpu_type = type[0](test_values)
        torch_type = type[1](test_values)

        assert hpu_type.size() == torch_type.size()

    def test_correct_device(self, type):
        # device set to HPU explicitly
        hpu = torch.device("hpu", 0)
        hpu_type = type[0](test_values, device="hpu")

        assert hpu_type.device == hpu

    def test_wrong_device(self, type):
        # explicitly device set to "non HPU" should raise exception
        with pytest.raises(RuntimeError) as ex_info:
            hpu_type = type[0](test_values, device="cpu")

        regexp_str = "^legacy constructor expects device type: hpu but device type: cpu was passed$"
        assert ex_info.match(regexp_str)

    def test_correct_dtype(self, type):
        # dtype set to HPU explicitly
        torch_type = type[1](test_values)
        hpu_type = type[0](test_values, dtype=torch_type.dtype)

        assert hpu_type.dtype == torch_type.dtype

    def test_wrong_dtype(self, type):
        # explicitly dtype set to "wrong one" should raise exception
        torch_type = type[1](test_values)

        wrong_dtype = torch.float32
        if torch_type.dtype == wrong_dtype:
            wrong_dtype = torch.int16

        with pytest.raises(RuntimeError) as ex_info:
            hpu_type = type[0](test_values, dtype=wrong_dtype)

        regexp_str = f"^legacy constructor expects dtype: {torch_type.dtype} but dtype: {wrong_dtype} was passed$"
        assert ex_info.match(regexp_str)
