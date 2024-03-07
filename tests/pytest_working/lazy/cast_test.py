# ******************************************************************************
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************
import itertools

import habana_frameworks.torch.utils.experimental as htexp
import pytest
import torch

dtype = [
    # torch.double, https://jira.habana-labs.com/browse/SW-115570
    torch.float,
    torch.bfloat16,
    # torch.long, https://jira.habana-labs.com/browse/SW-115570
    torch.int,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
    # torch.half,
    # torch.complex32,
    # torch.complex64,
    # torch.complex128,
    # torch.qint32,
    # torch.qint8,
]

if htexp._get_device_type() != htexp.synDeviceType.synDeviceGaudi:
    dtype.append(torch.half)


def get_name(param):
    return str(param).replace("torch.", "")


@pytest.mark.parametrize("dtype1, dtype2", itertools.product(dtype, dtype), ids=get_name)
def test_cast(dtype1, dtype2):
    device = "hpu"
    print(dtype1, dtype2)
    a = torch.ones(4, device=device, dtype=dtype1)
    b = a.to(dtype2)
    b.cpu()
