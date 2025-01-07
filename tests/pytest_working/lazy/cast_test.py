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
