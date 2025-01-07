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

import os
import shutil

import habana_frameworks.torch as htorch
import numpy as np
import pytest
import torch


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 32)
        self.fc2 = torch.nn.Linear(32, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.mean(x, dim=1)
        return x


def test_const_serialization_cache():
    torch.manual_seed(123456)
    htorch.core.hpu_set_inference_env()

    serial_path = "/tmp/const_section_test/"
    htorch.hpu.enable_const_section_serialization(serial_path, True, True)

    model = Net()
    model = model.to("hpu")

    htorch.core.hpu_inference_initialize(model)

    X = torch.randn((3, 3, 16))

    with torch.no_grad():
        out = model(X)
        htorch.core.mark_step()
        out_serialize = out.to("cpu")

    # check for 4 const weights serialized files
    num_files = len(os.listdir(os.path.join(serial_path, "0")))
    assert 4 == num_files

    # run from serialization
    with torch.no_grad():
        out = model(X)
        htorch.core.mark_step()
        out_deserialize = out.to("cpu")

    np.array_equal(out_serialize.detach().numpy(), out_deserialize.detach().numpy(), equal_nan=True)

    # clear config
    shutil.rmtree(serial_path)
    htorch.core.hpu_teardown_inference_env()
    htorch.hpu.disable_inference_mode()
    htorch.hpu.disable_const_section_serialization()
