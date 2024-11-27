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

import torch
from habana_frameworks import torch as _


class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        flag = x == y
        return flag


def test_fill_propagated_tensor_metadata_to_node():

    model = MyModule().to("hpu")
    compiled_model = torch.compile(model, backend="hpu_backend", dynamic=True)
    # forced dynamic compilation forces occurence of SymBool in this mini example as internal output type
    retval = compiled_model(2, 3)
    assert retval == False
