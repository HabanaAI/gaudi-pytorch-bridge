###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
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
