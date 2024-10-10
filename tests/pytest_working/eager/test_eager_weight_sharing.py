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
from test_utils import generic_setup_teardown_env

pytestmark = pytest.mark.skip(reason="Tests in this file are chaning env variables")


@pytest.fixture(autouse=True, scope="module")
def setup_teardown_env():
    # disable lazy weight sharing
    yield from generic_setup_teardown_env({"PT_HPU_WEIGHT_SHARING": 0})


def test_weight_share_across_model_to():
    class MyModule(torch.nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            self.a = torch.nn.Parameter(torch.ones([1]))
            self.b = torch.nn.Parameter(torch.ones([1]))

        def forward(self, input):
            c = self.a * input + self.b * input
            return c

    mod = MyModule()
    mod.a = mod.b  # Share parameters a and b from the model
    mod.to("hpu")
    assert mod.a.data_ptr() == mod.b.data_ptr()


def test_saved_param_update_across_model_to():
    class MyModule(torch.nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            self.a = torch.nn.Parameter(torch.randn([10, 10]))

        def forward(self, input):
            return self.a * input

    model = MyModule()
    device = "hpu"

    param_list = [model.a]
    model.to(device)

    for p in model.parameters():
        device_after_model_to = p.device
        data_ptr_after_model_to = p.data_ptr()

    assert param_list[0].device == device_after_model_to
    assert param_list[0].data_ptr() == data_ptr_after_model_to


def test_assign_to_module_param_data():
    class MyModule(torch.nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            self.a = torch.nn.Parameter(torch.ones([1]))
            self.b = torch.nn.Parameter(torch.ones([1]))

        def forward(self, input):
            c = self.a / input + self.b * input
            return c

    device = "hpu"
    mod = MyModule()
    mod.to(device)

    cpu_device = torch.randn([1]).device
    hpu_device = torch.randn([1]).to(device).device

    assert mod.a.device == hpu_device
    assert mod.b.device == hpu_device

    mod_a_data_ptr_after_model_to = mod.a.data_ptr()
    mod_b_data_ptr_after_model_to = mod.b.data_ptr()

    mod.a.data = mod.a.to("cpu")
    mod.b.data = mod.b.to("cpu")

    assert mod.a.device == cpu_device
    assert mod.b.device == cpu_device

    mod_a_data_ptr_after_move_to_cpu = mod.a.data_ptr()
    mod_b_data_ptr_after_move_to_cpu = mod.b.data_ptr()

    assert mod_a_data_ptr_after_move_to_cpu != mod_a_data_ptr_after_model_to
    assert mod_b_data_ptr_after_move_to_cpu != mod_b_data_ptr_after_model_to

    tmp_list = [mod.a, mod.b]
    new_list = []
    for t in tmp_list:
        t_device = t.device
        assert t_device == cpu_device
        t_data_ptr = t.data_ptr()
        if len(new_list) == 0:
            assert t_data_ptr == mod_a_data_ptr_after_move_to_cpu
        else:
            assert t_data_ptr == mod_b_data_ptr_after_move_to_cpu
        new_list.append(t.contiguous().view(-1))

    flat = torch.cat(new_list)
    flat_hpu = flat.to("hpu")
    assert flat_hpu.device == hpu_device
