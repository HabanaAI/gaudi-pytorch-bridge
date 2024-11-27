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

import pytest
import torch


@pytest.fixture
def disable_acc_par_mode():
    variable_name_acc_par_mode = "PT_HPU_LAZY_ACC_PAR_MODE"
    original_value_acc_par_mode = os.environ.get(variable_name_acc_par_mode)

    # Set the environment variable to the desired value
    os.environ[variable_name_acc_par_mode] = "0"

    # Yield to provide the value for the test
    yield "1"

    # Teardown: Restore the original value after the test
    if original_value_acc_par_mode is not None:
        os.environ[variable_name_acc_par_mode] = original_value_acc_par_mode
    else:
        del os.environ[variable_name_acc_par_mode]


def test_hpu_lazy_resize():
    src_cpu = torch.tensor([10])
    src_resized_cpu = src_cpu.resize_((2,))
    src_resized_cpu[1] = 10

    src = torch.tensor([10])
    hpu = torch.device("hpu")
    src_hpu = src.detach().to(hpu)
    src_resized_hpu = src_hpu.resize_((2,))
    src_resized_hpu[1] = 10

    assert torch.allclose(src_resized_cpu, src_resized_hpu, atol=0, rtol=0), "Data mismatch"


def test_hpu_lazy_view_resize():
    src_cpu = torch.tensor([10])
    src_cpu_view = src_cpu.view(1)
    src_resized_cpu = src_cpu_view.resize_((2,))
    src_resized_cpu[1] = 10

    src = torch.tensor([10])
    hpu = torch.device("hpu")
    src_hpu = src.detach().to(hpu)
    src_hpu_view = src_hpu.view(1)
    src_resized_hpu = src_hpu_view.resize_((2,))
    src_resized_hpu[1] = 10

    assert torch.allclose(src_resized_cpu, src_resized_hpu, atol=0, rtol=0), "Data mismatch"


def test_hpu_resize_nonzero_out_raise_warning():
    def fn(input, out_resized):
        a = input.relu()
        torch.abs(out_resized, out=a)
        return a

    input_hpu = torch.randn(100, dtype=torch.float, device="hpu")
    output_resized_hpu = torch.empty(10, dtype=torch.float, device="hpu")

    with pytest.warns(Warning, match=r"An output with one or more elements was resized"):
        result_hpu = fn(input_hpu, output_resized_hpu)
        result_hpu.cpu()


def test_hpu_resize_empty_out():
    def fn(input, out_resized):
        a = input.relu()
        a.resize_(0)
        torch.abs(out_resized, out=a)
        return a

    input_cpu = torch.randn(100, dtype=torch.float)
    output_resized_cpu = torch.empty(10, dtype=torch.float)

    result_cpu = fn(input_cpu, output_resized_cpu)

    input_hpu = input_cpu.to("hpu")
    output_resized_hpu = output_resized_cpu.to("hpu")

    result_hpu = fn(input_hpu, output_resized_hpu)

    torch.testing.assert_close(result_hpu.cpu(), result_cpu)


def test_hpu_lazy_multiple_resizes():
    t_cpu = torch.ones(1, dtype=torch.float32)
    t_cpu.resize_(2)
    t_cpu.fill_(2.0)
    t_cpu.resize_(3)
    t_cpu.fill_(3.0)

    t_hpu = torch.ones(1, device="hpu", dtype=torch.float32)
    t_hpu.resize_(2)
    t_hpu.fill_(2.0)
    t_hpu.resize_(3)
    t_hpu.fill_(3.0)

    assert torch.allclose(t_cpu, t_hpu, atol=0, rtol=0), "Data mismatch"


def test_hpu_lazy_op_before_resize():
    t_cpu = torch.ones(3, dtype=torch.float32)
    x_cpu = t_cpu.mul(3.0)
    t_cpu.resize_(2)
    t_cpu.fill_(2.0)

    t_hpu = torch.ones(3, device="hpu", dtype=torch.float32)
    x_hpu = t_hpu.mul(3.0)
    t_hpu.resize_(2)
    t_hpu.fill_(2.0)

    assert torch.allclose(t_cpu, t_hpu, atol=0, rtol=0), "Data mismatch"
    assert torch.allclose(x_cpu, x_hpu, atol=0, rtol=0), "Data mismatch"


def test_hpu_lazy_non_acc_thread_resizes(disable_acc_par_mode):
    t_cpu = torch.ones(1, dtype=torch.float32)
    t_cpu.resize_(2)
    t_cpu.fill_(2.0)
    t_cpu.resize_(3)
    t_cpu.fill_(3.0)

    t_hpu = torch.ones(1, device="hpu", dtype=torch.float32)
    t_hpu.resize_(2)
    t_hpu.fill_(2.0)
    t_hpu.resize_(3)
    t_hpu.fill_(3.0)

    assert torch.allclose(t_cpu, t_hpu, atol=0, rtol=0), "Data mismatch"
