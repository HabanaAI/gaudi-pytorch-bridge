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
from inspect import currentframe, getframeinfo

import habana_frameworks.torch.core as htcore
import pytest
import torch
import torch.nn as nn
from test_utils import compare_tensors

data_list = [
    (torch.randn(2, 4), torch.randn(4, 4), torch.randn(3, 4)),
    (torch.randn(2, 4, 4), torch.randn(4, 4, 4), torch.randn(3, 4, 4)),
]


@torch.jit.script
def test_cat(x1, x2, x3):
    z = torch.cat((x1, x2, x3), 0)
    y = torch.relu(z)
    return y


@pytest.mark.parametrize("in_tensors", data_list)
def test_jit_cat_dbg(in_tensors):
    fi = getframeinfo(currentframe())
    src = fi.filename
    base = os.path.splitext(src)[0]
    trace_file_name = base + "_trace.pt"

    hpu = torch.device("hpu")
    cpu = torch.device("cpu")

    with torch.jit.optimized_execution(True):
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)

        model_trace = torch.jit.trace(test_cat, [in_tensors[0], in_tensors[1], in_tensors[2]])

        torch.jit.save(model_trace, trace_file_name)
        cpu_result = test_cat(in_tensors[0], in_tensors[1], in_tensors[2])

        htcore.enable()
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        hpu_t1 = in_tensors[0].to(hpu)
        hpu_t2 = in_tensors[1].to(hpu)
        hpu_t3 = in_tensors[2].to(hpu)
        model_trace_hpu = torch.jit.load(trace_file_name, map_location=torch.device("hpu"))
        out = model_trace_hpu(hpu_t1, hpu_t2, hpu_t3)
        hpu_result = out.to(cpu)
        compare_tensors(hpu_result, cpu_result, atol=0.001, rtol=1.0e-3)


if __name__ == "__main__":
    test_jit_cat_dbg(data_list[0])
