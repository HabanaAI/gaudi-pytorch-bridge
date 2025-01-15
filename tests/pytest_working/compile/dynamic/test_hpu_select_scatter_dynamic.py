###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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

# torch.compile Dynamic Shapes test code for select_scatter op
# Set environment variable PT_HPU_LAZY_MODE to 0

import torch
from test_utils import cpu, hpu

torch._dynamo.config.specialize_int = False


# shape_src should be of the same size as torch.select(input_shape, dim,index)
# This equates to the second dimension of input_shape for dim=0
def test_select_scatter():
    # (input_shape, shape_src, dim, index)
    input_shapes = [
        ((16, 16), (16), 0, 4),
        ((16, 16), (16), 1, 8),
        ((16, 16), (16), 0, 5),
        ((16, 16), (16), 1, 6),
    ]

    # Created a mini graph for testing
    # add op -> select_scatter op -> mul op
    def wrapper_fn(t, t_src, dim, indices):
        t1 = t.add(t)
        t2 = t1.select_scatter(t_src, dim, indices)
        t3 = t2.mul(5)
        return t3

    f_hpu = torch.compile(wrapper_fn, backend="hpu_backend", dynamic=None)

    for shape in input_shapes:
        input_tensor = torch.rand(shape[0], requires_grad=False, device=cpu)
        src_tensor = torch.rand(shape[1], requires_grad=False, device=cpu)

        y_cpu = wrapper_fn(input_tensor, src_tensor, shape[2], shape[3])
        y_hpu = f_hpu(input_tensor.to(hpu), src_tensor.to(hpu), shape[2], shape[3])

        assert torch.allclose(y_cpu, y_hpu.to(cpu), atol=0.001, rtol=0.001)


def test_select_scatter_st_meta():
    # (input_shape, shape_src, index)
    input_shapes = [
        ((16, 16, 1, 2, 3), (16, 1, 2, 3), 0),
        ((17, 17, 1, 2, 3), (17, 1, 2, 3), 1),
        ((18, 18, 1, 2, 3), (18, 1, 2, 3), 2),
        ((19, 19, 1, 2, 3), (19, 1, 2, 3), 0),
        ((20, 20, 1, 2, 3), (20, 1, 2, 3), 1),
    ]

    # Created a mini graph for testing
    # add op -> select_scatter op -> mul op
    def wrapper_fn(t, t_src, indices):
        t2 = t.select_scatter(t_src, 0, indices)
        return t2

    f_hpu = torch.compile(wrapper_fn, backend="hpu_backend")

    for shape in input_shapes:
        print("Input shape: ", shape)
        input_tensor = torch.rand(shape[0], requires_grad=False, device=cpu)
        src_tensor = torch.rand(shape[1], requires_grad=False, device=cpu)

        y_cpu = wrapper_fn(input_tensor, src_tensor, shape[2])
        y_hpu = f_hpu(input_tensor.to(hpu), src_tensor.to(hpu), shape[2])

        assert torch.allclose(y_cpu, y_hpu.to(cpu), atol=0.001, rtol=0.001)
