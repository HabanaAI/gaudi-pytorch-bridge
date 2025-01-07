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

import habana_frameworks.torch
import torch
from test_utils import compare_tensors, cpu, hpu


def reference_lamb_norm(grads, max_grad_norm):
    global_grad_norm = torch.zeros(1, dtype=grads[0].dtype, device=grads[0].device)
    for grad in grads:
        global_grad_norm.add_(grad.pow(2).sum())

    global_grad_norm = global_grad_norm.sqrt()

    if global_grad_norm > max_grad_norm:
        clip_global_grad_norm = global_grad_norm / max_grad_norm
    else:
        clip_global_grad_norm = torch.tensor([1.0], dtype=grads[0].dtype)
    return clip_global_grad_norm


def create_grads(shapes):
    cpu_grads, hpu_grads = [], []
    for shape in shapes:
        cpu_grads.append(torch.randn(shape, device=cpu))
        hpu_grads.append(cpu_grads[-1].to(hpu))
    return cpu_grads, hpu_grads


def test_optimizer_lamb_fused_norm_sw_162999():
    cpu_grads, hpu_grads = create_grads([[1024, 2], [1024, 2]])

    result = torch.ops.hpu.optimizer_lamb_fused_norm(hpu_grads, 1.0)
    reference = reference_lamb_norm(cpu_grads, 1.0)

    unrelated_tensor = torch.tensor(1.0, device="hpu")

    compare_tensors(result, reference, atol=1e-08, rtol=1e-05)
