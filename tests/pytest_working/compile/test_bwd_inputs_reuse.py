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

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import pytest
import torch
from habana_frameworks.torch.dynamo.compile_backend import config as hpu_backend_config
from torch._dynamo import compiled_autograd

device = "hpu"
backend_compiler = "hpu_backend"


def compiler_fn(gm):
    return torch.compile(gm, backend=backend_compiler, fullgraph=True)


def hpu_partiton_breaker(x):
    x = x.to("cpu")
    x = torch.sigmoid(x)
    x = x.to(device)
    return x


class Module(torch.nn.Module):
    def __init__(self, ioc):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(ioc, ioc, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(ioc, ioc, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = hpu_partiton_breaker(x)
        x = self.conv2(x)
        return x.sum()


def zero_grad(module: torch.nn.Module):
    for name, param in module.named_parameters():
        if param.requires_grad and param.grad is not None:
            param.grad = None


def test_reuse_bwd_input_tensors():
    # force to sync mode to get accurate peak memory
    orig_sync_flag = os.environ.get("PT_HPU_SYNC_LAUNCH", "0")
    os.environ["PT_HPU_SYNC_LAUNCH"] = "1"

    bs = 8
    ioc = 32
    h, w = 64, 64

    model = Module(ioc).to(device)
    input = torch.randn([bs, ioc, h, w]).to(device)

    # eager
    hthpu.reset_peak_memory_stats()
    loss = model(input)
    loss.backward()
    eager_max_mem = hthpu.max_memory_allocated() // 1024 // 1024  # 21 MB
    zero_grad(model)
    print(f"Eager mode max mem allocated is {eager_max_mem} MB")

    # save original flag
    orig_flag = hpu_backend_config.use_boxed_input
    orig_eager_fallback = hpu_backend_config.use_eager_fallback
    hpu_backend_config.use_eager_fallback = True

    model_to_train = torch.compile(model, backend=backend_compiler)

    # compile without reuse
    hthpu.reset_peak_memory_stats()
    hpu_backend_config.use_boxed_input = False
    loss = model_to_train(input)
    loss.backward()
    compile_wo_reuse_max_mem = hthpu.max_memory_allocated() // 1024 // 1024  # 29 MB
    zero_grad(model_to_train)
    torch._dynamo.reset()  # clear the cached compiled function
    print(f"Compile mode w.o. reuse max mem allocated is {compile_wo_reuse_max_mem} MB")

    # compile with reuse
    hthpu.reset_peak_memory_stats()
    hpu_backend_config.use_boxed_input = True
    loss = model_to_train(input)
    loss.backward()
    compile_w_reuse_max_mem = hthpu.max_memory_allocated() // 1024 // 1024  # 25 MB
    zero_grad(model_to_train)
    torch._dynamo.reset()  # clear the cached compiled function
    print(f"Compile mode w. reuse max mem allocated is {compile_w_reuse_max_mem} MB")

    # recover original flag
    hpu_backend_config.use_boxed_input = orig_flag
    hpu_backend_config.use_eager_fallback = orig_eager_fallback
    os.environ["PT_HPU_SYNC_LAUNCH"] = orig_sync_flag

    mem_diff = compile_wo_reuse_max_mem - compile_w_reuse_max_mem

    # 8*32*64*64*4/1024/1024 == 4
    assert mem_diff == 4, "We should be able to reuse a 4 MB bwd input tensor with boxed input"


def test_reuse_bwd_inputs_with_compiled_autograd():
    # force to sync mode to get accurate peak memory
    orig_sync_flag = os.environ.get("PT_HPU_SYNC_LAUNCH", "0")
    os.environ["PT_HPU_SYNC_LAUNCH"] = "1"

    bs = 8
    ioc = 32
    h, w = 64, 64

    model = Module(ioc).to(device)
    input = torch.randn([bs, ioc, h, w]).to(device)

    # save original flag
    orig_flag = hpu_backend_config.use_boxed_input
    orig_eager_fallback = hpu_backend_config.use_eager_fallback
    hpu_backend_config.use_eager_fallback = True

    # compiled autograd without reuse
    hthpu.reset_peak_memory_stats()
    hpu_backend_config.use_boxed_input = False
    loss = model(input)
    with compiled_autograd.enable(compiler_fn):
        loss.backward()
    compile_wo_reuse_max_mem = hthpu.max_memory_allocated() // 1024 // 1024  # 25 MB
    zero_grad(model)
    torch._dynamo.reset()  # clear the cached compiled function
    print(f"Compiled Autograd mode w.o. reuse max mem allocated is {compile_wo_reuse_max_mem} MB")

    # compiled autograd with reuse
    hthpu.reset_peak_memory_stats()
    hpu_backend_config.use_boxed_input = True
    loss = model(input)
    with compiled_autograd.enable(compiler_fn):
        loss.backward()
    compile_w_reuse_max_mem = hthpu.max_memory_allocated() // 1024 // 1024  # 21 MB
    zero_grad(model)
    torch._dynamo.reset()  # clear the cached compiled function
    print(f"Compiled Autograd mode w. reuse max mem allocated is {compile_w_reuse_max_mem} MB")

    # recover original flag
    hpu_backend_config.use_boxed_input = orig_flag
    hpu_backend_config.use_eager_fallback = orig_eager_fallback
    os.environ["PT_HPU_SYNC_LAUNCH"] = orig_sync_flag

    mem_diff = compile_wo_reuse_max_mem - compile_w_reuse_max_mem

    # 8*32*64*64*4/1024/1024 == 4
    assert mem_diff == 4, "We should be able to reuse a 4 MB bwd input tensor with boxed input"


if __name__ == "__main__":
    test_reuse_bwd_input_tensors()
    test_reuse_bwd_inputs_with_compiled_autograd()
