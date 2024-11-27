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
from pathlib import Path

import habana_frameworks.torch.core
import torch

is_lazy = os.environ.get("PT_HPU_LAZY_MODE", "1") == "1"


def load_library(is_lazy, legacy):
    if is_lazy:
        mode = "lazy_legacy" if legacy else "lazy"
    else:
        mode = "eager"

    base_dir = Path(__file__).parent.resolve()
    build_dir = Path(os.path.join(base_dir, "build"))
    print(base_dir)
    print(build_dir)
    custom_op_lib_path = str(
        next(
            Path(next(build_dir.glob("lib.linux-x86_64-*"))).glob(f"hpu_custom_op_{mode}.cpython-*-x86_64-linux-gnu.so")
        )
    )
    torch.ops.load_library(custom_op_lib_path)


def custom_topk(compile, legacy):
    load_library(is_lazy, legacy)

    fn = torch.ops.custom_op.custom_topk
    if compile:
        fn = torch.compile(fn, backend="hpu_backend")
    a_cpu = torch.rand((6, 6))
    a_hpu = a_cpu.to("hpu")
    a_topk_hpu, a_topk_indices_hpu = fn(a_hpu, 3, 1, False)
    a_topk_cpu, a_topk_indices_cpu = a_cpu.topk(3, 1)
    assert torch.equal(a_topk_hpu.detach().cpu(), a_topk_cpu.detach().cpu())


def custom_add(compile, legacy):
    load_library(is_lazy, legacy)

    fn = torch.ops.custom_op.custom_add
    if compile:
        fn = torch.compile(fn, backend="hpu_backend")
    a = torch.rand((1))
    b = torch.rand((6, 4))
    a_hpu = a.to("hpu")
    b_hpu = b.to("hpu")
    res_hpu = fn(a_hpu, b_hpu)
    res_cpu = a + b
    assert torch.equal(res_hpu.detach().cpu(), res_cpu.detach().cpu())
