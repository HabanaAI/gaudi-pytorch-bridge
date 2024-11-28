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
import habana_frameworks.torch.internal._bridge_config_C as bc
import torch


def test_smoke_jit_forked_lowering():
    prev = bc.get_pt_hpu_use_jit_fork()
    bc.set_pt_hpu_use_jit_fork(True)
    t = torch.tensor([1.0], device="hpu")

    @torch.compile(backend="hpu_backend")
    def smoke(x):
        y = torch.tensor([2.0]).to(device="hpu")
        return y + x

    _ = smoke(t)
    bc.set_pt_hpu_use_jit_fork(prev)
