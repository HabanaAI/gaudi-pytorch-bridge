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
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer


def test_pass_fuse_view_chains():
    # It doesn't matter all that much which operations are performed
    # as long as there are view chains inside a graph
    # that need to be eagerized

    def view_chain(inp):
        a = inp[1:]
        b = a.transpose(0, 1)
        d = torch.as_strided(b, (2, 2), (1, 2), 2)
        return d

    def fn(inp):
        a = inp * 2
        b = inp * a
        chain_1 = view_chain(b)
        chain_2 = view_chain(b)
        chain_1_cpu = chain_1.to("cpu")
        e = chain_1 + 5
        return (e, chain_1_cpu, chain_2)

    with FxGraphAnalyzer() as fga:
        inp_hpu = torch.randn(4, 3, device="hpu")
        inp_cpu = inp_hpu.to("cpu")
        fnc_hpu = torch.compile(fn, dynamic=False, backend="hpu_backend", options={"use_eager_fallback": True})
        fnc_cpu = torch.compile(fn, dynamic=False, backend="inductor")
        results_hpu = fnc_hpu(inp_hpu)
        results_cpu = fnc_cpu(inp_cpu)
    ops_summary = fga.get_ops_summary()
    assert ops_summary[0]["torch.as_strided"].eager_count == 2
    for r_hpu, r_cpu in zip(results_hpu, results_cpu):
        torch.allclose(r_hpu.to("cpu"), r_cpu)
