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

import torch
from habana_frameworks.torch.dynamo.compile_backend.partition_fn import remove_unnecessary_clone
from torch.fx.experimental.proxy_tensor import make_fx


class ModuleWithUnnecessaryClone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        y = torch.relu(x)
        y_cloned = y.clone()
        z = y_cloned + y
        return z


class ModuleWithUnnecessaryCloneAndViews(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        y = torch.relu(x)
        y_cloned = y.clone()
        y_trans = y_cloned.transpose(0, 1)
        z = torch.matmul(y, y_trans)
        return z


class ModuleWithNecessaryClone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        y = torch.relu(x)
        y_cloned = y.clone()
        # y_cloned is modified but y not
        y_modified = y_cloned.add_(3.0)
        z = y + y_modified
        return z


class ModuleWithNecessaryCloneAndViews(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        y = torch.relu(x)
        y_cloned = y.clone()
        y_expanded = y_cloned.expand_as(x)
        # y_cloned is modified but y not
        y_modified = y_expanded.add_(3.0)
        z = torch.mul(y, y_modified)
        return z


def run_model_and_compare_results(model, no_clone=True):
    x = torch.randn(4, 4)
    cpu_ref = model(x)

    gm = make_fx(model)(x)
    optimized_gm = remove_unnecessary_clone(gm)

    if no_clone:
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.clone.default:
                return False

    hpu_output = optimized_gm(x.to("hpu"))
    return torch.allclose(cpu_ref, hpu_output.to(device=torch.device("cpu")), rtol=1e-6, atol=1e-6)


def test_remove_unnecessary_clone():
    model = ModuleWithUnnecessaryClone()
    assert run_model_and_compare_results(model, True)


def test_remove_unnecessary_clone_with_views():
    model = ModuleWithUnnecessaryCloneAndViews()
    assert run_model_and_compare_results(model, True)


def test_not_remove_necessary_clone():
    model = ModuleWithNecessaryClone()
    # if the necessary clone is removed, the correctness check will fail
    assert run_model_and_compare_results(model, False)


def test_not_remove_necessary_clone_with_views():
    model = ModuleWithNecessaryCloneAndViews()
    # if the necessary clone is removed, the correctness check will fail
    assert run_model_and_compare_results(model, False)
