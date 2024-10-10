###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
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
