###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import copy

import torch
import torch.nn as nn
from habana_frameworks.torch.dynamo.compile_backend import config as hpu_backend_config
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer
from test_utils import env_var_in_scope
from torch._dynamo import compiled_autograd
from torch.optim import Adam


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.param = nn.Parameter(torch.rand(input_dim, 8))
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim, 2 * input_dim),
                nn.BatchNorm1d(2 * input_dim),
                nn.ReLU(),
                nn.Linear(2 * input_dim, 10),
                nn.Softmax(),
            ]
        )

    def forward(self, x):
        x = torch.ops.aten.add.Tensor(x, self.param.t())
        for layer in self.layers:
            x = layer(x)
        return x


def assert_ops(ops_summary_1, ops_summary_2):
    assert len(ops_summary_1) == len(ops_summary_2)
    for i in range(len(ops_summary_1)):
        for op in ops_summary_1[i]:
            assert op in ops_summary_2[i]
            assert ops_summary_1[i][op].graph_count == ops_summary_2[i][op].graph_count
            assert ops_summary_1[i][op].eager_count == ops_summary_2[i][op].eager_count


def compiler_fn(gm):
    return torch.compile(gm, backend="hpu_backend", fullgraph=True)


def test_propose_partitions():
    torch.manual_seed(123)

    with compiled_autograd.enable(compiler_fn):
        input_dim = 100
        input = torch.rand((8, input_dim), dtype=torch.float, device="hpu")
        input_c = input.clone().detach()
        model = Net(input_dim)
        model_c = copy.deepcopy(model)

        hpu_backend_config.use_cpp_partitioner = True
        with FxGraphAnalyzer(reset_dynamo=True) as fga:
            model = torch.compile(model, backend="hpu_backend", options={"keep_input_mutations": True}).to(
                torch.device("hpu")
            )
            optim = Adam(model.parameters())
            output_1 = model(input)
            output_1.sum().backward()
            optim.step()
        ops_summary_1 = fga.get_ops_summary()

        hpu_backend_config.use_cpp_partitioner = False
        with FxGraphAnalyzer(reset_dynamo=True) as fga:
            model_c = torch.compile(model_c, backend="hpu_backend", options={"keep_input_mutations": True}).to(
                torch.device("hpu")
            )
            optim = Adam(model_c.parameters())
            output_2 = model_c(input_c)
            output_2.sum().backward()
            optim.step()
        ops_summary_2 = fga.get_ops_summary()

    assert_ops(ops_summary_1, ops_summary_2)
    assert torch.all(torch.isclose(output_2, output_1)).item()
