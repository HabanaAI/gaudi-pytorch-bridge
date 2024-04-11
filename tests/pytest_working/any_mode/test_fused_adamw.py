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
from copy import deepcopy

import pytest
import torch
from habana_frameworks.torch.hpex.optimizers import FusedAdamW
from test_utils import is_pytest_mode_compile
from torch.optim import AdamW


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(32, 1)

    def forward(self, x):
        return self.fc(x)


def train_model(model, optimizer, loss_fn, x, y):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

    return y_pred


@pytest.mark.skipif(is_pytest_mode_compile(), reason="Test is not adjusted to compile mode")
def test_fused_adamw_checkpoint_reading():
    adamw_model = Net().to("hpu")
    adamw_optim = AdamW(adamw_model.parameters(), lr=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    x = torch.randn((4, 32)).to("hpu")
    y = torch.tensor([0, 1, 1, 0]).to("hpu")

    train_model(adamw_model, adamw_optim, loss_fn, x, y)
    train_model(adamw_model, adamw_optim, loss_fn, x, y)

    model_state_dict = deepcopy(adamw_model.state_dict())
    optimizer_state_dict = deepcopy(adamw_optim.state_dict())

    adamw_y = train_model(adamw_model, adamw_optim, loss_fn, x, y)

    model_fused_adamw = Net().to("hpu")
    model_fused_adamw.load_state_dict(model_state_dict)
    fused_adamw_optim = FusedAdamW(model_fused_adamw.parameters())
    fused_adamw_optim.load_state_dict(optimizer_state_dict)

    fused_adamw_y = train_model(model_fused_adamw, fused_adamw_optim, loss_fn, x, y)

    torch.testing.assert_close(adamw_y.cpu(), fused_adamw_y.cpu())
