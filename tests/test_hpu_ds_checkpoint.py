import os

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.utils.debug as htdebug
import pytest
import torch
from pytest_working.test_utils import env_var_in_scope

pytestmark = pytest.mark.skip(reason="Tests in this file are chaning env variables")


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 32)
        self.fc2 = torch.nn.Linear(32, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.mean(x, dim=1)
        return x

    def loss(self, x, y):
        y_pred = self(x)
        loss = torch.nn.functional.nll_loss(y_pred, y)
        return loss


def run_model(args):
    model = Net()
    model = model.to("hpu")
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    channel_size_list = torch.randint(4, 20, (20,))

    if args.resume_checkpoint:
        htdebug.load_ds_checkpoint("{0:s}/ds_ckt_epoch_{1:d}".format(args.save_dir, args.start_idx - 1))

    for idx, c in enumerate(range(args.start_idx, args.end_idx)):
        X = torch.randn((3, channel_size_list[c], 16))
        y = torch.randint(0, 2, (3,))
        X = X.to("hpu")
        y = y.to("hpu")

        optim.zero_grad()
        loss = model.loss(X, y)
        htcore.mark_step()

        loss.backward()
        htcore.mark_step()

        optim.step()
        htcore.mark_step()
        if args.save_checkpoint:
            htdebug.save_ds_checkpoint("{0:s}/ds_ckt_epoch_{1:d}".format(args.save_dir, idx + args.start_idx))


def add_op(input_shape):
    input1 = torch.randn(input_shape)
    input2 = torch.randn(input_shape)

    h_input1 = input1.to("hpu")
    h_input2 = input2.to("hpu")
    torch.add(h_input1, h_input2)
    htcore.mark_step()


def test_add_op(args):
    with env_var_in_scope(
        {
            "PT_COMPILATION_STATS_PATH": "/tmp/save_dir/json_run",
            "PT_RECIPE_TRACE_PATH": "save_dir/recipe_trace_run.csv",
        }
    ):
        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)

        save_dir = "/tmp/save_dir"
        resume_checkpoint = 0
        save_checkpoint = 0
        start_idx = 0
        end_idx = 20

        channel_size_list = [6, 8, 10, 4]

        if resume_checkpoint:
            htdebug.load_ds_checkpoint(save_dir)

        for c in range(start_idx, end_idx):
            add_op((4, channel_size_list[c], 3))
            if save_checkpoint:
                htdebug.save_ds_checkpoint(save_dir)
