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

import os

import pytest
import torch
import torchvision
from test_utils import _is_simulator

os.environ["PT_HPU_LAZY_MODE"] = "1"
device = "hpu"


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize(
    "activities", [(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU), None]
)
def test_activity_profiler(device, activities):
    if _is_simulator():
        return

    inputs = torch.ones(2, 4).to(device)
    targets = torch.ones(2, 4).to(device)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("."),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        loss_cls = torchvision.ops.sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="sum")
        loss_cls.cpu()
    trace_file_gen = False
    for fname in os.listdir("."):
        if fname.endswith(".pt.trace.json"):
            os.remove(fname)
            trace_file_gen = True
            break
    assert trace_file_gen


if __name__ == "__main__":
    test_activity_profiler()
