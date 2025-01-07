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
