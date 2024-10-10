###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
import random

import habana_frameworks.torch.core as htcore
import numpy as np
import pytest
import torch
import torch.nn as nn
from habana_frameworks.torch.hpex.optimizers import FusedLars, FusedResourceApplyMomentum
from lars import Lars, ResourceApplyMomentum


def cosine_sim(a, b):
    a = a.detach().numpy().flatten().astype(np.float64)
    b = b.detach().numpy().flatten().astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    angle = np.arccos(min(np.dot(a, b) / na / nb, 1.0)) / np.pi * 180
    angle = np.around(angle, 3)
    return angle


# Test configuration parameters
S = 2  # Num iterations
N = 1  # batch size
# As of now C must be equal to O for linear to work; nothing to do with optimizer
C = 8  # channels = num input features
O = 8  # num output features # noqa
NUM_LAYERS = 5

cfg_lr = 0.02
cfg_momentum = 0.9
cfg_weight_decay = 0.001

rtol = 0.001
atol = 0.001

print_params_after_opt = False  # Print params after S iterations


devr = "cpu"  # Device for reference (full python) optimizer. can be cpu or hpu
devt = "hpu"  # device for Fused optimizer : hpu

# skip mask modes
ALL_ZEROS = 1
ALL_ONES = 2
RANDOM = 3
EVERY_X = 4


def set_skip_mask(model, mode, every=2):
    skip_mask = []
    k = 0
    for _ in model.parameters():
        if mode == ALL_ZEROS:
            skip_mask.append(0)
        elif mode == ALL_ONES:
            skip_mask.append(1)
        elif mode == EVERY_X:
            if (k % every) == 0:
                skip_mask.append(1)
            else:
                skip_mask.append(0)
        elif mode == RANDOM:
            skip_mask.append(random.randint(0, 1))
        k = k + 1
    return skip_mask


def run_model(dev, m, x, optim):

    model_output = m(x)
    loss = torch.sum(model_output)
    loss.backward()
    if dev == "hpu":
        htcore.mark_step()
    optim.step()


@pytest.mark.skip(reason="Graph compile failed")
def test_lars():
    model = nn.Sequential(*[nn.Linear(C, O, bias=True) for _ in range(NUM_LAYERS)])

    model_ref = copy.deepcopy(model).to(devr)
    model_test = copy.deepcopy(model).to(devt)

    momentum_optimizer_ref = ResourceApplyMomentum
    optimizer_ram_ref = momentum_optimizer_ref(
        model_ref.parameters(),
        lr=cfg_lr,
        momentum=cfg_momentum,
        weight_decay=cfg_weight_decay,
    )

    momentum_optimizer_test = FusedResourceApplyMomentum
    optimizer_ram_test = momentum_optimizer_test(
        model_test.parameters(),
        lr=cfg_lr,
        momentum=cfg_momentum,
        weight_decay=cfg_weight_decay,
    )

    skip_mask = set_skip_mask(model_ref, RANDOM)
    print("skip mask used = ", skip_mask)

    opt_ref = Lars(optimizer_ram_ref, skip_mask, eps=1e-8)
    opt_test = FusedLars(optimizer_ram_test, skip_mask, eps=1e-8)

    x = torch.randn(N, C)
    xr = x.to(devr)
    xt = x.to(devt)

    for _ in range(S):
        run_model(devr, model_ref, xr, opt_ref)
        run_model(devt, model_test, xt, opt_test)

    for pr, pt in zip(model_ref.parameters(), model_test.parameters()):
        prc = pr.to("cpu")
        ptc = pt.to("cpu")
        print(" Cosine similarity angle= ", cosine_sim(prc, ptc))
        np.testing.assert_allclose(prc.detach().numpy(), ptc.detach().numpy(), rtol, atol)
