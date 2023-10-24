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

import torch


def test_hpu_rng_state():
    cpu_rng_state_0 = torch.get_rng_state()
    hpu_rng_state_0 = torch.hpu.get_rng_state()
    assert torch.equal(cpu_rng_state_0, hpu_rng_state_0)

    torch.manual_seed(12345678)
    cpu_rng_state_1 = torch.get_rng_state()
    hpu_rng_state_1 = torch.hpu.get_rng_state()
    assert torch.equal(cpu_rng_state_1, hpu_rng_state_1)
    assert not torch.equal(hpu_rng_state_1, hpu_rng_state_0)

    torch.hpu.manual_seed(3456)
    cpu_rng_state_2 = torch.get_rng_state()
    hpu_rng_state_2 = torch.hpu.get_rng_state()
    assert not torch.equal(hpu_rng_state_2, hpu_rng_state_1)
    assert torch.equal(cpu_rng_state_2, cpu_rng_state_1)

    torch.hpu.set_rng_state(hpu_rng_state_1)
    hpu_rng_state_3 = torch.hpu.get_rng_state()
    assert torch.equal(hpu_rng_state_3, hpu_rng_state_1)


def test_hpu_seed():
    torch.seed()
    cpu_seed_0 = torch.initial_seed()
    hpu_seed_0 = torch.hpu.initial_seed()
    assert cpu_seed_0 != hpu_seed_0

    torch.hpu.seed()
    cpu_seed_1 = torch.initial_seed()
    hpu_seed_1 = torch.hpu.initial_seed()
    assert hpu_seed_1 != hpu_seed_0
    assert cpu_seed_1 == cpu_seed_0

    torch.seed()
    cpu_seed_2 = torch.initial_seed()
    hpu_seed_2 = torch.hpu.initial_seed()
    assert hpu_seed_2 == hpu_seed_1
    assert cpu_seed_2 != cpu_seed_1
