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

import habana_frameworks.torch.internal.bridge_config as bc
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
    assert cpu_seed_0 == hpu_seed_0

    torch.hpu.seed()
    cpu_seed_1 = torch.initial_seed()
    hpu_seed_1 = torch.hpu.initial_seed()
    assert hpu_seed_1 != hpu_seed_0
    assert cpu_seed_1 == cpu_seed_0

    torch.seed()
    cpu_seed_2 = torch.initial_seed()
    hpu_seed_2 = torch.hpu.initial_seed()
    assert hpu_seed_2 != hpu_seed_1
    assert cpu_seed_2 != cpu_seed_1


def test_fork_rng():
    rng_state_0 = torch.hpu.get_rng_state()

    # fork_rng should restore rng_state after its scope when device_type is "hpu"
    with torch.random.fork_rng(device_type="hpu"):
        torch.manual_seed(12345678)
        rng_state_temp_0 = torch.hpu.get_rng_state()

    assert not torch.equal(rng_state_0, rng_state_temp_0)

    rng_state_1 = torch.hpu.get_rng_state()
    assert torch.equal(rng_state_1, rng_state_0)

    # fork_rng should restore rng_state after its scope with default device_type,
    # since we overwritten default implementation, so default device_type is "hpu"
    with torch.random.fork_rng():
        torch.manual_seed(654321)
        rng_state_temp_1 = torch.hpu.get_rng_state()

    assert not torch.equal(rng_state_temp_1, rng_state_temp_0)
    assert not torch.equal(rng_state_1, rng_state_temp_1)

    rng_state_2 = torch.hpu.get_rng_state()
    assert torch.equal(rng_state_2, rng_state_1)

    # fork_rng shouldn't restore rng_state after its scope with device_type != "hpu"
    # unless GPU Migration is used, in which case fork_rng should restore rng_state
    with torch.random.fork_rng(device_type="cuda"):
        torch.manual_seed(424242)
        rng_state_temp_2 = torch.hpu.get_rng_state()

    assert not torch.equal(rng_state_temp_2, rng_state_temp_1)
    assert not torch.equal(rng_state_2, rng_state_temp_2)

    rng_state_3 = torch.hpu.get_rng_state()
    is_restored = torch.equal(rng_state_3, rng_state_temp_2)
    if bc.get_pt_hpu_gpu_migration():
        assert not is_restored
    else:
        assert is_restored
