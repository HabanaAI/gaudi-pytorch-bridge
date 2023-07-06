import habana_frameworks.torch.hpu.random as rand_hpu
import pytest
import torch


def test_get_and_set_rng_state():
    state = rand_hpu.get_rng_state()
    before0 = torch.FloatTensor(100).to("hpu").normal_()
    before1 = torch.FloatTensor(100).to("hpu").normal_()
    rand_hpu.set_rng_state(state)
    after0 = torch.FloatTensor(100).to("hpu").normal_()
    after1 = torch.FloatTensor(100).to("hpu").normal_()

    assert (after0 == before0).sum().item() == 100
    assert (after1 == before1).sum().item() == 100


def test_manual_initial_seed():
    rng_state = torch.get_rng_state()
    torch.manual_seed(2)
    rand_hpu.manual_seed(2)
    x_cpu = torch.randn(100)
    x_hpu = torch.randn(100).to("hpu")

    assert torch.initial_seed() == 2
    assert rand_hpu.initial_seed() == 2

    torch.manual_seed(2)
    rand_hpu.manual_seed(2)
    y_cpu = torch.randn(100)
    y_hpu = torch.randn(100).to("hpu")

    assert torch.allclose(x_cpu, y_cpu)
    assert torch.allclose(x_hpu, y_hpu)

    # check all the boundary conditions.
    # (Taken from test_manual_seed in pytorch/test/test_torch.py)
    max_int64 = 0x7FFF_FFFF_FFFF_FFFF
    min_int64 = -max_int64 - 1
    max_uint64 = 0xFFFF_FFFF_FFFF_FFFF

    test_cases = [
        # (seed, expected_initial_seed)
        # Positive seeds should be unchanged
        (max_int64, max_int64),
        (max_int64 + 1, max_int64 + 1),
        (max_uint64, max_uint64),
        (0, 0),
        # Negative seeds wrap around starting from the largest seed value
        (-1, max_uint64),
        (min_int64, max_int64 + 1),
    ]

    for seed, expected_initial_seed in test_cases:
        torch.manual_seed(seed)
        rand_hpu.manual_seed(seed)
        actual_cpu = torch.initial_seed()
        actual_hpu = rand_hpu.initial_seed()

        assert expected_initial_seed == actual_cpu
        assert expected_initial_seed == actual_hpu

    torch.set_rng_state(rng_state)
    rand_hpu.set_rng_state(rng_state)


@pytest.mark.xfail(reason="seeds are different")
def test_initial_seed():
    seed_hpu = rand_hpu.initial_seed()
    seed_cpu = torch.random.initial_seed()
    assert seed_hpu == seed_cpu

    rand_hpu.seed()
    seed_hpu = rand_hpu.initial_seed()
    seed_cpu = torch.random.initial_seed()
    assert seed_hpu == seed_cpu
