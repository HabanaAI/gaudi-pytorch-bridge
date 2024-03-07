import random

import habana_frameworks.torch.core as htcore
import pytest
import torch

# Tests for index add with duplicate indices for the following
# scenarios and dims 0 or 1
# scenario 0 : # index size = self size at specified dim 0 or 1
# scenario 1 : # index size > self size at specified dim 0 or 1


def index_add(device, seed, st, scenario):
    dtype = torch.int32
    d = 0
    sort = st
    sf = (10, 10)  # self shape
    # k = index shape
    # u = range of indices
    if d == 0:
        if scenario == 0:  # index size <= self size
            k = sf[0]
            u = int(sf[0] / 2)
        else:
            k = sf[0] * 2  # index size > self size
            u = sf[0]
        ss = (k, sf[1])  # source shape
    else:
        if scenario == 0:  # index size <= self size
            k = sf[1]
            u = int(sf[1] / 2)
        else:
            k = sf[1] * 2  # index size > self size
            u = sf[1]
        ss = (sf[0], k)  # source shape

    self = torch.arange(1, sf[0] * sf[1] + 1).reshape(sf).to(dtype).to(device)
    # print("self =", self)
    if sort:
        a, b = torch.sort(torch.randint(0, u, (k,)))
        index = a.to(device)
    else:
        index = torch.randint(0, u, (k,)).to(device)

    source = torch.ones(ss).to(dtype).to(device)
    r = self.index_add_(d, index, source, alpha=1)

    if device == "hpu":
        htcore.mark_step()
    return r.to("cpu")


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(0, marks=[pytest.mark.xfail(reason="results mismatch")]),
        pytest.param(1, marks=[pytest.mark.xfail(reason="results mismatch")]),
    ],
)
def test_index_add_duplicate_indices(scenario):
    N = 20
    mms = 0
    mmu = 0
    for _ in range(N):
        seed = random.randint(10000, 99999999)
        st = True
        c = index_add("cpu", seed, st, scenario)
        h = index_add("hpu", seed, st, scenario)
        ms = torch.equal(c, h)
        assert torch.equal(c, h), "sorted case doesn't match"
        st = False
        c = index_add("cpu", seed, st, scenario)
        h = index_add("hpu", seed, st, scenario)
        mu = torch.equal(c, h)
        mms = mms + ms
        mmu = mmu + mu

        assert torch.equal(c, h), "unsorted case doesn't match"
