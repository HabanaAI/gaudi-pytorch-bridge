import habana_frameworks.torch as ht
import pytest
import torch

pytestmark = pytest.mark.skip(reason="Too long/hang")


def create_events(enable_timing):
    in_shape = (10, 2)
    tA_h = torch.zeros(in_shape).to("hpu")
    tB_h = torch.ones(in_shape).to("hpu")

    ht.hpu.Stream()
    events = []
    for _ in range(1000000):
        startEv = ht.hpu.Event(enable_timing)
        endEv = ht.hpu.Event(enable_timing)
        assert endEv.query() is True, "Event query on unrecorded event returned False (expected True)"

        events.append((startEv, endEv))

    for _, (startEv, endEv) in enumerate(events):
        startEv.record()
        for _ in range(100):
            tA_h = torch.add(tA_h, tB_h)
        endEv.record()
        endEv.synchronize()


def testMaxLimitForProfileEvents():
    create_events(True)


def testMaxLimitForEvents():
    create_events(False)


if __name__ == "__main__":
    testMaxLimitForProfileEvents()
    testMaxLimitForEvents()
