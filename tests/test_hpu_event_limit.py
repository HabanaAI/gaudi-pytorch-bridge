import torch
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch as ht
import time
from threading import Thread
from test_utils import reset_seed, compare_tensors
import numpy as np


def create_events(enable_timing):
    import datetime;
    in_shape = (10,2)
    tA_h = torch.zeros(in_shape).to('hpu')
    tB_h = torch.ones(in_shape).to('hpu')

    s = ht.hpu.Stream()
    events = []
    # for i in range(1024 * 1024 // 2):
    for i in range(1000000):
        startEv =ht.hpu.Event(enable_timing)
        endEv = ht.hpu.Event(enable_timing)
        assert endEv.query()== True , "Event query on unrecorded event returned False (expected True)"

        events.append((startEv, endEv))
        print(f"{datetime.datetime.now()} Added start end {enable_timing=} events (loop #{i})",flush=True)
    print(f"{datetime.datetime.now()} Added {len(events)*2} {enable_timing=} events", flush=True)

    for i, (startEv, endEv) in enumerate(events):
        print("interation i::", i)
        startEv.record()
        # time.sleep(0.5)
        for _ in range(100):
            tA_h = torch.add(tA_h,tB_h)
        endEv.record()
        endEv.synchronize()
        if enable_timing:
            print(f'{datetime.datetime.now()} {i=} {enable_timing=} Time Elapsed={startEv.elapsed_time(endEv)}')  # milliseconds
        print(f'After record {enable_timing=} :endEv info={repr(endEv)}')

def testMaxLimitForProfileEvents():
    create_events(True)

def testMaxLimitForEvents():
    create_events(False)

if __name__ == "__main__":
    testMaxLimitForProfileEvents()
    testMaxLimitForEvents()
