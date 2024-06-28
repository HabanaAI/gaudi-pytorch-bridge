import time
from threading import Thread

import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import numpy as np
import torch


def doWork():
    in_shape = (4, 2)
    tA_h = torch.zeros(in_shape).to("hpu")
    tB_h = torch.ones(in_shape).to("hpu")

    tC_h = tA_h + tB_h
    print(f"output={tC_h.cpu()}")


def testBasic():
    print("Starting STREAMS BASIC TEST")
    s0 = ht.hpu.Stream()
    s1 = ht.hpu.Stream()

    # breakpoint()
    # print(type(s0))

    print("QUERY s0 - START")
    print(s0.query())
    print("QUERY - FINISHED")
    print("QUERY s1 - START")
    print(s1.query())
    print("QUERY - FINISHED")

    print(s0.synchronize())
    print("SYNC FINISHED")


def testAddOnStreams():
    print("Starting Add in Stream Context TEST")
    print("Create s0")
    s0 = ht.hpu.Stream()
    print("Created s0")
    ht.hpu.Stream()
    in_shape = (10, 2)
    tA_h = torch.zeros(in_shape).to("hpu")
    tB_h = torch.ones(in_shape).to("hpu")

    for _ in range(500):
        torch.add(tA_h, tB_h)

    print("StreamSync-Default Stream - Start")
    ht.hpu.default_stream().synchronize()
    print("StreamSync-Default Stream - End")

    with ht.hpu.stream(s0):
        for _ in range(500):
            torch.add(tA_h, tB_h)

    # with ht.hpu.stream(s1):
    #     tOut3 = torch.add(tB_h,tB_h)

    print("StreamSync-s0 - Start")
    s0.synchronize()
    print("StreamSync-s0 - End")

    # print('StreamSync-s1 - Start')
    # s1.synchronize()
    # print('StreamSync-s1 - End')

    # s1.synchronize()
    # print(f'S0={s0.query()} S1={s1.query()}')
    print(f"{s0.id()}")  # {s1.id()}')
    # print(tOut1.cpu())
    # exit()
    # with ht.hpu.stream(s1):
    #     tOut = torch.add(tB_h,tB_h)

    # print(tOut.cpu())

    # with ht.hpu.stream(s1):
    #     print('Inside context for S1')


def testStreamSyncBasic():
    print("Starting testStreamSync")
    print("Creating s0")
    s0 = ht.hpu.Stream()
    print("Creating s1")
    s1 = ht.hpu.Stream()

    print("StreamSync-Default Stream - Start")
    ht.hpu.default_stream().synchronize()
    print("StreamSync-Default Stream - End")

    print("StreamSync-User s0 - Start")
    s0.synchronize()
    print("StreamSync-User s0 - End")
    print("StreamSync-User s1 - Start")
    s1.synchronize()
    print("StreamSync-User s1 - End")


def testAddFwdBwd():
    print("TEST: AddFwdBwd - START")
    s1 = ht.hpu.Stream()
    ht.hpu.Stream()
    in_shape = (10, 2)
    tA_h = torch.zeros(in_shape).to("hpu")
    tB_h = torch.ones(in_shape).to("hpu")
    tC_h = torch.zeros(in_shape).to("hpu")
    tD_h = torch.ones(in_shape).to("hpu")

    tA_h.requires_grad = True
    tB_h.requires_grad = True
    tC_h.requires_grad = True
    tD_h.requires_grad = True

    tOut0 = torch.add(tA_h, tB_h).sum()

    with ht.hpu.stream(s1):
        tOut1 = torch.add(tC_h, tD_h).sum()

    print("START BWD ")
    tOut0.backward()
    tOut1.backward()

    # # TBD: how to sync for default stream ??

    s1.synchronize()

    # # print(f'S0={s0.query()} S1={s1.query()}')
    # print(tOut0.cpu())
    # print(tOut1.cpu())
    # print(tOut2.cpu())

    # with ht.hpu.stream(s1):
    #     print('Inside context for S1')

    print("TEST: AddFwdBwd - END")


def testIf():
    print("TEST: testIf - START")

    print("Creating s1")
    s1 = ht.hpu.Stream()
    print("Creating s2")
    s2 = ht.hpu.Stream()

    print("TEST:OUTSIDE CTX")
    ht.hpu.default_stream()
    ht.hpu.current_stream()
    print("TEST:STARTING CTX")
    with ht.hpu.stream(s1):
        print("INSIDE CTX")
        ht.hpu.default_stream()
        ht.hpu.current_stream()
        print(
            f"Id/device of default stream={ht.hpu.default_stream().id(),ht.hpu.default_stream().device_index} Id/dev of current stream={ht.hpu.current_stream().id(),ht.hpu.current_stream().device_index}"
        )
    print("TEST:Exiting Context")

    print("TEST:Setting stream S2")
    ht.hpu.set_stream(s2)
    ht.hpu.default_stream()
    ht.hpu.current_stream()
    print("TEST:Setting stream to default")
    ht.hpu.set_stream(ht.hpu.default_stream())
    ht.hpu.default_stream()
    ht.hpu.current_stream()

    # breakpoint()


def test_stream_none():
    print("TEST: stream_none - START")
    ht.hpu.stream(None)


def test_stream_event_uninit():
    print("TEST: stream_none - START")
    ht.hpu.Stream()
    # e1 = ht.hpu.Event()


def testInfo():
    d = ht.hpu.default_stream()
    s1 = ht.hpu.Stream()
    s2 = ht.hpu.Stream()
    s1_info = ht.hpu.get_stream_info(s1)
    s2_info = ht.hpu.get_stream_info(s2)
    print(f"S1 Info: On device={s1_info[0]}, stream_id={s1_info[1]}", repr(s1))
    print(f"S2 Info: On device={s2_info[0]}, stream_id={s2_info[1]}", repr(s2))
    print("D==s1  :: ", d == s1)
    print("s1==s1 :: ", s1 == s1)
    print("s1==s2 :: ", s1 == s2)
    # print(s1==3)
    # breakpoint()
    print(f"s1.device_index={s1.device_index} , Default stream id={d.id()} s1.id()={s1.id()} s2.id()={s2.id()}")


def testProfiling():

    in_shape = (10, 2)
    torch.zeros(in_shape).to("hpu")
    torch.ones(in_shape).to("hpu")

    ht.hpu.Stream()
    startEv = ht.hpu.Event(enable_timing=True)
    endEv = ht.hpu.Event(enable_timing=True)
    assert endEv.query() is True, "Event query on unrecorded event returned False (expected True)"
    print(f"Before record :endEv info={repr(endEv)}")
    startEv.record()
    time.sleep(0.5)
    # for _ in range(100):
    #     tA_h = torch.add(tA_h,tB_h)
    endEv.record()
    endEv.synchronize()
    print(f"Time Elapsed={startEv.elapsed_time(endEv)}")  # milliseconds
    print(f"After record :endEv info={repr(endEv)}")


def testProfiling2():

    in_shape = (10, 2)
    tA_h = torch.zeros(in_shape).to("hpu")
    tB_h = torch.ones(in_shape).to("hpu")

    s1 = ht.hpu.Stream()
    s2 = ht.hpu.Stream()
    startEv = ht.hpu.Event(enable_timing=True)
    endEv = ht.hpu.Event(enable_timing=True)
    assert endEv.query() is True, "Event query on unrecorded event returned False (expected True)"
    print(f"Before record :endEv info={repr(endEv)}")
    # startEv.record()
    with ht.hpu.stream(s1):
        tA_h = torch.add(tA_h, tB_h)
    with ht.hpu.stream(s2):
        tA_h = torch.add(tA_h, tB_h)

    s1.record_event(startEv)
    time.sleep(0.5)
    for _ in range(100):
        tA_h = torch.add(tA_h, tB_h)
    endEv.record()
    endEv.synchronize()
    print(f"Time Elapsed={startEv.elapsed_time(endEv)}")  # milliseconds
    print(f"After record :endEv info={repr(endEv)}")


def testEventSyncEmptyGraph():
    print("Starting testEventSyncEmptyGraph TEST")
    ev1 = ht.hpu.Event()
    ev2 = ht.hpu.Event()
    ev1.record()
    s = ht.hpu.Stream()
    s.record_event(ev2)


def testEventSync():
    print("Starting testEventSync TEST")
    in_shape = (10, 2)
    tA_h = torch.zeros(in_shape).to("hpu")
    tB_h = torch.ones(in_shape).to("hpu")

    # s = ht.hpu.Stream()
    startEv = ht.hpu.Event()
    # # print(type(startEv),type(s))
    endEv = ht.hpu.Event()
    # assert endEv.query()is True , "Event query on unrecorded event returned False (expected True)"
    startEv.record()
    print(f"START: start of loop - query()={startEv.query()}")
    for _ in range(3):
        tA_h = torch.add(tA_h, tB_h)
    print(f"START: end of loop - query()={startEv.query()}")
    # htcore.mark_step()
    endEv.record()
    # print(f'START: start of loop2')
    # # endEv.wait()
    # for _ in range(3):
    #     tA_h = torch.add(tA_h,tB_h)
    # print(f'START: end of loop2')
    # # Waits for everything to finish running
    endEv.synchronize()
    # print(tA_h.cpu())


def testEventSyncUserStream():

    print("Starting testEventSyncUserStream TEST")
    print("Create s0")
    s0 = ht.hpu.Stream()
    print("Created s0")
    # s1 = ht.hpu.Stream()
    in_shape = (10, 2)
    tA_h = torch.zeros(in_shape).to("hpu")
    tB_h = torch.ones(in_shape).to("hpu")
    startEv = ht.hpu.Event()
    torch.add(tA_h, tB_h)
    with ht.hpu.stream(s0):
        torch.add(tA_h, tB_h)
        startEv.record()

    startEv.synchronize()
    # print(tA_h.cpu())


def testStreamEvents():
    print("Starting testStreamEvents TEST")
    print("Create s0")
    s0 = ht.hpu.Stream()
    ht.hpu.Event()
    in_shape = (10, 2)
    tA_h = torch.zeros(in_shape).to("hpu")
    tB_h = torch.ones(in_shape).to("hpu")
    ht.hpu.default_stream().record_event()
    u1 = s0.record_event()
    with ht.hpu.stream(s0):
        torch.add(tA_h, tB_h)
    s0.record_event(u1)
    u1.synchronize()


def testStreamEventsSimple():
    print("Starting testStreamEventsSimple TEST")
    print("---" * 5 + "PY:Create & Record e0 Event")
    ht.hpu.default_stream().record_event()
    print("---" * 5 + "PY:Create & Record e1 Event")
    ht.hpu.default_stream().record_event()
    print("---" * 5 + "PY:testStreamEvents finished")
    s0 = ht.hpu.Stream()
    print("---" * 5 + "PY:Create & Record e2 Event on S0")
    s0.record_event()


def testStreamEventsFull():
    print("Starting testStreamEventsFull TEST")
    print("---" * 5 + "PY:Create s0 Stream")
    s0 = ht.hpu.Stream()
    print("---" * 5 + "PY:Create e0 Event")
    e1 = ht.hpu.Event()
    in_shape = (10, 2)
    tA_h = torch.zeros(in_shape).to("hpu")
    tB_h = torch.ones(in_shape).to("hpu")
    print("---" * 5 + "PY:Record  Event on default stream ")
    ht.hpu.default_stream().record_event()
    print("---" * 5 + "PY:Record  Event e0 on user stream s0 ")
    s0.record_event()
    with ht.hpu.stream(s0):
        torch.add(tA_h, tB_h)
    print("---" * 5 + "PY:Record  Event e1 on user stream s0 ")
    s0.record_event(e1)
    e1.synchronize()
    print("---" * 5 + "PY:testStreamEventsFull finished")


def testEventWait():
    print("Starting testEventWait TEST")
    print("---" * 5 + "PY:Create s0 Stream")
    s0 = ht.hpu.Stream()
    print("---" * 5 + "PY:Create e0 Event")
    e1 = ht.hpu.Event()

    in_shape = (4, 2)
    tA_h = torch.zeros(in_shape).to("hpu")
    tB_h = torch.ones(in_shape).to("hpu")
    tC_h = torch.empty_like(tA_h)
    tD_h = torch.empty_like(tA_h)

    with ht.hpu.stream(s0):
        tC_h = tA_h + tB_h
        e1.record()

    e1.wait(ht.hpu.default_stream())
    tD_h = tC_h * 2

    print(f"output={tD_h.cpu()}")
    print("Starting testEventWait TEST - finished")


def testWaitStream():
    print("Starting testWaitStream TEST")
    s0 = ht.hpu.Stream()
    d0 = ht.hpu.default_stream()
    d0.wait_stream(s0)
    s0.wait_stream(d0)


def testStreamWaitEvent():
    print("Starting testStreamWaitEvent TEST")
    s0 = ht.hpu.Stream()
    d0 = ht.hpu.default_stream()

    e1 = ht.hpu.Event()

    in_shape = (4, 2)
    tA_h = torch.zeros(in_shape).to("hpu")
    tB_h = torch.ones(in_shape).to("hpu")
    torch.empty_like(tA_h)
    torch.empty_like(tA_h)

    with ht.hpu.stream(s0):
        tA_h + tB_h
        e1.record()

    d0.wait_event(e1)

    with ht.hpu.stream(d0):
        tA_h + tB_h
        e1.record()

    s0.wait_event(e1)
    print("Starting testStreamWaitEvent TEST - Finished")


def testStreamWaitEventWAR():
    print("Starting testStreamWaitEventWAR TEST")
    s0 = ht.hpu.Stream()
    d0 = ht.hpu.default_stream()

    e1 = ht.hpu.Event()

    in_shape = (4, 2)
    tA = torch.zeros(in_shape)
    tB = torch.ones(in_shape)
    tC = torch.empty_like(tA)
    tD = torch.empty_like(tA)

    tA_h = tA.to("hpu")
    tB_h = tB.to("hpu")
    tC_h = torch.empty_like(tA_h)
    tD_h = torch.empty_like(tA_h)

    tC = tA + tB
    tD = tC
    with ht.hpu.stream(s0):
        tC_h = tA_h + tB_h
        # e1.record()
    s0.record_event(e1)
    d0.wait_event(e1)
    tD_h = tC_h.to("cpu")
    np.testing.assert_allclose(tD.detach().numpy(), tD_h.detach().numpy(), atol=0, rtol=0)

    print("Starting testStreamWaitEventWAR TEST - Finished")


def sync(t, b, a, count):
    with ht.hpu.stream(t):
        t.synchronize()
    np.testing.assert_allclose(b.detach().numpy(), a.detach().numpy(), atol=0, rtol=0)


def testStreamCopyH2DNonBlocking():
    import os

    use_generic_stream = 1
    if "PT_HPU_ENABLE_GENERIC_STREAM" in os.environ:
        use_generic_stream = int(os.environ["PT_HPU_ENABLE_GENERIC_STREAM"])

    # this feature only supported in generic stream
    if not use_generic_stream:
        return

    print("Starting testStreamCopyH2DNonBlocking TEST")
    s0 = ht.hpu.Stream()
    List = []
    count = 0
    while count < 10:
        in_shape = (1000, 2000)
        tA = torch.ones(in_shape)
        tB = torch.ones(in_shape)
        tC = torch.empty_like(tA)

        tA_h = tA.to("hpu", non_blocking=True)
        tB_h = tB.to("hpu", non_blocking=True)
        tC_h = torch.empty_like(tA_h)

        tC = tA + tB
        tC_h = tA_h + tB_h
        y = torch.empty_like(tA).pin_memory(device="hpu")
        with ht.hpu.stream(s0):
            y.copy_(tC_h, non_blocking=True)

        t1 = Thread(target=sync, args=(s0, y, tC, count))
        t1.start()
        List.append(t1)
        count = count + 1
    for x in List:
        x.join()
    print("Finshed testStreamCopyH2DNonBlocking TEST")


def testProfiling_copy_h2d():
    import os

    use_generic_stream = 1
    if "PT_HPU_ENABLE_GENERIC_STREAM" in os.environ:
        use_generic_stream = int(os.environ["PT_HPU_ENABLE_GENERIC_STREAM"])

    # this feature only supported in generic stream
    if not use_generic_stream:
        return

    in_shape = (10, 2)
    cpu_tensor = torch.randn(in_shape)
    s = ht.hpu.Stream()

    startEv = ht.hpu.Event(enable_timing=True)
    endEv = ht.hpu.Event(enable_timing=True)
    startEv.record()
    with ht.hpu.stream(s):
        cpu_tensor.to("hpu")

    endEv.record()
    endEv.synchronize()
    print(f"Time Elapsed={startEv.elapsed_time(endEv)}")  # milliseconds
    print(f"After record :endEv info={repr(endEv)}")


def testProfiling_copy_d2h():
    import os

    use_generic_stream = 1
    if "PT_HPU_ENABLE_GENERIC_STREAM" in os.environ:
        use_generic_stream = int(os.environ["PT_HPU_ENABLE_GENERIC_STREAM"])

    # this feature only supported in generic stream
    if not use_generic_stream:
        return

    in_shape = (10, 2)
    cpu_tensor = torch.randn(in_shape)
    s = ht.hpu.Stream()

    hpu_tensor = cpu_tensor.to("hpu")
    hpu_tensor.fill_(2.2)
    htcore.mark_step()
    startEv = ht.hpu.Event(enable_timing=True)
    endEv = ht.hpu.Event(enable_timing=True)
    startEv.record(s)
    with ht.hpu.stream(s):
        hpu_tensor.to("cpu")
    endEv.record()
    endEv.synchronize()
    print(f"Time Elapsed={startEv.elapsed_time(endEv)}")  # milliseconds
    print(f"After record :endEv info={repr(endEv)}")


def testStreamUseDifferentStreamForEachOP():
    import os

    use_generic_stream = 1
    if "PT_HPU_ENABLE_GENERIC_STREAM" in os.environ:
        use_generic_stream = int(os.environ["PT_HPU_ENABLE_GENERIC_STREAM"])

    # this feature only supported in generic stream
    if not use_generic_stream:
        return

    print("Starting testStreamUseDifferentStreamForEachOP TEST")
    s0 = ht.hpu.Stream()
    s1 = ht.hpu.Stream()
    count = 0
    while count < 10:
        in_shape = (1000, 2000)
        tA = torch.ones(in_shape)
        tB = torch.ones(in_shape)
        tC = torch.empty_like(tA)
        tC = tA + tB

        tA_h = tA.to("hpu", non_blocking=True)
        tB_h = tB.to("hpu", non_blocking=True)
        tC_h = torch.empty_like(tA_h)
        with ht.hpu.stream(s0):
            tC_h = tA_h + tB_h
        y = torch.empty_like(tA)
        with ht.hpu.stream(s1):
            y = tC_h.to("cpu")
        np.testing.assert_allclose(y.detach().numpy(), tC.detach().numpy(), atol=0, rtol=0)
        count = count + 1
    print("Finshed testStreamUseDifferentStreamForEachOP TEST")


def testStreamUseDifferentStreamForEachOPNonBlocking():
    import os

    use_generic_stream = 1
    if "PT_HPU_ENABLE_GENERIC_STREAM" in os.environ:
        use_generic_stream = int(os.environ["PT_HPU_ENABLE_GENERIC_STREAM"])

    # this feature only supported in generic stream
    if not use_generic_stream:
        return

    print("Starting testStreamUseDifferentStreamForEachOPNonBlocking TEST")
    s0 = ht.hpu.Stream()
    s1 = ht.hpu.Stream()
    count = 0
    while count < 10:
        in_shape = (1000, 2000)
        tA = torch.ones(in_shape)
        tB = torch.ones(in_shape)
        tC = torch.empty_like(tA)
        tC = tA + tB

        tA_h = tA.to("hpu", non_blocking=True)
        tB_h = tB.to("hpu", non_blocking=True)
        tC_h = torch.empty_like(tA_h)
        with ht.hpu.stream(s0):
            tC_h = tA_h + tB_h
        y = torch.empty_like(tA).pin_memory(device="hpu")
        with ht.hpu.stream(s1):
            y.copy_(tC_h, non_blocking=True)
        s1.synchronize()
        np.testing.assert_allclose(y.detach().numpy(), tC.detach().numpy(), atol=0, rtol=0)
        count = count + 1
    print("Finshed testStreamUseDifferentStreamForEachOP TEST")


def testCopyNonBlocking():
    print("Start testCopyNonBlocking TEST")

    def _test_copy_non_blocking(a, b, dir1):
        event = ht.hpu.Event()
        a.copy_(b, non_blocking=True)
        event.record()
        event.synchronize()
        if dir1:
            b = b.to("cpu")
        else:
            a = a.to("cpu")
        np.testing.assert_allclose(a.detach().numpy(), b.detach().numpy(), atol=0, rtol=0)
        print("copy done")

    # 10MB copies
    x = torch.ones(10000000, dtype=torch.uint8, device="hpu")
    y = torch.zeros(10000000, dtype=torch.uint8).pin_memory(device="hpu")
    _test_copy_non_blocking(x, y, 0)

    x = torch.zeros(10000000, dtype=torch.uint8).pin_memory(device="hpu")
    y = torch.ones(10000000, dtype=torch.uint8, device="hpu")
    _test_copy_non_blocking(x, y, 1)

    # Test the case where the pinned data_ptr is not equal to the storage data_ptr.
    x_base = torch.zeros(10000000, dtype=torch.uint8).pin_memory(device="hpu")
    x = x_base[1:]
    # commenting below, view not working correctly
    # assert (x.is_pinned(device='hpu') is True)
    assert x_base.is_pinned(device="hpu") is True
    assert x_base.data_ptr() != x.data_ptr()
    assert x_base.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    y = torch.ones(10000000 - 1, dtype=torch.uint8, device="hpu")
    _test_copy_non_blocking(x, y, 1)
    print("Finish testCopyNonBlocking TEST")


def testProfiling_default_stream():

    in_shape = (10, 2)
    tA_h = torch.zeros(in_shape).to("hpu")
    tB_h = torch.ones(in_shape).to("hpu")

    d = ht.hpu.default_stream()
    startEv = ht.hpu.Event(enable_timing=True)
    endEv = ht.hpu.Event(enable_timing=True)
    assert endEv.query() is True, "Event query on unrecorded event returned False (expected True)"
    print(f"Before record :endEv info={repr(endEv)}")
    tA_h = torch.add(tA_h, tB_h)
    tA_h = torch.add(tA_h, tB_h)

    d.record_event(startEv)
    time.sleep(0.5)
    for _ in range(100):
        tA_h = torch.add(tA_h, tB_h)
    endEv.record()
    endEv.synchronize()
    print(f"Time Elapsed={startEv.elapsed_time(endEv)}")  # milliseconds
    print(f"After record :endEv info={repr(endEv)}")


def test_events():
    in_shape = (10000, 2)
    tA_h = torch.zeros(in_shape).to("hpu")
    tB_h = torch.ones(in_shape).to("hpu")
    stream = ht.hpu.current_stream()
    event = ht.hpu.Event(enable_timing=True)
    assert event.query() is True
    start_event = ht.hpu.Event(enable_timing=True)
    stream.record_event(start_event)
    tA_h = torch.add(tA_h, tB_h)
    tA_h = torch.add(tA_h, tB_h)
    htcore.mark_step()
    stream.record_event(event)
    # depends on how fast the op is exectued, so it may return true/false
    # assert event.query() is False
    event.synchronize()
    assert event.query() is True
    print("elaped time value", start_event.elapsed_time(event))


def test_d2h_h2d_default_stream():
    for iter in range(50):
        print(f"------------------- {iter=} -------------------")
        t1 = torch.arange(1, 5, dtype=torch.bfloat16, device="hpu:0")
        t2 = torch.zeros(4)
        t2.copy_(t1, non_blocking=True)
        t2 = t2.to(t1.device)
        assert t2.equal(t1), f"t1 and t2 are not equal! \n{t1=} \n{t2=}"


def test_d2h_h2d_user_stream():
    s0 = ht.hpu.Stream()
    for iter in range(50):
        with ht.hpu.stream(s0):
            print(f"------------------- {iter=} -------------------")
            t1 = torch.arange(1, 5, dtype=torch.bfloat16, device="hpu:0")
            t2 = torch.zeros(4)
            t2.copy_(t1, non_blocking=True)
            t2 = t2.to(t1.device)
            assert t2.equal(t1), f"t1 and t2 are not equal! \n{t1=} \n{t2=}"


if __name__ == "__main__":
    test_stream_none()
    test_stream_event_uninit()
    testStreamSyncBasic()
    testAddOnStreams()
    testAddFwdBwd()
    testIf()
    ht.hpu.set_sync_debug_mode(True)
    testInfo()
    testEventSync()
    testProfiling()
    testEventSyncUserStream()
    testStreamEvents()
    testStreamEventsFull()
    testStreamEventsSimple()
    testEventWait()
    testWaitStream()
    testStreamWaitEvent()
    testStreamWaitEventWAR()
    testEventSyncEmptyGraph()
    testStreamCopyH2DNonBlocking()
    testProfiling_copy_h2d()
    testProfiling_copy_d2h()
    testStreamUseDifferentStreamForEachOP()
    testStreamUseDifferentStreamForEachOPNonBlocking()
    testCopyNonBlocking()
    testProfiling_default_stream()
    test_events()
    test_d2h_h2d_default_stream()
    test_d2h_h2d_user_stream()
