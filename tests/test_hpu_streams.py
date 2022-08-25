import torch
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch as ht
import time

def doWork():
    in_shape = (4,2)
    tA_h = torch.zeros(in_shape).to('hpu')
    tB_h = torch.ones(in_shape).to('hpu')

    tC_h = tA_h + tB_h
    print(f'output={tC_h.cpu()}')


def testBasic():
    print('Starting STREAMS BASIC TEST')
    s0 = ht.hpu.Stream()
    s1 = ht.hpu.Stream()

    # breakpoint()
    #print(type(s0))

    print('QUERY s0 - START')
    print(s0.query())
    print('QUERY - FINISHED')
    print('QUERY s1 - START')
    print(s1.query())
    print('QUERY - FINISHED')

    print(s0.synchronize())
    print('SYNC FINISHED')

def testAddOnStreams():
    print('Starting Add in Stream Context TEST')
    print('Create s0')
    s0 = ht.hpu.Stream()
    print('Created s0')
    s1 = ht.hpu.Stream()
    in_shape = (10,2)
    tA_h = torch.zeros(in_shape).to('hpu')
    tB_h = torch.ones(in_shape).to('hpu')

    for _ in range(500):
        tOut1 = torch.add(tA_h,tB_h)

    print('StreamSync-Default Stream - Start')
    ht.hpu.default_stream().synchronize()
    print('StreamSync-Default Stream - End')

    with ht.hpu.stream(s0):
        for _ in range(500):
            tOut2 = torch.add(tA_h,tB_h)

    # with ht.hpu.stream(s1):
    #     tOut3 = torch.add(tB_h,tB_h)

    print('StreamSync-s0 - Start')
    s0.synchronize()
    print('StreamSync-s0 - End')

    # print('StreamSync-s1 - Start')
    # s1.synchronize()
    # print('StreamSync-s1 - End')

    # s1.synchronize()
    # print(f'S0={s0.query()} S1={s1.query()}')
    print(f'{s0.id()}') # {s1.id()}')
    # print(tOut1.cpu())
    # exit()
    # with ht.hpu.stream(s1):
    #     tOut = torch.add(tB_h,tB_h)

    # print(tOut.cpu())

    # with ht.hpu.stream(s1):
    #     print('Inside context for S1')

def testStreamSyncBasic():
    print('Starting testStreamSync')
    print('Creating s0')
    s0 = ht.hpu.Stream()
    print('Creating s1')
    s1 = ht.hpu.Stream()

    print('StreamSync-Default Stream - Start')
    ht.hpu.default_stream().synchronize()
    print('StreamSync-Default Stream - End')

    print('StreamSync-User s0 - Start')
    s0.synchronize()
    print('StreamSync-User s0 - End')
    print('StreamSync-User s1 - Start')
    s1.synchronize()
    print('StreamSync-User s1 - End')

def testAddFwdBwd():
    print('TEST: AddFwdBwd - START')
    s1 = ht.hpu.Stream()
    s2 = ht.hpu.Stream()
    in_shape = (10,2)
    tA_h = torch.zeros(in_shape).to('hpu')
    tB_h = torch.ones(in_shape).to('hpu')
    tC_h = torch.zeros(in_shape).to('hpu')
    tD_h = torch.ones(in_shape).to('hpu')

    tA_h.requires_grad = True
    tB_h.requires_grad = True
    tC_h.requires_grad = True
    tD_h.requires_grad = True


    tOut0 = torch.add(tA_h,tB_h).sum()

    with ht.hpu.stream(s1):
        tOut1 = torch.add(tC_h,tD_h).sum()

    print('START BWD ')
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

    print('TEST: AddFwdBwd - END')

def testIf():
    print('TEST: testIf - START')

    print('Creating s1')
    s1 = ht.hpu.Stream()
    print('Creating s2')
    s2 = ht.hpu.Stream()

    print(f'TEST:OUTSIDE CTX')
    ht.hpu.default_stream()
    ht.hpu.current_stream()
    print(f'TEST:STARTING CTX')
    with ht.hpu.stream(s1):
        print(f'INSIDE CTX')
        ht.hpu.default_stream()
        ht.hpu.current_stream()
        print(f'Id/device of default stream={ht.hpu.default_stream().id(),ht.hpu.default_stream().device_index()} Id/dev of current stream={ht.hpu.current_stream().id(),ht.hpu.current_stream().device_index()}')
    print('TEST:Exiting Context')

    print('TEST:Setting stream S2')
    ht.hpu.set_stream(s2)
    ht.hpu.default_stream()
    ht.hpu.current_stream()
    print('TEST:Setting stream to default')
    ht.hpu.set_stream(ht.hpu.default_stream())
    ht.hpu.default_stream()
    ht.hpu.current_stream()

    # breakpoint()

def test_stream_none():
    print('TEST: stream_none - START')
    ht.hpu.stream(None)

def test_stream_event_uninit():
    print('TEST: stream_none - START')
    s1 = ht.hpu.Stream()
    # e1 = ht.hpu.Event()


def testInfo():
    d = ht.hpu.default_stream()
    s1 = ht.hpu.Stream()
    s2 = ht.hpu.Stream()
    s1_info = ht.hpu.get_stream_info(s1)
    s2_info = ht.hpu.get_stream_info(s2)
    print(f'S1 Info: On device={s1_info[0]}, stream_id={s1_info[1]}',repr(s1))
    print(f'S2 Info: On device={s2_info[0]}, stream_id={s2_info[1]}',repr(s2))
    print('D==s1  :: ',d==s1)
    print('s1==s1 :: ',s1==s1)
    print('s1==s2 :: ',s1==s2)
    # print(s1==3)
    # breakpoint()
    print(f's1.device_index={s1.device_index()} , Default stream id={d.id()} s1.id()={s1.id()} s2.id()={s2.id()}')

def testProfiling():

    in_shape = (10,2)
    tA_h = torch.zeros(in_shape).to('hpu')
    tB_h = torch.ones(in_shape).to('hpu')

    s = ht.hpu.Stream()
    startEv =ht.hpu.Event(enable_timing=True)
    endEv = ht.hpu.Event(enable_timing=True)
    assert endEv.query()== True , "Event query on unrecorded event returned False (expected True)"
    print(f'Before record :endEv info={repr(endEv)}')
    startEv.record()
    time.sleep(0.5)
    # for _ in range(100):
    #     tA_h = torch.add(tA_h,tB_h)
    endEv.record()
    endEv.synchronize()
    print(f'Time Elapsed={startEv.elapsed_time(endEv)}')  # milliseconds
    print(f'After record :endEv info={repr(endEv)}')


def testProfiling2():

    in_shape = (10,2)
    tA_h = torch.zeros(in_shape).to('hpu')
    tB_h = torch.ones(in_shape).to('hpu')

    s = ht.hpu.Stream()
    startEv =ht.hpu.Event(enable_timing=True)
    endEv = ht.hpu.Event(enable_timing=True)
    assert endEv.query()== True , "Event query on unrecorded event returned False (expected True)"
    print(f'Before record :endEv info={repr(endEv)}')
    # startEv.record()
    with ht.hpt.stream(s1):
        tA_h = torch.add(tA_h,tB_h)
    with ht.hpt.stream(s2):
        tA_h = torch.add(tA_h,tB_h)

    s1.record_event(startEv)
    time.sleep(0.5)
    for _ in range(100):
        tA_h = torch.add(tA_h,tB_h)
    endEv.record()
    endEv.synchronize()
    print(f'Time Elapsed={startEv.elapsed_time(endEv)}')  # milliseconds
    print(f'After record :endEv info={repr(endEv)}')

def testEventSyncEmptyGraph():
    print('Starting testEventSyncEmptyGraph TEST')
    ev1 =ht.hpu.Event()
    ev2 = ht.hpu.Event()
    ev1.record()
    s = ht.hpu.Stream()
    s.record_event(ev2)


def testEventSync():
    print('Starting testEventSync TEST')
    in_shape = (10,2)
    tA_h = torch.zeros(in_shape).to('hpu')
    tB_h = torch.ones(in_shape).to('hpu')

    # s = ht.hpu.Stream()
    startEv =ht.hpu.Event()
    # # print(type(startEv),type(s))
    endEv = ht.hpu.Event()
    # assert endEv.query()== True , "Event query on unrecorded event returned False (expected True)"
    startEv.record()
    print(f'START: start of loop - query()={startEv.query()}')
    for _ in range(3):
        tA_h = torch.add(tA_h,tB_h)
    print(f'START: end of loop - query()={startEv.query()}')
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

    print('Starting testEventSyncUserStream TEST')
    print('Create s0')
    s0 = ht.hpu.Stream()
    print('Created s0')
    # s1 = ht.hpu.Stream()
    in_shape = (10,2)
    tA_h = torch.zeros(in_shape).to('hpu')
    tB_h = torch.ones(in_shape).to('hpu')
    startEv =ht.hpu.Event()
    tOut1 = torch.add(tA_h,tB_h)
    with ht.hpu.stream(s0):
        tOut2 = torch.add(tA_h,tB_h)
        startEv.record()

    startEv.synchronize()
    # print(tA_h.cpu())

def testStreamEvents():
    print('Starting testStreamEvents TEST')
    print('Create s0')
    s0 = ht.hpu.Stream()
    u0 = ht.hpu.Event()
    in_shape = (10,2)
    tA_h = torch.zeros(in_shape).to('hpu')
    tB_h = torch.ones(in_shape).to('hpu')
    d0 = ht.hpu.default_stream().record_event()
    u1 = s0.record_event()
    with ht.hpu.stream(s0):
        tOut2 = torch.add(tA_h,tB_h)
    s0.record_event(u1)
    u1.synchronize()

def testStreamEventsSimple():
    print('Starting testStreamEventsSimple TEST')
    print('---'*5 + 'PY:Create & Record e0 Event')
    e0 = ht.hpu.default_stream().record_event()
    print('---'*5 + 'PY:Create & Record e1 Event')
    e1 = ht.hpu.default_stream().record_event()
    print('---'*5 + 'PY:testStreamEvents finished')
    s0 = ht.hpu.Stream()
    print('---'*5 + 'PY:Create & Record e2 Event on S0')
    e2 = s0.record_event()

def testStreamEventsFull():
    print('Starting testStreamEventsFull TEST')
    print('---'*5 + 'PY:Create s0 Stream')
    s0 = ht.hpu.Stream()
    print('---'*5 + 'PY:Create e0 Event')
    e1 = ht.hpu.Event()
    in_shape = (10,2)
    tA_h = torch.zeros(in_shape).to('hpu')
    tB_h = torch.ones(in_shape).to('hpu')
    print('---'*5 + 'PY:Record  Event on default stream ')
    d0 = ht.hpu.default_stream().record_event()
    print('---'*5 + 'PY:Record  Event e0 on user stream s0 ')
    e0 = s0.record_event()
    with ht.hpu.stream(s0):
        tOut2 = torch.add(tA_h,tB_h)
    print('---'*5 + 'PY:Record  Event e1 on user stream s0 ')
    s0.record_event(e1)
    e1.synchronize()
    print('---'*5 + 'PY:testStreamEventsFull finished')

def testEventWait():
    print('Starting testEventWait TEST')
    print('---'*5 + 'PY:Create s0 Stream')
    s0 = ht.hpu.Stream()
    print('---'*5 + 'PY:Create e0 Event')
    e1 = ht.hpu.Event()

    in_shape = (4,2)
    tA_h = torch.zeros(in_shape).to('hpu')
    tB_h = torch.ones(in_shape).to('hpu')
    tC_h = torch.empty_like(tA_h)
    tD_h = torch.empty_like(tA_h)

    with ht.hpu.stream(s0):
        tC_h = tA_h + tB_h
        e1.record()

    e1.wait(ht.hpu.default_stream())
    tD_h = tC_h * 2

    print(f'output={tD_h.cpu()}')
    print('Starting testEventWait TEST - finished')

def testWaitStream():
    print('Starting testWaitStream TEST')
    s0 = ht.hpu.Stream()
    d0 = ht.hpu.default_stream()
    d0.wait_stream(s0)
    s0.wait_stream(d0)

def testStreamWaitEvent():
    print('Starting testStreamWaitEvent TEST')
    s0 = ht.hpu.Stream()
    d0 = ht.hpu.default_stream()

    e1 = ht.hpu.Event()

    in_shape = (4,2)
    tA_h = torch.zeros(in_shape).to('hpu')
    tB_h = torch.ones(in_shape).to('hpu')
    tC_h = torch.empty_like(tA_h)
    tD_h = torch.empty_like(tA_h)

    with ht.hpu.stream(s0):
        tC_h = tA_h + tB_h
        e1.record()

    d0.wait_event(e1)

    with ht.hpu.stream(d0):
        tC_h = tA_h + tB_h
        e1.record()

    s0.wait_event(e1)
    print('Starting testStreamWaitEvent TEST - Finished')


def testStreamWaitEventWAR():
    print('Starting testStreamWaitEventWAR TEST')
    s0 = ht.hpu.Stream()
    d0 = ht.hpu.default_stream()

    e1 = ht.hpu.Event()

    in_shape = (4,2)
    tA_h = torch.zeros(in_shape).to('hpu')
    tB_h = torch.ones(in_shape).to('hpu')
    tC_h = torch.empty_like(tA_h)
    tD_h = torch.empty_like(tA_h)

    with ht.hpu.stream(s0):
        tC_h = tA_h + tB_h
        # e1.record()
    s0.record_event(e1)
    d0.wait_event(e1)
    tD_h = tC_h.to(dtype=torch.bfloat16)
    htcore.mark_step()

    print('Starting testStreamWaitEventWAR TEST - Finished')

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
