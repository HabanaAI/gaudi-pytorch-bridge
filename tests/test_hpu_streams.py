import torch
import habana_frameworks.torch.hpu as htcore
import habana_frameworks.torch as ht

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

def testAdd():
    print('Starting Add in Stream Context TEST')
    s0 = ht.hpu.Stream()
    s1 = ht.hpu.Stream()
    in_shape = (10,2)
    tA_h = torch.zeros(in_shape).to('hpu')
    tB_h = torch.ones(in_shape).to('hpu')

    tOut1 = torch.add(tA_h,tA_h)

    with ht.hpu.stream(s0):
        tOut2 = torch.add(tA_h,tB_h)

    with ht.hpu.stream(s1):
        tOut3 = torch.add(tB_h,tB_h)


    s0.synchronize()
    s1.synchronize()
    print(f'S0={s0.query()} S1={s1.query()}')
    print(tOut1.cpu())
    exit()
    with ht.hpu.stream(s1):
        tOut = torch.add(tB_h,tB_h)

    print(tOut.cpu())

    # with ht.hpu.stream(s1):
    #     print('Inside context for S1')

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

    # s1.synchronize()

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

if __name__ == "__main__":
    test_stream_none()
    testAddFwdBwd()
    testIf()
    ht.hpu.set_sync_debug_mode(True)
