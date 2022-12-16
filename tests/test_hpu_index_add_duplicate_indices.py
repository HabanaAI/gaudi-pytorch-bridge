# Tests for index add with duplicate indices for the following
# scenarios and dims 0 or 1
# scenario 0 : # index size = self size at specified dim 0 or 1
# scenario 1 : # index size > self size at specified dim 0 or 1

import torch
scenario = 1

def test_index_add2(device, seed, st):
    dtype = torch.int32
    torch.manual_seed(seed)
    print("seed = ",seed)
    print("Running on device = ", device)
    if device == 'hpu':
        import habana_frameworks.torch.core as htcore
    d = 0
    alpha =1
    sort = st
    sf = (10, 10) # self shape
    # k = index shape
    # u = range of indices
    if d == 0:
        if scenario == 0: # index size <= self size
            k = sf[0]
            u = int(sf[0]/2)
        else:
            k = sf[0]*2 # index size > self size
            u = sf[0]
        ss = (k, sf[1]) # source shape
    else:
        if scenario == 0: # index size <= self size
            k = sf[1]
            u = int(sf[1]/2)
        else:
            k = sf[1]*2 # index size > self size
            u = sf[1]
        ss = (sf[0], k) # source shape

    self  = torch.arange(1, sf[0]*sf[1]+1).reshape(sf).to(dtype).to(device)
    #print("self =", self)
    if sort:
        a,b = torch.sort(torch.randint(0,u ,(k,)))
        index = a.to(device)
    else:
        index = torch.randint(0,u ,(k,)).to(device)

    source = torch.ones(ss).to(dtype).to(device)

    #print("source =", source)
    print("index =", index)

    r = self.index_add_(d, index, source, alpha=1)

    #print("result =", r.to("cpu"))

    if device == 'hpu':
        htcore.mark_step()
    return r.to("cpu")

import random
N = 20

mms =0
mmu =0
for i in range(N):
    print("*"*80)
    seed = random.randint(10000, 99999999)
    st = True
    c = test_index_add2("cpu", seed, st)
    h = test_index_add2("hpu", seed, st)
    ms = torch.equal(c, h)
    print("\nsorted match ", torch.equal(c, h))
    st = False
    c = test_index_add2("cpu", seed, st)
    h = test_index_add2("hpu", seed, st)
    mu = torch.equal(c, h)
    mms = mms +ms
    mmu = mmu +mu

    print("\nunsorted match ", torch.equal(c, h))
    print("*"*80)

print ("runs matched with CPU sorted case = ", mms,  "/", N, "unsorted case  = ", mmu, "/", N)
