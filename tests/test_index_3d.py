import torch
import habana_frameworks.torch.core as htcore

s0 = 4
s1 = 3
s2 = 3
def index_original(device) -> torch.Tensor:
    x = torch.arange(s0*s1*s2, device=device).view(s0, s1, s2)
    #x = torch.arange(s0*s1*s2).view(s0, s1, s2).to(device)
    p = torch.Tensor([0, 2]).to(device).to(torch.int64)
    #z = torch.Tensor([1, 2]).to(device).to(torch.int64)
    q = torch.Tensor([1, 2]).to(device).to(torch.int64)
    r = torch.Tensor([1]).to(device).to(torch.int64)
    bmask = torch.tensor([[False, True, False],[True, False, True],[False,True,True]]).to(device)
    bmask1 = torch.tensor([False, True, False]).to(device)
    #return x[p, q, r]
    #return x[p, q, :]
    #return x[:, q, r]
    #return x[p, :, q]
    #return x[p, :, :]
    #return x[:, q, :]
    #return x[:, :, q]
    #return x[..., q]
    #return x[q,...]
    #return x[:,bmask]
    return x[:,torch.tensor([0,1]).to('cpu'),bmask1]
    #return x[torch.tensor([1,3]).to('cpu'),torch.tensor([0,1]),torch.tensor([1,2])]
if __name__ == '__main__':
    device = torch.device("hpu")
    index_res = index_original(device)
    print('device = ',device, ', res shape = ',index_res.shape, ', res = ',index_res.to("cpu"))
    device = torch.device("cpu")
    index_res = index_original(device)
    print('device = ',device, ', res shape = ',index_res.shape, ', res = ',index_res.to("cpu"))
