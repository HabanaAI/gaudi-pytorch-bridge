import os
import math
import torch

# Our module!
import preproc_cpp

class PreProcFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,indices,offsets,tableLen):
        out1,out2,out3,out4 = preproc_cpp.forward(indices,offsets,tableLen)
        return out1,out2,out3,out4

    @staticmethod
    def backward(ctx):
        #TODO
        return torch.ones([4,4])

class PreProc(torch.nn.Module):
    def __init__(self):
        super(PreProc, self).__init__()

    def forward(self, indices,offsets,tableLen):
        return PreProcFunction.apply(indices,offsets,tableLen)


table_len  = 5
embedding_size = 3
num_indices =  4

# indices = torch.randint(0,table_len,[num_indices])
indices = torch.tensor([0,3,7,2,3,1,4,6,0,1,6],dtype=torch.int32)
offsets = torch.tensor([0,3,5,6,8,11],dtype=torch.int32)
# offsets = torch.tensor([0,3,5,6,8],dtype=torch.int32)
print('Indices\n', indices.detach().cpu().numpy())
print('Offsets\n', indices.detach().cpu().numpy())
valid_count = 2
g=PreProc()
out1,out2,out3,out4 = g(indices,offsets,table_len)

print('Output of preproc is on {} device'.format(out1.device))
print('Input tensor\n', indices.detach().cpu().numpy())
print('Output tensor(countUniqueIndices)\n', out1.detach().cpu().numpy())
print('Output tensor(uniqueIndexes)\n', out2.detach().cpu().numpy())
print('Output tensor(outputRows)\n', out3.detach().cpu().numpy())
print('Output tensor(outputRowOffsets)\n', out4.detach().cpu().numpy())
