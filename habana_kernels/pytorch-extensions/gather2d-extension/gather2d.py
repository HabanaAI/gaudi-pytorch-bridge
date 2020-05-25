import os
import math
import torch

# Our module!
import gather2d_cpp

torch.ops.load_library(os.path.join(os.environ['BUILD_ROOT_LATEST'], "libhabana_pytorch_plugin.so"))
device = torch.device("habana")

class Gather2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,weights,indices, valid_count ):
        outputs = gather2d_cpp.forward(weights,indices, valid_count)
        return outputs

    @staticmethod
    def backward(ctx):
        #TODO
        return torch.ones([4,4])

class Gather2D(torch.nn.Module):
    def __init__(self, table_len, embedding_size):
        super(Gather2D, self).__init__()
        self.weights = torch.randn(table_len,embedding_size)
        print('Embedding Table Created. Content:')
        print(self.weights)

    def forward(self, indices, valid_count):
        return Gather2DFunction.apply(self.weights.to(device), indices, valid_count)


table_len  = 5
embedding_size = 3
num_indices =  4

g= Gather2D(table_len,embedding_size)

indices = torch.randint(0,table_len,[num_indices]).to(device)
valid_count = 2
out = g(indices, valid_count)

print('Output of gather2D is on {} device'.format(out.device))
print('Output tensor\n', out.detach().cpu().numpy())
