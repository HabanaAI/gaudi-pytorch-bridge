import os
import math
import torch
import numpy as np

# Our module!
import HabanaEmbeddingBag_cpp

torch.ops.load_library(os.path.join(os.environ['BUILD_ROOT_LATEST'], "libhabana_pytorch_plugin.so"))
device = torch.device("habana")


class EmbeddingBagSumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, indices, offsets, valid_count, kernel_mode):
        outputs = HabanaEmbeddingBag_cpp.forward(weights, indices, offsets, valid_count, kernel_mode)
        return outputs

    @staticmethod
    def backward(ctx):
        # TODO
        return torch.ones([4, 4])


class HabanaEmbeddingBag(torch.nn.Module):
    def __init__(self, table_len, embedding_size):
        super(HabanaEmbeddingBag, self).__init__()
        #self.weights = torch.randn(table_len, embedding_size)
        self.weights = torch.ones(table_len, embedding_size)
        print('Embedding Table Created. Content:')
        print(self.weights)

    def forward(self, indices, offsets, valid_count, kernel_mode):
        return EmbeddingBagSumFunction.apply(self.weights.to(device), indices, offsets, valid_count, kernel_mode)


table_len = 5
embedding_size = 3
num_indices = 4

emb = HabanaEmbeddingBag(table_len, embedding_size)

indices = torch.LongTensor([0, 1, 2, 3]).to(device)
offsets = torch.LongTensor([0, 2, 4]).to(device)
valid_count = torch.LongTensor([4, 3]).to(device)
kernel_mode = 1
out = emb(indices, offsets, valid_count, kernel_mode)

print('indices', indices.detach().cpu().numpy())
print('Output of embedding bag sum is on {} device'.format(out.device))
print('Output tensor\n', out.detach().cpu().numpy())
