import os
import math
import torch

# Our module!
import habanaOptimizerSparseAdagrad_cpp

torch.ops.load_library(
    os.path.join(os.environ["BUILD_ROOT_LATEST"], "libhabana_pytorch_plugin.so")
)
device = torch.device("hpu")


class HabanaOptimizerSparseAdagradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, gradients, weights_in, moments_in, indices, learning_rate, valid_count
    ):
        outputs = habanaOptimizerSparseAdagrad_cpp.forward(
            gradients, weights_in, moments_in, indices, learning_rate, valid_count
        )
        return outputs

    @staticmethod
    def backward(ctx):
        # TODO
        return torch.ones([4, 4])


class HabanaOptimizerSparseAdagrad(torch.nn.Module):
    def __init__(self, table_len, embedding_size):
        super(HabanaOptimizerSparseAdagrad, self).__init__()
        self.gradients = torch.randn(table_len, embedding_size)
        self.weights = torch.randn(table_len, embedding_size)
        self.moments = torch.randn(table_len, embedding_size)
        print("HabanaOptimizerSparseAdagrad")
        print(self.weights)
        print(self.moments)

    def forward(self, indices, valid_count):
        return HabanaOptimizerSparseAdagradFunction.apply(
            self.gradients.to(device),
            self.weights.to(device),
            self.moments.to(device),
            indices,
            learning_rate,
            valid_count,
        )


table_len = 5
embedding_size = 3
num_indices = 4

opt = HabanaOptimizerSparseAdagrad(table_len, embedding_size)

indices = torch.randint(0, table_len, [num_indices]).to(device)
valid_count = torch.randint(0, table_len, [1]).to(device)
learning_rate = torch.tensor([0.1]).to(device)
new_weights, new_moments = opt(indices, valid_count)

print(
    "Weights Output of HabanaOptimizerSparseAdagrad is on {} device".format(
        new_weights.device
    )
)
print("Weights Output tensor\n", new_weights.detach().cpu().numpy())
print(
    "Moments Output of HabanaOptimizerSparseAdagrad is on {} device".format(
        new_moments.device
    )
)
print("Moments Output tensor\n", new_moments.detach().cpu().numpy())
