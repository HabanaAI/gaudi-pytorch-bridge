import torch

ops_dict = {
    # This dictionary contains the list of ops which can be considered for casting.
    # In general, each op can belong multiple modules. E.g. add belongs to torch
    # as well as torch.tensor.
    # conv
    "conv1d": [torch.nn.functional, torch],
    "conv2d": [torch.nn.functional, torch],
    "conv3d": [torch.nn.functional, torch],
    "conv_transpose2d": [torch.nn.functional, torch],
    # pool
    "avg_pool2d": [torch.nn.functional],
    "max_pool2d": [torch.nn.functional],
    "max_pool2d_with_indices": [torch.nn.functional],
    # reductions
    "sum": [torch, torch.Tensor],
    "mean": [torch, torch.Tensor],
    "any": [torch, torch.Tensor],
    # GEMM
    "addmm": [torch, torch.Tensor],
    "bmm": [torch, torch.Tensor],
    "dot": [torch, torch.Tensor],
    "linear": [torch.nn.functional],
    "matmul": [torch, torch.Tensor],
    "mm": [torch, torch.Tensor],
    "mv": [torch, torch.Tensor],
    # Misc modules
    "batch_norm": [torch.nn.functional],
    "dropout": [torch.nn.functional],
    "embedding_bag_sum_fwd": [torch],
    "embedding_bag_sum_bwd": [torch],
    "embedding": [torch],
    "layer_norm": [torch.nn.functional],
    "instance_norm": [torch.nn.functional],
    # Classifiers and loss metrics
    "binary_cross_entropy": [torch.nn.functional],
    "cross_entropy": [torch.nn.functional],
    "log_softmax": [torch.nn.functional],
    "softmax": [torch.nn.functional],
    "topk": [torch, torch.Tensor],
    "nll_loss": [torch.nn.functional],
    "mse_loss": [torch.nn.functional],
    # Binary
    "add": [torch, torch.Tensor],
    "addcmul": [torch, torch.Tensor],
    "addcdiv": [torch, torch.Tensor],
    "div": [torch, torch.Tensor],
    "exp": [torch, torch.Tensor],
    "iadd": [torch.Tensor],
    "idiv": [torch.Tensor],
    "imul": [torch.Tensor],
    "ipow": [torch.Tensor],
    "isub": [torch.Tensor],
    "itruediv": [torch.Tensor],
    "mul": [torch, torch.Tensor],
    "pow": [torch, torch.Tensor],
    "rsub": [torch, torch.Tensor],
    "sub": [torch, torch.Tensor],
    "truediv": [torch.Tensor],
    "eq": [torch, torch.Tensor],
    "gt": [torch, torch.Tensor],
    # Activations
    "gelu": [torch.nn.functional],
    "relu": [torch.nn.functional],
    "leaky_relu": [torch.nn.functional],
    # Shapes
    "t": [torch, torch.Tensor],
    "flatten": [torch, torch.Tensor],
    "view": [torch.Tensor],
    "cat": [torch],
}
