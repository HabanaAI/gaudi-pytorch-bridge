import torch
# This dictionary contains the list of ops which are supported for casting.
# In general, each op can belong multiple modules. Ex: add belongs to torch as well as
# torch.tensor to handle out-of-place and inplace operations respectively
ops_dict = {
    # conv
    'conv1d': [torch.nn.functional, torch],
    'conv2d': [torch.nn.functional, torch],
    'conv3d': [torch.nn.functional, torch],

    # GEMM
    'bmm': [torch, torch.Tensor],
    'dot': [torch, torch.Tensor],
    'linear': [torch.nn.functional],
    'matmul': [torch, torch.Tensor],
    'mm': [torch, torch.Tensor],
    'mv': [torch, torch.Tensor],


    # Misc nn modules
    'batch_norm': [torch.nn.functional],

    # Classifiers and loss metrics
    'cross_entropy': [torch.nn.functional],
    'log_softmax': [torch.nn.functional],
    'softmax': [torch.nn.functional],
    'topk': [torch, torch.Tensor],

    # Binary
    'add': [torch, torch.Tensor],
    'addcmul': [torch, torch.Tensor],
    'addcdiv': [torch, torch.Tensor],
    'div': [torch, torch.Tensor],
    'exp': [torch, torch.Tensor],
    'mul': [torch, torch.Tensor],
    'pow': [torch, torch.Tensor],
    'sub': [torch, torch.Tensor],

    # Activations
    'gelu': [torch.nn.functional],
    'relu': [torch.nn.functional],
}
