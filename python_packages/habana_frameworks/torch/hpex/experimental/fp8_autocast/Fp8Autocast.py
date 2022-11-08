from .MatmulFp8 import MatmulFp8NoSr, MatmulFp8SrBwdOnly
import torch

_TORCH_MATMUL = torch.matmul
_TORCH_LINEAR = torch.nn.functional.linear

def custom_linear(input, weight, bias=None):
    tmp = torch.matmul(input, weight.t())
    if bias is not None:
        tmp = tmp + bias
    return tmp

def matmul_fp8(input: torch.Tensor, other: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    return MatmulFp8NoSr.apply(input, other, out)

def matmul_fp8_sr_bwd(input: torch.Tensor, other: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    return MatmulFp8SrBwdOnly.apply(input, other, out)

def matmul(input: torch.Tensor, other: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    return _TORCH_MATMUL(input, other, out=out)

_FP8_MATMUL = {"no_sr": matmul_fp8, "sr_bwd": matmul_fp8_sr_bwd, "none": matmul}

class Fp8Autocast:
    def __init__(self, mode="no_sr"):
        self.mode = mode
        self.torch_matmul = None
        self.torch_linear = None

    def __enter__(self):
        self.torch_matmul = torch.matmul
        self.torch_linear = torch.nn.functional.linear
        torch.nn.functional.linear = custom_linear
        torch.matmul = _FP8_MATMUL[self.mode]

    def __exit__(self, *args):
        torch.matmul = self.torch_matmul
        torch.nn.functional.linear = self.torch_linear
