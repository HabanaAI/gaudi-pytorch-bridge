from .MatmulFp8 import MatmulFp8NoSr, MatmulFp8SrBwdOnly
from .LinearFp8 import LinearFp8NoSr, LinearFp8SrBwdOnly
import torch

_TORCH_MATMUL = torch.matmul
_TORCH_LINEAR = torch.nn.functional.linear

def matmul_fp8(input: torch.Tensor, other: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    return MatmulFp8NoSr.apply(input, other, out)

def matmul_fp8_sr_bwd(input: torch.Tensor, other: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    return MatmulFp8SrBwdOnly.apply(input, other, out)

def matmul(input: torch.Tensor, other: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    return _TORCH_MATMUL(input, other, out=out)

def linear_fp8(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    return LinearFp8NoSr.apply(input, weight, bias)

def linear_fp8_sr_bwd(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    return LinearFp8SrBwdOnly.apply(input, weight, bias)

def linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    return _TORCH_LINEAR(input, weight, bias)

_FP8_MATMUL = {"no_sr": matmul_fp8, "sr_bwd": matmul_fp8_sr_bwd, "none": matmul}
_FP8_LINEAR = {"no_sr": linear_fp8, "sr_bwd": linear_fp8_sr_bwd, "none": linear}

class Fp8Autocast:
    def __init__(self, mode="no_sr"):
        self.mode = mode
        self.torch_matmul = None
        self.torch_linear = None

    def __enter__(self):
        self.torch_matmul = torch.matmul
        self.torch_linear = torch.nn.functional.linear
        torch.matmul = _FP8_MATMUL[self.mode]
        torch.nn.functional.linear = _FP8_LINEAR[self.mode]

    def __exit__(self, *args):
        torch.matmul = self.torch_matmul
        torch.nn.functional.linear = self.torch_linear
