from typing import Tuple, Union
import torch
from habana_frameworks.torch import _hpex_C

_TORCH_MATMUL = torch.matmul
_HPU_MATMUL = _hpex_C.matmul_ex
_HPU_MATMUL_BACKWARD = _hpex_C.matmul_ex_backward

class MatmulFp8NoSr(torch.autograd.Function):
    """
    Fused operation which performs following two operations in sequence
    1. Downcast input tensors to fp8.
    2. Perform matmul.
    """
    @staticmethod
    def cast_to_fp8_fwd(tensor):
        return tensor.to(torch.fp8r152)

    @staticmethod
    def cast_to_fp8_bwd(tensor):
        return tensor.to(torch.fp8r152)

    @classmethod
    def forward(cls, ctx, input: torch.Tensor, other: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        """MatmulFp8NoSr fwd"""
        input_fp8 = cls.cast_to_fp8_fwd(input)
        other_fp8 = cls.cast_to_fp8_fwd(other)
        ctx.save_for_backward(input_fp8, other_fp8)
        return _HPU_MATMUL(input_fp8, other_fp8, torch.bfloat16).to(torch.bfloat16)


    @classmethod
    def backward(cls,
        ctx, output_grads: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """MatmulFp8NoSr bwd"""
        input_fp8, other_fp8 = ctx.saved_tensors
        grad_0, grad_1 = _HPU_MATMUL_BACKWARD(cls.cast_to_fp8_bwd(output_grads), input_fp8, other_fp8, torch.bfloat16)
        return grad_0.to(torch.bfloat16), grad_1.to(torch.bfloat16), None

class MatmulFp8SrBwdOnly(MatmulFp8NoSr):
    """
    Fused operation which performs following two operations in sequence
    1. Downcast input tensors to fp8 with SR mode in the bwd part.
    2. Perform matmul.
    """
    @staticmethod
    def cast_to_fp8_bwd(tensor):
        from habana_frameworks.torch.hpex.kernels.CastToFp8 import cast_to_fp8
        return cast_to_fp8(tensor, True)

