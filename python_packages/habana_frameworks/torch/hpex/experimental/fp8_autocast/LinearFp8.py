from typing import Tuple, Union
import torch
from habana_frameworks.torch import _hpex_C

_HPU_LINEAR = _hpex_C.linear_ex
_HPU_LINEAR_BACKWARD = _hpex_C.linear_ex_backward

class LinearFp8NoSr(torch.autograd.Function):
    """
    Fused operation which performs following two operations in sequence
    1. Downcast input tensors to fp8.
    2. Perform weight transpose.
    3. Perform matmul.
    4. Perform biad add
    """
    @staticmethod
    def cast_to_fp8_fwd(tensor):
        return tensor.to(torch.fp8r152)

    @staticmethod
    def cast_to_fp8_bwd(tensor):
        return tensor.to(torch.fp8r152)

    @classmethod
    def forward(cls, ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        """LinearFp8NoSr fwd"""
        bias_bf16 = bias.to(torch.bfloat16) if bias is not None else None
        ctx.save_for_backward(input, weight, bias_bf16)
        return _HPU_LINEAR(cls.cast_to_fp8_fwd(input), cls.cast_to_fp8_fwd(weight), bias_bf16, torch.bfloat16).to(torch.bfloat16)

    @classmethod
    def backward(cls,
        ctx, output_grads: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """LinearFp8NoSr bwd"""
        input, weight, bias_bf16 = ctx.saved_tensors
        grad_0, grad_1, grad_2 = _HPU_LINEAR_BACKWARD(
            cls.cast_to_fp8_bwd(output_grads),
            cls.cast_to_fp8_bwd(input),
            cls.cast_to_fp8_bwd(weight),
            bias_bf16,
            output_grads.to(torch.bfloat16),
            torch.bfloat16)
        return grad_0.to(torch.bfloat16), grad_1.to(torch.bfloat16), grad_2.to(torch.bfloat16) if grad_2 is not None else None

class LinearFp8SrBwdOnly(LinearFp8NoSr):
    """
    Fused operation which performs following two operations in sequence
    1. Downcast input tensors to fp8 with SR mode in the bwd part.
    2. Perform weight transpose.
    3. Perform matmul.
    4. Perform biad add
    """
    @staticmethod
    def cast_to_fp8_bwd(tensor):
        from habana_frameworks.torch.hpex.kernels.CastToFp8 import cast_to_fp8
        return cast_to_fp8(tensor, True)

