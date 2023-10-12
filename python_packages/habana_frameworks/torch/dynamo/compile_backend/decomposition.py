###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

from typing import Optional

import torch
import torch._prims_common as utils
from torch._decomp import get_decompositions

aten = torch.ops.aten

from .config import configuration_flags
from .logger import get_compile_backend_logger

logger = get_compile_backend_logger()

# List of built-in pytorch framework decompositions we would like to use in HPU
# backend in both training and inference.
hpu_backend_decompositions_common = get_decompositions(
    [
        aten.addcdiv.default,
        aten.addcdiv.out,
        aten.addcdiv_.default,
        aten.addcmul.default,
        aten.addcmul.out,
        aten.addcmul_.default,
        aten.addr.default,
        aten.addr.out,
        aten.binary_cross_entropy.out,
        aten.binary_cross_entropy_backward.default,
        aten.binary_cross_entropy_backward.grad_input,
        aten.binary_cross_entropy_with_logits.default,
        aten.binary_cross_entropy_with_logits.out,
        aten.bucketize.Tensor,
        aten.bucketize.Tensor_out,
        aten.bucketize.Scalar,
        aten.bucketize.Scalar_out,
        aten.celu.default,
        aten.celu.out,
        aten.col2im.default,
        aten.col2im.out,
        aten.cudnn_batch_norm.default,
        aten.cudnn_batch_norm.out,
        aten.cudnn_batch_norm_backward.default,
        aten.cudnn_batch_norm_backward.out,
        aten.detach.default,
        aten.diag_embed.default,
        aten.diag_embed.out,
        aten.diagonal.default,
        aten.diagonal.Dimname,
        aten.dot.default,
        aten.dot.out,
        aten.elu.default,
        aten.elu.out,
        aten.elu_backward.default,
        aten.elu_backward.grad_input,
        aten.embedding_dense_backward.out,
        aten.eye.default,
        aten.eye.m,
        aten.eye.out,
        aten.eye.m_out,
        aten.fill.Scalar,
        aten.fill.Tensor,
        aten.frac.default,
        aten.frac.out,
        aten.gelu.out,
        aten.gelu_backward.grad_input,
        aten.glu_backward.default,
        aten.glu_backward.grad_input,
        aten.grid_sampler_2d.out,
        aten.hardshrink.out,
        aten.hardshrink_backward.grad_input,
        aten.hardsigmoid.out,
        aten.hardsigmoid_backward.grad_input,
        aten.hardswish.default,
        aten.hardswish.out,
        aten.hardswish_.default,
        aten.hardswish_backward.default,
        aten.hardswish_backward.out,
        aten.hardtanh.out,
        aten.hardtanh_.default,
        aten.hardtanh_backward.grad_input,
        aten.heaviside.out,
        aten.huber_loss.out,
        aten.huber_loss_backward.out,
        aten.im2col.default,
        aten.im2col.out,
        aten.index_add.out,
        aten.index_add.dimname,
        aten.index_add_.default,
        aten.index_copy.default,
        aten.index_copy.dimname,
        aten.index_copy.out,
        aten.index_copy_.default,
        aten.index_copy_.dimname,
        aten.index_fill.int_Tensor,
        aten.index_fill.int_Scalar,
        aten.index_fill.Dimname_Scalar,
        aten.index_fill.Dimname_Tensor,
        aten.index_fill.int_Scalar_out,
        aten.index_fill.int_Tensor_out,
        aten.index_fill_.int_Tensor,
        aten.index_fill_.int_Scalar,
        aten.index_fill_.Dimname_Scalar,
        aten.index_fill_.Dimname_Tensor,
        aten.index_select.out,
        aten.index_select.dimname,
        aten.index_select.dimname_out,
        aten.isneginf.default,
        aten.isneginf.out,
        aten.isposinf.default,
        aten.isposinf.out,
        aten.leaky_relu.out,
        aten.leaky_relu_.default,
        aten.leaky_relu_backward.grad_input,
        aten.lerp.Scalar,
        aten.lerp.Scalar_out,
        aten.lerp.Tensor_out,
        aten.linspace.default,
        aten.linspace.out,
        aten.logaddexp.default,
        aten.logaddexp.out,
        aten.logit.out,
        aten.log_sigmoid_backward.grad_input,
        aten.log_sigmoid_forward.output,
        aten._log_softmax.out,
        aten._log_softmax_backward_data.out,
        aten.logspace.default,
        aten.logspace.out,
        aten.logsumexp.default,
        aten.masked_fill.Tensor,
        aten.masked_fill.Scalar,
        aten.masked_fill.Scalar_out,
        aten.masked_fill.Tensor_out,
        aten.masked_fill_.Scalar,
        aten.masked_fill_.Tensor,
        aten.mish.default,
        aten.mish.out,
        aten.mse_loss.default,
        aten.mse_loss.out,
        aten.mse_loss_backward.default,
        aten.mse_loss_backward.grad_input,
        aten.mv.default,
        aten.mv.out,
        aten.mvlgamma.default,
        aten.mvlgamma.out,
        aten.nan_to_num.out,
        aten.native_batch_norm.default,
        aten.native_batch_norm.out,
        aten.native_batch_norm_backward.out,
        aten._native_batch_norm_legit.default,
        aten.native_dropout_backward.default,
        aten.native_dropout_backward.out,
        aten.native_group_norm.default,
        aten.native_group_norm_backward.default,
        aten.native_group_norm_backward.out,
        aten.native_layer_norm.out,
        aten.native_layer_norm_backward.out,
        aten.new_empty.default,
        aten.new_empty.out,
        aten.new_empty_strided.default,
        aten.new_full.default,
        aten.new_full.out,
        aten.new_ones.default,
        aten.new_ones.out,
        aten.new_zeros.default,
        aten.new_zeros.out,
        aten.nll_loss_backward.grad_input,
        aten.nll_loss_forward.output,
        aten.norm.Scalar,
        aten.norm.ScalarOpt_dim,
        aten.norm.names_ScalarOpt_dim,
        aten.norm.ScalarOpt_dim_dtype,
        aten.norm.dtype_out,
        aten.norm.out,
        aten.norm.ScalarOpt_dtype,
        aten.norm.ScalarOpt_dtype_out,
        aten.norm.Scalar_out,
        aten.norm.names_ScalarOpt_dim_dtype,
        aten.norm.names_dtype_out,
        aten.norm.names_out,
        aten.ones_like.out,
        aten._prelu_kernel.default,
        aten._prelu_kernel_backward.default,
        aten._reshape_alias.default,
        aten.rot90.default,
        aten.rot90.out,
        aten.rsub.Tensor,
        aten.select_backward.default,
        aten.select_backward.out,
        aten.sgn.default,
        aten.sgn.out,
        aten.sigmoid_backward.default,
        aten.sigmoid_backward.grad_input,
        aten.silu.default,
        aten.silu.out,
        aten.silu_.default,
        aten.silu_backward.default,
        aten.silu_backward.grad_input,
        aten.sinc.out,
        aten.slice_backward.default,
        aten.slice_backward.out,
        aten.soft_margin_loss.default,
        aten.soft_margin_loss.out,
        aten.soft_margin_loss_backward.default,
        aten.soft_margin_loss_backward.grad_input,
        aten._softmax.out,
        aten._softmax_backward_data.out,
        aten.softplus.default,
        aten.softplus.out,
        aten.softplus_backward.default,
        aten.softplus_backward.grad_input,
        aten.softshrink.out,
        aten.softshrink_backward.grad_input,
        aten.special_entr.out,
        aten.special_log_ndtr.default,
        aten.special_log_ndtr.out,
        aten.special_xlog1py.default,
        aten.special_xlog1py.other_scalar,
        aten.special_xlog1py.self_scalar,
        aten.special_xlog1py.out,
        aten.special_xlog1py.self_scalar_out,
        aten.special_xlog1py.other_scalar_out,
        aten.stack.default,
        aten.stack.out,
        aten.tanh_backward.default,
        aten.tanh_backward.grad_input,
        aten.threshold.out,
        aten.threshold_backward.grad_input,
        aten.trace.out,
        aten.unbind.int,
        aten.unfold.default,
        aten.unfold_backward.default,
        aten.unfold_backward.out,
        aten.upsample_bilinear2d.vec,
        aten.upsample_bilinear2d.default,
        aten.xlogy.Tensor,
        aten.xlogy.Scalar_Other,
        aten.xlogy.Scalar_Self,
        aten.xlogy.OutTensor,
        aten.xlogy.OutScalar_Self,
        aten.xlogy.OutScalar_Other,
        aten.zero.default,
        aten.zero.out,
        aten.zero_.default,
        aten.zeros.default,
        aten.zeros_like.default,
        aten.zeros_like.out,
    ]
)

# List of built-in pytorch framework decompositions we would like to use in HPU
# backend in training only.
hpu_backend_decompositions_training = get_decompositions(
    [
        # no entries for now
    ]
)

# List of built-in pytorch framework decompositions we would like to use in HPU
# backend in inference only.
hpu_backend_decompositions_inference = get_decompositions(
    [
        # no entries for now
    ]
)


# This function should be used to attach additional custom decompositions on top of builtin ones above.
def register_custom_decomposition(ops, decomposition_list):
    for op in [ops] if callable(ops) else ops:
        if op in decomposition_list:
            logger.warning(f"duplicate decomp: {ops}")
    logger.info(f"registering custom decomposition of: {ops}")
    return torch._decomp.register_decomposition(ops, decomposition_list)


@register_custom_decomposition(aten.full_like, hpu_backend_decompositions_common)
def full_like(
    a: utils.TensorLikeType,
    fill_value: utils.NumberType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
    memory_format: torch.memory_format = torch.preserve_format,
) -> utils.TensorLikeType:
    dtype = a.dtype if dtype is None else dtype
    layout = a.layout if layout is None else layout
    device = a.device if device is None else device

    return torch.full(
        a.shape,
        fill_value,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


@register_custom_decomposition(aten.bernoulli.p, hpu_backend_decompositions_common)
def bernoulli(input, p, *, generator=None):
    return torch.bernoulli(torch.full_like(input, p), generator=generator)


@register_custom_decomposition(aten.sort, hpu_backend_decompositions_common)
def sort(
    a: utils.Tensor,
    dim: int = -1,
    descending: bool = False,
) -> utils.Tuple[utils.Tensor, utils.Tensor]:
    k = a.size(dim)
    return torch.topk(a, k, dim, descending)


@register_custom_decomposition(
    torch.ops.aten.squeeze.dim, hpu_backend_decompositions_common
)
def squeeze(input, dim):
    return torch.squeeze(input, [dim])


@register_custom_decomposition(
    torch.ops.aten.squeeze.default, hpu_backend_decompositions_common
)
def squeeze(input):
    inp_size = len(input.size())
    dim_list = list(range(0, inp_size))
    return torch.squeeze(input, dim_list)


# Decomposition based on https://github.com/pytorch/pytorch/blob/v2.1.0/torch/_decomp/decompositions.py#L1055
# with bernoulli instead of rand_like
@register_custom_decomposition(
    aten.native_dropout.default, hpu_backend_decompositions_common
)
def native_dropout(input, p, train=None):
    if train and p != 0:
        if p == 1:
            return (torch.zeros_like(input), torch.zeros_like(input, dtype=torch.bool))
        p1m = 1.0 - p
        bool_mask = torch.ops.aten.bernoulli(torch.ops.aten.empty_like(input), p1m)
        res = bool_mask * input * float(1.0 / p1m)
        return (res, bool_mask)
    else:
        return (input, torch.ones_like(input, dtype=torch.bool))


def get_hpu_decompositions(is_training: bool):
    if configuration_flags["use_decompositions"]:
        if is_training:
            return {
                **hpu_backend_decompositions_common,
                **hpu_backend_decompositions_training,
            }
        else:
            return {
                **hpu_backend_decompositions_common,
                **hpu_backend_decompositions_inference,
            }
    else:
        return None
