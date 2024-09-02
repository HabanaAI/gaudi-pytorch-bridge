###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

from contextlib import contextmanager
from itertools import accumulate
from typing import Optional

import torch
import torch._prims_common as utils
from torch._decomp import core_aten_decompositions, get_decompositions
from torch._ops import DispatchKey

aten = torch.ops.aten

from habana_frameworks.torch.dynamo.compile_backend import config as hpu_backend_config

from .logger import get_compile_backend_logger

logger = get_compile_backend_logger()

# List of built-in pytorch framework decompositions we would like to use in HPU
# backend in both training and inference.
hpu_backend_decompositions_common = get_decompositions(
    [
        aten.addcdiv.default,
        aten.addcdiv_.default,
        aten.addcmul.default,
        aten.addcmul_.default,
        aten.addr.default,
        aten.all.default,
        aten.aminmax.default,
        aten.arange.default,
        aten.arange.start,
        aten.binary_cross_entropy.out,
        aten.binary_cross_entropy_backward.default,
        aten.binary_cross_entropy_backward.grad_input,
        aten.binary_cross_entropy_with_logits.default,
        aten.binary_cross_entropy_with_logits.out,
        aten.celu.default,
        aten.celu.out,
        aten.clamp_max.default,
        aten.clamp_min.default,
        aten.col2im.default,
        aten.col2im.out,
        aten.cudnn_batch_norm.default,
        aten.cudnn_batch_norm.out,
        aten.cudnn_batch_norm_backward.default,
        aten.cudnn_batch_norm_backward.out,
        aten.detach.default,
        aten.diag_embed.default,
        aten.diag_embed.out,
        aten.dot.default,
        aten.dot.out,
        aten.elu.default,
        aten.elu.out,
        aten.elu_backward.default,
        aten.elu_backward.grad_input,
        aten.embedding_dense_backward.out,
        aten.empty_like,
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
        aten.hardtanh_backward,
        aten.heaviside.out,
        aten.huber_loss.out,
        aten.huber_loss_backward.out,
        aten.im2col.default,
        aten.im2col.out,
        aten.index_add.out,
        aten.index_add.dimname,
        aten.index_add_.default,
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
        aten.log_sigmoid_forward.default,
        aten.log_sigmoid_backward.default,
        aten._log_softmax.out,
        aten._log_softmax_backward_data.out,
        aten.logsumexp.default,
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
        aten.ones.default,
        aten.ones_like.default,
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
        aten.std_mean,
        aten.t,
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


def override_instance_norm(dispatch_key):
    def internal(input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled):
        if input.device.type == "hpu":
            out, mean_tensor, istd_tensor = torch.ops.hpu.instance_norm.default(input, weight, bias, eps)
            return out
        else:
            new_impl = torch.ops.aten.instance_norm.default.py_kernels.pop(dispatch_key, None)
            torch.ops.aten.instance_norm.default._dispatch_cache.clear()
            # Call the original operation
            out = torch.ops.aten.instance_norm(
                input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled
            )
            # Restore the implementation
            if new_impl is not None:
                torch.ops.aten.instance_norm.default.py_impl(dispatch_key)(new_impl)
            return out

    return internal


@contextmanager
def override_composite_ops():
    ops = [
        (DispatchKey.CompositeImplicitAutograd, torch.ops.aten.instance_norm.default, override_instance_norm),
    ]

    old_tables = {}

    for dispatch_key, origin_impl, new_impl in ops:
        old_tables[(dispatch_key, origin_impl)] = origin_impl.py_kernels.copy()
        origin_impl.py_impl(dispatch_key)(new_impl(dispatch_key))

    try:
        yield
    finally:
        for (dispatch_key, origin_impl), old_table in old_tables.items():
            origin_impl.py_kernels.clear()
            origin_impl.py_kernels.update(old_table)
            origin_impl._dispatch_cache.clear()


# This function should be used to attach additional custom decompositions on top of builtin ones above.
def register_custom_decomposition(ops, decomposition_list):
    for op in [ops] if callable(ops) else ops:
        if op in decomposition_list:
            logger.warn(f"duplicate decomp: {ops}")
    logger.info(f"registering custom decomposition of: {ops}")
    return torch._decomp.register_decomposition(ops, decomposition_list)


def get_like_layout(tensor: torch.Tensor, memory_format: Optional[torch.memory_format]) -> torch.memory_format:
    if memory_format in (torch.preserve_format, None):
        return utils.suggest_memory_format(tensor)
    else:
        return memory_format


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


@register_custom_decomposition(aten.diagonal, hpu_backend_decompositions_common)
def diagonal(
    self: utils.TensorLikeType,
    offset: int = 0,
    dim1: int = 0,
    dim2: int = 1,
) -> utils.TensorLikeType:
    """
    Reference implementation of torch.diagonal
    """
    num_dims = self.dim()
    dim1 = utils.canonicalize_dim(idx=dim1, rank=num_dims)
    dim2 = utils.canonicalize_dim(idx=dim2, rank=num_dims)

    torch._check(dim1 != dim2, lambda: f"diagonal dimensions cannot be identical {dim1}, {dim2}")

    storage_offset = self.storage_offset()

    if offset >= 0:
        diag_size = max(min(self.size()[dim1], self.size()[dim2] - offset), 0)
    else:
        diag_size = max(min(self.size()[dim1] + offset, self.size()[dim2]), 0)

    if diag_size > 0:
        if offset >= 0:
            storage_offset += offset * self.stride()[dim2]
        else:
            storage_offset -= offset * self.stride()[dim1]

    sizes = [s for i, s in enumerate(self.size()) if i not in (dim1, dim2)]
    sizes.append(diag_size)

    strides = [s for i, s in enumerate(self.stride()) if i not in (dim1, dim2)]
    strides.append(self.stride()[dim1] + self.stride()[dim2])

    result = self.as_strided(size=sizes, stride=strides, storage_offset=storage_offset)

    return result


@register_custom_decomposition(aten.bernoulli.p, hpu_backend_decompositions_common)
def bernoulli(input, p, *, generator=None):
    p_like_input = torch.full(
        [],
        p,
        dtype=input.dtype,
        layout=input.layout,
        device=input.device,
    ).expand(input.shape)
    return torch.bernoulli(p_like_input, generator=generator)


@register_custom_decomposition(aten.randn.generator, hpu_backend_decompositions_common)
def randngen(
    size,
    generator=None,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: bool = False,
):
    mean = torch.full(
        [],
        0.0,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    ).expand(size)
    stddev = torch.full(
        [],
        1.0,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    ).expand(size)
    return torch.normal(mean, stddev, generator=generator)


@register_custom_decomposition(aten.sort.default, hpu_backend_decompositions_common)
def sort(
    a: utils.Tensor,
    dim: int = -1,
    descending: bool = False,
) -> utils.Tuple[utils.Tensor, utils.Tensor]:
    k = a.size(dim) if a.dim() > 0 else 1
    return torch.topk(a, k, dim, descending)


@register_custom_decomposition(torch.ops.aten.squeeze.dim, hpu_backend_decompositions_common)
def squeeze(input, dim):
    return torch.squeeze(input, [dim])


@register_custom_decomposition(torch.ops.aten.squeeze.default, hpu_backend_decompositions_common)
def squeeze(input):
    inp_size = len(input.size())
    dim_list = list(range(0, inp_size))
    return torch.squeeze(input, dim_list)


# Random op decompositions mainly based on pytorch/torch/_inductor/decomposition.py
# and pytorch/torch/_decomp/decompositions_for_rng.py


@register_custom_decomposition(torch.ops.aten.rand_like.default, hpu_backend_decompositions_common)
def rand_like(self, *, dtype=None, device=None, memory_format=None, **kwargs):
    return torch.rand(
        [*self.size()],
        dtype=dtype or self.dtype,
        device=device or self.device,
        **kwargs,
    ).to(memory_format=get_like_layout(self, memory_format))


@register_custom_decomposition(torch.ops.aten.randn_like.default, hpu_backend_decompositions_common)
def randn_like(self, *, dtype=None, device=None, memory_format=None, **kwargs):
    return torch.randn(
        [*self.size()],
        dtype=dtype or self.dtype,
        device=device or self.device,
        **kwargs,
    ).to(memory_format=get_like_layout(self, memory_format))


@register_custom_decomposition(aten.rand.generator, hpu_backend_decompositions_common)
def rand_generator(size, **kwargs):
    kwargs.pop("generator", None)
    return torch.rand(size, **kwargs)


# empty_permuted op decompositions based on pytorch/torch/_inductor/decomposition.py
@register_custom_decomposition(aten.empty_permuted.default, hpu_backend_decompositions_common)
def empty_permuted(size, physical_layout, **kwargs):
    perm = [0] * len(size)
    for p, l in enumerate(physical_layout):
        perm[l] = p
    return torch.empty([size[l] for l in physical_layout], **kwargs).permute(perm)


@register_custom_decomposition(torch.ops.hpu.sdpa_recomp_fwd.default, hpu_backend_decompositions_common)
def sdpa_recomp_fwd(
    q,
    k,
    v,
    attn_mask,
    dropout_p,
    scale,
    is_causal,
    requires_backward,
    fast_softmax_mode,
    valid_seq_len,
    seq_padding_type,
):
    op = torch.ops.hpu.sdpa_recomp_fwd_dropout if dropout_p > 0.0 else torch.ops.hpu.sdpa_recomp_fwd_non_dropout
    return op(
        q,
        k,
        v,
        attn_mask,
        dropout_p,
        scale,
        is_causal,
        requires_backward,
        fast_softmax_mode,
        valid_seq_len,
        seq_padding_type,
    )


@register_custom_decomposition(torch.ops.hpu.sdpa_fwd.default, hpu_backend_decompositions_common)
def sdpa_fwd(
    q,
    k,
    v,
    attn_mask,
    dropout_p,
    scale,
    is_causal,
    fast_softmax_mode,
    valid_seq_len,
    seq_padding_type,
):
    op = torch.ops.hpu.sdpa_fwd_dropout if dropout_p > 0.0 else torch.ops.hpu.sdpa_fwd_non_dropout
    return op(q, k, v, attn_mask, dropout_p, scale, is_causal, fast_softmax_mode, valid_seq_len, seq_padding_type)


@register_custom_decomposition(torch.ops.hpu.fp8_sdpa_fwd.default, hpu_backend_decompositions_common)
def fp8_sdpa_fwd(
    q,
    k,
    v,
    attn_mask,
    dropout_p,
    scale,
    is_causal,
    softmax_mode,
    d_scale_q,
    d_scale_k,
    d_scale_v,
    q_scale_s,
    q_scale_o,
    d_scale_s,
    is_amax_s,
    valid_seq_len,
    seq_padding_type,
):
    op = torch.ops.hpu.fp8_sdpa_fwd_dropout if dropout_p > 0.0 else torch.ops.hpu.fp8_sdpa_fwd_non_dropout
    return op(
        q,
        k,
        v,
        attn_mask,
        dropout_p,
        scale,
        is_causal,
        softmax_mode,
        d_scale_q,
        d_scale_k,
        d_scale_v,
        q_scale_s,
        q_scale_o,
        d_scale_s,
        is_amax_s,
        valid_seq_len,
        seq_padding_type,
    )


@register_custom_decomposition(torch.ops.hpu.fp8_sdpa_recomp_fwd.default, hpu_backend_decompositions_common)
def fp8_sdpa_recomp_fwd(
    q,
    k,
    v,
    attn_mask,
    dropout_p,
    scale,
    is_causal,
    requires_backward,
    softmax_mode,
    d_scale_q,
    d_scale_k,
    d_scale_v,
    q_scale_s,
    q_scale_o,
    d_scale_s,
    is_amax_s,
    is_amax_o,
    valid_seq_len,
    seq_padding_type,
):

    op = torch.ops.hpu.fp8_sdpa_recomp_fwd_dropout if dropout_p > 0.0 else torch.ops.hpu.fp8_sdpa_recomp_fwd_non_dropout
    return op(
        q,
        k,
        v,
        attn_mask,
        dropout_p,
        scale,
        is_causal,
        requires_backward,
        softmax_mode,
        d_scale_q,
        d_scale_k,
        d_scale_v,
        q_scale_s,
        q_scale_o,
        d_scale_s,
        is_amax_s,
        is_amax_o,
        valid_seq_len,
        seq_padding_type,
    )


@register_custom_decomposition(aten.randint_like.default, hpu_backend_decompositions_common)
def randint_like(self, high, *, dtype=None, device=None, memory_format=None, **kwargs):
    return aten.randint.low(
        0,
        high,
        [*self.size()],
        dtype=dtype or self.dtype,
        device=device or self.device,
        **kwargs,
    ).to(memory_format=get_like_layout(self, memory_format))


@register_custom_decomposition(aten.randint_like.low_dtype, hpu_backend_decompositions_common)
def randint_like_low(self, low, high, *, dtype=None, device=None, memory_format=None, **kwargs):
    return aten.randint.low(
        low,
        high,
        [*self.size()],
        dtype=dtype or self.dtype,
        device=device or self.device,
        **kwargs,
    ).to(memory_format=get_like_layout(self, memory_format))


@register_custom_decomposition(aten.randint.default, hpu_backend_decompositions_common)
def randint(high, size, **kwargs):
    return aten.randint.low(0, high, size, **kwargs)


@register_custom_decomposition(aten.randint.low_generator, hpu_backend_decompositions_common)
def randint_low_generator(*args, **kwargs):
    kwargs.pop("generator", None)
    return aten.randint.low(*args, **kwargs)


# pytorch decomposes aten.cdist op to at::_euclidean_dist in some cases
# For hpu we prefer to call _cdist_forward in all cases
@register_custom_decomposition(aten._euclidean_dist, hpu_backend_decompositions_common)
def euclidean_dist(x1, x2):
    return torch.ops.aten._cdist_forward(x1, x2, 2.0, 1)


@register_custom_decomposition([aten.split.Tensor, aten.split_with_sizes], hpu_backend_decompositions_common)
def split(self, split_size, dim=0):
    if dim < 0:
        dim += self.dim()
    assert dim < self.dim() and dim >= 0, " given dimension value is out of range"
    cur_size = self.size(dim)
    assert (
        type(split_size) == int or type(split_size) == list or type(split_size) == torch.SymInt
    ), "split_size_or_sections is not a int value or list"
    # create a new list based on split_size(int)
    if type(split_size) != list:
        split_size = [split_size] * (cur_size // split_size)
        if cur_size != sum(split_size):
            split_size.append(cur_size - sum(split_size))
    # create a new list based on split list for calculating start and end indices
    new_split = [0] + split_size
    split_len = len(new_split)
    result = [None] * (split_len - 1)

    # accumulate the list that will help us to fetch start and end index
    new_split = list(accumulate(new_split))

    for idx in range(1, split_len):
        # workaround for: https://jira.habana-labs.com/browse/SW-162350
        # Slice op is not yet supported for dynamic shape in torch compile
        result[idx - 1] = aten.slice(self, dim, new_split[idx - 1], new_split[idx], 1)
    return tuple(result)


def get_hpu_decompositions():
    if hpu_backend_config.decomposition_mode == "habana":
        return {
            **hpu_backend_decompositions_common,
        }
    elif hpu_backend_config.decomposition_mode == "core_aten":
        return {**core_aten_decompositions()}
    else:
        return None
