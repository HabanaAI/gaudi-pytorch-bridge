// Autogenerated file by gen_op.py. Do not edit directly!
/******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include "pytorch_helpers/habana_helpers/pt_version_check.h"

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>
#include "hpu_ops/autocast_helpers.h"

namespace at {
namespace autocast {
namespace {

using tuple_2_tensors = std::tuple<Tensor, Tensor>;
using tuple_3_tensors = std::tuple<Tensor, Tensor, Tensor>;
using tuple_4_tensors = std::tuple<Tensor, Tensor, Tensor, Tensor>;
using tuple_5_tensors = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>;
using tuple_6_tensors = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;
using tuple_4_tensors_int64 = std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t>;
using tuple_2_tensors_double_int64 = std::tuple<Tensor,Tensor,double,int64_t>;
using tuple_3_tensors_vector = std::tuple<Tensor,Tensor,Tensor,::std::vector<Tensor>>;
using tuple_double_int64 = std::tuple<double,int64_t>;
using tuple_tensor_vector = std::tuple<Tensor,::std::vector<Tensor>>;
using tuple_vector_tensor = std::tuple<::std::vector<Tensor>,Tensor>;
using tuple_tensor_2_vectors = std::tuple<Tensor,::std::vector<Tensor>,::std::vector<Tensor>>;
using tuple_4_tensors_2_int64_2_tensors = std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,at::Tensor,at::Tensor>;
using tuple_4_tensors_4_int64_tensor = std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t,int64_t,int64_t,int64_t,Tensor>;
using tuple_2_tensors_2_int64_tensor = std::tuple<Tensor,Tensor,int64_t,int64_t,Tensor>;
using tuple_4_tensors_2_int64_3_tensor = std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t,int64_t,Tensor,Tensor,Tensor>;
using tuple_4_tensors_2_int64 = std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t,int64_t>;
using tuple_3_vectors = std::tuple<::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>>;
using tuple_5_vectors = std::tuple<::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>>;
#if IS_PYTORCH_AT_LEAST(2, 4)
using tuple_4_vectors = std::tuple<::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>>;
#endif

TORCH_LIBRARY_IMPL(aten, AutocastHPU, m) {
  Hpu_KERNEL(clamp_max, "clamp_max.Tensor", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(clamp_min, "clamp_min", at::Tensor(const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(clamp_min, "clamp_min.Tensor", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(clip, "clip", at::Tensor(const at::Tensor &, const c10::optional<at::Scalar> &, const c10::optional<at::Scalar> &))
  Hpu_KERNEL(clip, "clip.Tensor", at::Tensor(const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &))
  Hpu_KERNEL(complex, "complex", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(polar, "polar", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(constant_pad_nd, "constant_pad_nd", at::Tensor(const at::Tensor &, c10::IntArrayRef, const at::Scalar &))
  Hpu_KERNEL(convolution, "convolution", at::Tensor(const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, bool, c10::IntArrayRef, int64_t))
  Hpu_KERNEL(convolution_overrideable, "convolution_overrideable", at::Tensor(const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, bool, c10::IntArrayRef, int64_t))
  Hpu_KERNEL(_convolution, "_convolution", at::Tensor(const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, bool, c10::IntArrayRef, int64_t, bool, bool, bool, bool))
  Hpu_KERNEL(_convolution, "_convolution.deprecated", at::Tensor(const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, bool, at::IntArrayRef, int64_t, bool, bool, bool))
  Hpu_KERNEL(_convolution_mode, "_convolution_mode", at::Tensor(const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, c10::IntArrayRef, c10::string_view, c10::IntArrayRef, int64_t))
  Hpu_KERNEL(conv1d, "conv1d", at::Tensor(const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, int64_t))
  Hpu_KERNEL(conv2d, "conv2d", at::Tensor(const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, int64_t))
  Hpu_KERNEL(conv3d, "conv3d", at::Tensor(const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, int64_t))
  Hpu_KERNEL(conv1d, "conv1d.padding", at::Tensor(const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, c10::IntArrayRef, c10::string_view, c10::IntArrayRef, int64_t))
  Hpu_KERNEL(conv2d, "conv2d.padding", at::Tensor(const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, c10::IntArrayRef, c10::string_view, c10::IntArrayRef, int64_t))
  Hpu_KERNEL(conv3d, "conv3d.padding", at::Tensor(const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, c10::IntArrayRef, c10::string_view, c10::IntArrayRef, int64_t))
  Hpu_KERNEL(conv_tbc, "conv_tbc", at::Tensor(const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t))
  Hpu_KERNEL(conv_transpose1d, "conv_transpose1d", at::Tensor(const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, int64_t, c10::IntArrayRef))
  Hpu_KERNEL(conv_transpose2d, "conv_transpose2d.input", at::Tensor(const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, int64_t, c10::IntArrayRef))
  Hpu_KERNEL(conv_transpose3d, "conv_transpose3d.input", at::Tensor(const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, int64_t, c10::IntArrayRef))
  Hpu_KERNEL(copy, "copy", at::Tensor(const at::Tensor &, const at::Tensor &, bool))
  Hpu_KERNEL(_copy_from, "_copy_from", at::Tensor(const at::Tensor &, const at::Tensor &, bool))
  Hpu_KERNEL(_copy_from_and_resize, "_copy_from_and_resize", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(cos, "cos", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(cosh, "cosh", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(cosine_embedding_loss, "cosine_embedding_loss", at::Tensor(const at::Tensor &, const at::Tensor &, const at::Tensor &, double, int64_t))
  Hpu_KERNEL(count_nonzero, "count_nonzero.dim_IntList", at::Tensor(const at::Tensor &, at::IntArrayRef))
  Hpu_KERNEL(count_nonzero, "count_nonzero", at::Tensor(const at::Tensor &, c10::optional<int64_t>))
  Hpu_KERNEL(cov, "cov", at::Tensor(const at::Tensor &, int64_t, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &))
  Hpu_KERNEL(corrcoef, "corrcoef", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(_mps_convolution_transpose, "_mps_convolution_transpose", at::Tensor(const at::Tensor &, const at::Tensor &, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, int64_t))
  Hpu_KERNEL(cummax, "cummax", tuple_2_tensors(const at::Tensor &, int64_t))
  Hpu_KERNEL(cummax, "cummax.dimname", tuple_2_tensors(const at::Tensor &, at::Dimname))
  Hpu_KERNEL(cummin, "cummin", tuple_2_tensors(const at::Tensor &, int64_t))
  Hpu_KERNEL(cummin, "cummin.dimname", tuple_2_tensors(const at::Tensor &, at::Dimname))
  Hpu_KERNEL(cumprod, "cumprod", at::Tensor(const at::Tensor &, int64_t, c10::optional<at::ScalarType>))
  Hpu_KERNEL(cumprod, "cumprod.dimname", at::Tensor(const at::Tensor &, at::Dimname, c10::optional<at::ScalarType>))
  Hpu_KERNEL(cumsum, "cumsum", at::Tensor(const at::Tensor &, int64_t, c10::optional<at::ScalarType>))
  Hpu_KERNEL(cumsum, "cumsum.dimname", at::Tensor(const at::Tensor &, at::Dimname, c10::optional<at::ScalarType>))
  Hpu_KERNEL(cumulative_trapezoid, "cumulative_trapezoid.x", at::Tensor(const at::Tensor &, const at::Tensor &, int64_t))
  Hpu_KERNEL(cumulative_trapezoid, "cumulative_trapezoid.dx", at::Tensor(const at::Tensor &, const at::Scalar &, int64_t))
  Hpu_KERNEL(ctc_loss, "ctc_loss.IntList", at::Tensor(const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, int64_t, int64_t, bool))
  Hpu_KERNEL(ctc_loss, "ctc_loss.Tensor", at::Tensor(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, int64_t, bool))
  Hpu_KERNEL(_ctc_loss, "_ctc_loss", tuple_2_tensors(const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, int64_t, bool))
  Hpu_KERNEL(_ctc_loss, "_ctc_loss.Tensor", tuple_2_tensors(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, bool))
  Hpu_KERNEL(diag_embed, "diag_embed", at::Tensor(const at::Tensor &, int64_t, int64_t, int64_t))
  Hpu_KERNEL(diagflat, "diagflat", at::Tensor(const at::Tensor &, int64_t))
  Hpu_KERNEL(diagonal, "diagonal", at::Tensor(const at::Tensor &, int64_t, int64_t, int64_t))
  Hpu_KERNEL(linalg_diagonal, "linalg_diagonal", at::Tensor(const at::Tensor &, int64_t, int64_t, int64_t))
  Hpu_KERNEL(diagonal, "diagonal.Dimname", at::Tensor(const at::Tensor &, at::Dimname, at::Dimname, at::Dimname, int64_t))
  Hpu_KERNEL(diff, "diff", at::Tensor(const at::Tensor &, int64_t, int64_t, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &))
  Hpu_KERNEL(gradient, "gradient.scalarint", ::std::vector<at::Tensor>(const at::Tensor &, const c10::optional<at::Scalar> &, c10::optional<int64_t>, int64_t))
  Hpu_KERNEL(gradient, "gradient.scalararray", ::std::vector<at::Tensor>(const at::Tensor &, const at::Scalar &, at::IntArrayRef, int64_t))
  Hpu_KERNEL(gradient, "gradient.array", ::std::vector<at::Tensor>(const at::Tensor &, at::IntArrayRef, int64_t))
  Hpu_KERNEL(gradient, "gradient.scalarrayint", ::std::vector<at::Tensor>(const at::Tensor &, at::ArrayRef<at::Scalar>, c10::optional<int64_t>, int64_t))
  Hpu_KERNEL(gradient, "gradient.scalarrayarray", ::std::vector<at::Tensor>(const at::Tensor &, at::ArrayRef<at::Scalar>, at::IntArrayRef, int64_t))
  Hpu_KERNEL(gradient, "gradient.tensorarrayint", ::std::vector<at::Tensor>(const at::Tensor &, at::TensorList, c10::optional<int64_t>, int64_t))
  Hpu_KERNEL(gradient, "gradient.tensorarray", ::std::vector<at::Tensor>(const at::Tensor &, at::TensorList, at::IntArrayRef, int64_t))
  Hpu_KERNEL(div, "div.Tensor", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(div, "div.Tensor_mode", at::Tensor(const at::Tensor &, const at::Tensor &, c10::optional<c10::string_view>))
  Hpu_KERNEL(div, "div.Scalar", at::Tensor(const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(div, "div.Scalar_mode", at::Tensor(const at::Tensor &, const at::Scalar &, c10::optional<c10::string_view>))
  Hpu_KERNEL(divide, "divide.Tensor", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(divide, "divide.Scalar", at::Tensor(const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(divide, "divide.Tensor_mode", at::Tensor(const at::Tensor &, const at::Tensor &, c10::optional<c10::string_view>))
  Hpu_KERNEL(divide, "divide.Scalar_mode", at::Tensor(const at::Tensor &, const at::Scalar &, c10::optional<c10::string_view>))
  Hpu_KERNEL(true_divide, "true_divide.Tensor", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(true_divide, "true_divide.Scalar", at::Tensor(const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(dot, "dot", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(vdot, "vdot", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(einsum, "einsum", at::Tensor(c10::string_view, at::TensorList, at::OptionalIntArrayRef))
  Hpu_KERNEL(embedding, "embedding", at::Tensor(const at::Tensor &, const at::Tensor &, int64_t, bool, bool))
  Hpu_KERNEL(_embedding_bag_forward_only, "_embedding_bag_forward_only", tuple_4_tensors(const at::Tensor &, const at::Tensor &, const at::Tensor &, bool, int64_t, bool, const c10::optional<at::Tensor> &, bool, int64_t))
  Hpu_KERNEL(_rowwise_prune, "_rowwise_prune", tuple_2_tensors(const at::Tensor &, const at::Tensor &, at::ScalarType))
  Hpu_KERNEL(row_stack, "row_stack", at::Tensor(at::TensorList))
  Hpu_KERNEL(embedding_bag, "embedding_bag", tuple_4_tensors(const at::Tensor &, const at::Tensor &, const at::Tensor &, bool, int64_t, bool, const c10::optional<at::Tensor> &, bool))
  Hpu_KERNEL(embedding_bag, "embedding_bag.padding_idx", tuple_4_tensors(const at::Tensor &, const at::Tensor &, const at::Tensor &, bool, int64_t, bool, const c10::optional<at::Tensor> &, bool, c10::optional<int64_t>))
  Hpu_KERNEL(_embedding_bag, "_embedding_bag", tuple_4_tensors(const at::Tensor &, const at::Tensor &, const at::Tensor &, bool, int64_t, bool, const c10::optional<at::Tensor> &, bool, int64_t))
  Hpu_KERNEL(empty, "empty.names", at::Tensor(at::IntArrayRef, c10::optional<at::DimnameList>, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<at::MemoryFormat>))
  Hpu_KERNEL(empty, "empty.memory_format", at::Tensor(c10::IntArrayRef, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<at::MemoryFormat>))
  Hpu_KERNEL(empty_permuted, "empty_permuted", at::Tensor(c10::IntArrayRef, at::IntArrayRef, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(_empty_affine_quantized, "_empty_affine_quantized", at::Tensor(c10::IntArrayRef, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>, double, int64_t, c10::optional<at::MemoryFormat>))
  Hpu_KERNEL(_empty_per_channel_affine_quantized, "_empty_per_channel_affine_quantized", at::Tensor(c10::IntArrayRef, const at::Tensor &, const at::Tensor &, int64_t, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<at::MemoryFormat>))
  Hpu_KERNEL(empty_quantized, "empty_quantized", at::Tensor(at::IntArrayRef, const at::Tensor &, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<at::MemoryFormat>))
  Hpu_KERNEL(empty_like, "empty_like", at::Tensor(const at::Tensor &, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<at::MemoryFormat>))
  Hpu_KERNEL(empty_strided, "empty_strided", at::Tensor(c10::IntArrayRef, c10::IntArrayRef, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(erf, "erf", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(erfc, "erfc", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(exp, "exp", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(exp2, "exp2", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(expm1, "expm1", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(eye, "eye", at::Tensor(int64_t, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(eye, "eye.m", at::Tensor(int64_t, int64_t, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(flatten, "flatten.using_ints", at::Tensor(const at::Tensor &, int64_t, int64_t))
  Hpu_KERNEL(flatten, "flatten.named_out_dim", at::Tensor(const at::Tensor &, int64_t, int64_t, at::Dimname))
  Hpu_KERNEL(flatten, "flatten.using_names", at::Tensor(const at::Tensor &, at::Dimname, at::Dimname, at::Dimname))
  Hpu_KERNEL(flatten, "flatten.DimnameList", at::Tensor(const at::Tensor &, at::DimnameList, at::Dimname))
  Hpu_KERNEL(unflatten, "unflatten.int", at::Tensor(const at::Tensor &, int64_t, c10::IntArrayRef))
  Hpu_KERNEL(unflatten, "unflatten.Dimname", at::Tensor(const at::Tensor &, at::Dimname, c10::IntArrayRef, at::DimnameList))
  Hpu_KERNEL(fill, "fill.Scalar", at::Tensor(const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(fill, "fill.Tensor", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(floor, "floor", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(floor_divide, "floor_divide", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(floor_divide, "floor_divide.Scalar", at::Tensor(const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(frac, "frac", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(full, "full.names", at::Tensor(at::IntArrayRef, const at::Scalar &, c10::optional<at::DimnameList>, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(full, "full", at::Tensor(c10::IntArrayRef, const at::Scalar &, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(full_like, "full_like", at::Tensor(const at::Tensor &, const at::Scalar &, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<at::MemoryFormat>))
  Hpu_KERNEL(from_file, "from_file", at::Tensor(c10::string_view, c10::optional<bool>, c10::optional<int64_t>, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(gcd, "gcd", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(lcm, "lcm", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(grid_sampler, "grid_sampler", at::Tensor(const at::Tensor &, const at::Tensor &, int64_t, int64_t, bool))
  Hpu_KERNEL(grid_sampler_2d, "grid_sampler_2d", at::Tensor(const at::Tensor &, const at::Tensor &, int64_t, int64_t, bool))
  Hpu_KERNEL(_grid_sampler_2d_cpu_fallback, "_grid_sampler_2d_cpu_fallback", at::Tensor(const at::Tensor &, const at::Tensor &, int64_t, int64_t, bool))
  Hpu_KERNEL(grid_sampler_3d, "grid_sampler_3d", at::Tensor(const at::Tensor &, const at::Tensor &, int64_t, int64_t, bool))
  Hpu_KERNEL(hann_window, "hann_window", at::Tensor(int64_t, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(hann_window, "hann_window.periodic", at::Tensor(int64_t, bool, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(hamming_window, "hamming_window", at::Tensor(int64_t, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(hamming_window, "hamming_window.periodic", at::Tensor(int64_t, bool, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(hamming_window, "hamming_window.periodic_alpha", at::Tensor(int64_t, bool, double, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(hamming_window, "hamming_window.periodic_alpha_beta", at::Tensor(int64_t, bool, double, double, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(kaiser_window, "kaiser_window", at::Tensor(int64_t, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(kaiser_window, "kaiser_window.periodic", at::Tensor(int64_t, bool, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(kaiser_window, "kaiser_window.beta", at::Tensor(int64_t, bool, double, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(hinge_embedding_loss, "hinge_embedding_loss", at::Tensor(const at::Tensor &, const at::Tensor &, double, int64_t))
  Hpu_KERNEL(group_norm, "group_norm", at::Tensor(const at::Tensor &, int64_t, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, double, bool))
  Hpu_KERNEL(native_group_norm, "native_group_norm", tuple_3_tensors(const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, int64_t, int64_t, int64_t, int64_t, double))
  Hpu_KERNEL(_fft_r2c, "_fft_r2c", at::Tensor(const at::Tensor &, at::IntArrayRef, int64_t, bool))
  Hpu_KERNEL(_fft_c2r, "_fft_c2r", at::Tensor(const at::Tensor &, at::IntArrayRef, int64_t, int64_t))
  Hpu_KERNEL(_fft_c2c, "_fft_c2c", at::Tensor(const at::Tensor &, c10::IntArrayRef, int64_t, bool))
  Hpu_KERNEL(_validate_compressed_sparse_indices, "_validate_compressed_sparse_indices", void(bool, const at::Tensor &, const at::Tensor &, int64_t, int64_t, int64_t))
  Hpu_KERNEL(_cufft_get_plan_cache_size, "_cufft_get_plan_cache_size", int64_t(DeviceIndex))
  Hpu_KERNEL(_cufft_get_plan_cache_max_size, "_cufft_get_plan_cache_max_size", int64_t(DeviceIndex))
}

} // namespace
} // namespace autocast
} // namespace at

