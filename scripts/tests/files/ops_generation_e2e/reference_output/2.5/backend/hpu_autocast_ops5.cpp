// Autogenerated file by gen_op.py. Do not edit directly!
/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
using tuple_4_vectors = std::tuple<::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>>;

TORCH_LIBRARY_IMPL(aten, AutocastHPU, m) {
  Hpu_KERNEL(_validate_sparse_bsc_tensor_args, "_validate_sparse_bsc_tensor_args", void(const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef))
  Hpu_KERNEL(_sparse_coo_tensor_with_dims, "_sparse_coo_tensor_with_dims", at::Tensor(int64_t, int64_t, at::IntArrayRef, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(_sparse_coo_tensor_with_dims_and_tensors, "_sparse_coo_tensor_with_dims_and_tensors", at::Tensor(int64_t, int64_t, c10::IntArrayRef, const at::Tensor &, const at::Tensor &, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<bool>))
  Hpu_KERNEL(_to_cpu, "_to_cpu", ::std::vector<at::Tensor>(at::TensorList))
  Hpu_KERNEL(_coalesce, "_coalesce", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(hspmm, "hspmm", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(unbind, "unbind.int", ::std::vector<at::Tensor>(const at::Tensor &, int64_t))
  Hpu_KERNEL(unbind, "unbind.Dimname", ::std::vector<at::Tensor>(const at::Tensor &, at::Dimname))
  Hpu_KERNEL(_to_sparse_semi_structured, "_to_sparse_semi_structured", tuple_2_tensors(const at::Tensor &))
  Hpu_KERNEL(mkldnn_reorder_conv2d_weight, "mkldnn_reorder_conv2d_weight", at::Tensor(const at::Tensor &, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, int64_t, at::OptionalIntArrayRef))
  Hpu_KERNEL(mkldnn_reorder_conv3d_weight, "mkldnn_reorder_conv3d_weight", at::Tensor(const at::Tensor &, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, int64_t))
  Hpu_KERNEL(quantize_per_tensor_dynamic, "quantize_per_tensor_dynamic", at::Tensor(const at::Tensor &, at::ScalarType, bool))
  Hpu_KERNEL(quantize_per_tensor, "quantize_per_tensor", at::Tensor(const at::Tensor &, double, int64_t, at::ScalarType))
  Hpu_KERNEL(quantize_per_tensor, "quantize_per_tensor.tensor_qparams", at::Tensor(const at::Tensor &, const at::Tensor &, const at::Tensor &, at::ScalarType))
  Hpu_KERNEL(quantize_per_tensor, "quantize_per_tensor.tensors", ::std::vector<at::Tensor>(at::TensorList, const at::Tensor &, const at::Tensor &, at::ScalarType))
  Hpu_KERNEL(quantize_per_channel, "quantize_per_channel", at::Tensor(const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, at::ScalarType))
  Hpu_KERNEL(dequantize, "dequantize.self", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(dequantize, "dequantize.tensors", ::std::vector<at::Tensor>(at::TensorList))
  Hpu_KERNEL(q_scale, "q_scale", double(const at::Tensor &))
  Hpu_KERNEL(q_zero_point, "q_zero_point", int64_t(const at::Tensor &))
  Hpu_KERNEL(q_per_channel_scales, "q_per_channel_scales", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(q_per_channel_zero_points, "q_per_channel_zero_points", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(q_per_channel_axis, "q_per_channel_axis", int64_t(const at::Tensor &))
  Hpu_KERNEL(int_repr, "int_repr", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(_make_per_tensor_quantized_tensor, "_make_per_tensor_quantized_tensor", at::Tensor(const at::Tensor &, double, int64_t))
  Hpu_KERNEL(_make_per_channel_quantized_tensor, "_make_per_channel_quantized_tensor", at::Tensor(const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t))
  Hpu_KERNEL(fake_quantize_per_tensor_affine, "fake_quantize_per_tensor_affine", at::Tensor(const at::Tensor &, double, int64_t, int64_t, int64_t))
  Hpu_KERNEL(fake_quantize_per_tensor_affine, "fake_quantize_per_tensor_affine.tensor_qparams", at::Tensor(const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, int64_t))
  Hpu_KERNEL(fake_quantize_per_tensor_affine_cachemask, "fake_quantize_per_tensor_affine_cachemask", tuple_2_tensors(const at::Tensor &, double, int64_t, int64_t, int64_t))
  Hpu_KERNEL(_fake_quantize_per_tensor_affine_cachemask_tensor_qparams, "_fake_quantize_per_tensor_affine_cachemask_tensor_qparams", tuple_2_tensors(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, int64_t))
  Hpu_KERNEL(_fake_quantize_learnable_per_tensor_affine, "_fake_quantize_learnable_per_tensor_affine", at::Tensor(const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, int64_t, double))
  Hpu_KERNEL(fake_quantize_per_channel_affine, "fake_quantize_per_channel_affine", at::Tensor(const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, int64_t, int64_t))
  Hpu_KERNEL(fake_quantize_per_channel_affine_cachemask, "fake_quantize_per_channel_affine_cachemask", tuple_2_tensors(const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, int64_t, int64_t))
  Hpu_KERNEL(_fake_quantize_learnable_per_channel_affine, "_fake_quantize_learnable_per_channel_affine", at::Tensor(const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, int64_t, int64_t, double))
  Hpu_KERNEL(_choose_qparams_per_tensor, "_choose_qparams_per_tensor", tuple_double_int64(const at::Tensor &, bool))
  Hpu_KERNEL(_saturate_weight_to_fp16, "_saturate_weight_to_fp16", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(choose_qparams_optimized, "choose_qparams_optimized", tuple_2_tensors(const at::Tensor &, int64_t, int64_t, double, int64_t))
  Hpu_KERNEL(_to_copy, "_to_copy", at::Tensor(const at::Tensor &, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>, bool, c10::optional<at::MemoryFormat>))
  Hpu_KERNEL(meshgrid, "meshgrid", ::std::vector<at::Tensor>(at::TensorList))
  Hpu_KERNEL(meshgrid, "meshgrid.indexing", ::std::vector<at::Tensor>(at::TensorList, c10::string_view))
  Hpu_KERNEL(cartesian_prod, "cartesian_prod", at::Tensor(at::TensorList))
  Hpu_KERNEL(combinations, "combinations", at::Tensor(const at::Tensor &, int64_t, bool))
  Hpu_KERNEL(result_type, "result_type.Tensor", at::ScalarType(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(result_type, "result_type.Scalar", at::ScalarType(const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(result_type, "result_type.Scalar_Tensor", at::ScalarType(const at::Scalar &, const at::Tensor &))
  Hpu_KERNEL(result_type, "result_type.Scalar_Scalar", at::ScalarType(const at::Scalar &, const at::Scalar &))
  Hpu_KERNEL(can_cast, "can_cast", bool(at::ScalarType, at::ScalarType))
  Hpu_KERNEL(promote_types, "promote_types", at::ScalarType(at::ScalarType, at::ScalarType))
  Hpu_KERNEL(_local_scalar_dense, "_local_scalar_dense", at::Scalar(const at::Tensor &))
  Hpu_KERNEL(_lstm_mps, "_lstm_mps", tuple_6_tensors(const at::Tensor &, at::TensorList, at::TensorList, bool, int64_t, double, bool, bool, bool))
  Hpu_KERNEL(_thnn_fused_lstm_cell, "_thnn_fused_lstm_cell", tuple_3_tensors(const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &))
  Hpu_KERNEL(_thnn_fused_gru_cell, "_thnn_fused_gru_cell", tuple_2_tensors(const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &))
  Hpu_KERNEL(lstm, "lstm.input", tuple_3_tensors(const at::Tensor &, at::TensorList, at::TensorList, bool, int64_t, double, bool, bool, bool))
  Hpu_KERNEL(lstm, "lstm.data", tuple_3_tensors(const at::Tensor &, const at::Tensor &, at::TensorList, at::TensorList, bool, int64_t, double, bool, bool))
  Hpu_KERNEL(gru, "gru.input", tuple_2_tensors(const at::Tensor &, const at::Tensor &, at::TensorList, bool, int64_t, double, bool, bool, bool))
  Hpu_KERNEL(gru, "gru.data", tuple_2_tensors(const at::Tensor &, const at::Tensor &, const at::Tensor &, at::TensorList, bool, int64_t, double, bool, bool))
  Hpu_KERNEL(rnn_tanh, "rnn_tanh.input", tuple_2_tensors(const at::Tensor &, const at::Tensor &, at::TensorList, bool, int64_t, double, bool, bool, bool))
  Hpu_KERNEL(rnn_tanh, "rnn_tanh.data", tuple_2_tensors(const at::Tensor &, const at::Tensor &, const at::Tensor &, at::TensorList, bool, int64_t, double, bool, bool))
  Hpu_KERNEL(rnn_relu, "rnn_relu.input", tuple_2_tensors(const at::Tensor &, const at::Tensor &, at::TensorList, bool, int64_t, double, bool, bool, bool))
  Hpu_KERNEL(rnn_relu, "rnn_relu.data", tuple_2_tensors(const at::Tensor &, const at::Tensor &, const at::Tensor &, at::TensorList, bool, int64_t, double, bool, bool))
  Hpu_KERNEL(lstm_cell, "lstm_cell", tuple_2_tensors(const at::Tensor &, at::TensorList, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &))
  Hpu_KERNEL(gru_cell, "gru_cell", at::Tensor(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &))
  Hpu_KERNEL(rnn_tanh_cell, "rnn_tanh_cell", at::Tensor(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &))
  Hpu_KERNEL(rnn_relu_cell, "rnn_relu_cell", at::Tensor(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &))
  Hpu_KERNEL(quantized_lstm_cell, "quantized_lstm_cell", tuple_2_tensors(const at::Tensor &, at::TensorList, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &))
  Hpu_KERNEL(quantized_gru_cell, "quantized_gru_cell", at::Tensor(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &))
  Hpu_KERNEL(quantized_rnn_relu_cell, "quantized_rnn_relu_cell", at::Tensor(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &))
  Hpu_KERNEL(quantized_rnn_tanh_cell, "quantized_rnn_tanh_cell", at::Tensor(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &))
  Hpu_KERNEL(_pack_padded_sequence, "_pack_padded_sequence", tuple_2_tensors(const at::Tensor &, const at::Tensor &, bool))
  Hpu_KERNEL(_pad_packed_sequence, "_pad_packed_sequence", tuple_2_tensors(const at::Tensor &, const at::Tensor &, bool, const at::Scalar &, int64_t))
  Hpu_KERNEL(lift, "lift", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(lift_fresh, "lift_fresh", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(lift_fresh_copy, "lift_fresh_copy", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(masked_fill, "masked_fill.Scalar", at::Tensor(const at::Tensor &, const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(masked_fill, "masked_fill.Tensor", at::Tensor(const at::Tensor &, const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(masked_scatter, "masked_scatter", at::Tensor(const at::Tensor &, const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(_masked_softmax, "_masked_softmax", at::Tensor(const at::Tensor &, const at::Tensor &, c10::optional<int64_t>, c10::optional<int64_t>))
  Hpu_KERNEL(put, "put", at::Tensor(const at::Tensor &, const at::Tensor &, const at::Tensor &, bool))
  Hpu_KERNEL(index_add, "index_add", at::Tensor(const at::Tensor &, int64_t, const at::Tensor &, const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(index_add, "index_add.dimname", at::Tensor(const at::Tensor &, at::Dimname, const at::Tensor &, const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(index_reduce, "index_reduce", at::Tensor(const at::Tensor &, int64_t, const at::Tensor &, const at::Tensor &, c10::string_view, bool))
  Hpu_KERNEL(index_fill, "index_fill.int_Scalar", at::Tensor(const at::Tensor &, int64_t, const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(index_fill, "index_fill.int_Tensor", at::Tensor(const at::Tensor &, int64_t, const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(index_fill, "index_fill.Dimname_Scalar", at::Tensor(const at::Tensor &, at::Dimname, const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(index_fill, "index_fill.Dimname_Tensor", at::Tensor(const at::Tensor &, at::Dimname, const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(scatter, "scatter.src", at::Tensor(const at::Tensor &, int64_t, const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(scatter, "scatter.value", at::Tensor(const at::Tensor &, int64_t, const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(scatter, "scatter.reduce", at::Tensor(const at::Tensor &, int64_t, const at::Tensor &, const at::Tensor &, c10::string_view))
  Hpu_KERNEL(scatter, "scatter.value_reduce", at::Tensor(const at::Tensor &, int64_t, const at::Tensor &, const at::Scalar &, c10::string_view))
  Hpu_KERNEL(scatter, "scatter.dimname_src", at::Tensor(const at::Tensor &, at::Dimname, const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(scatter, "scatter.dimname_value", at::Tensor(const at::Tensor &, at::Dimname, const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(scatter_add, "scatter_add", at::Tensor(const at::Tensor &, int64_t, const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(scatter_add, "scatter_add.dimname", at::Tensor(const at::Tensor &, at::Dimname, const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(scatter_reduce, "scatter_reduce.two", at::Tensor(const at::Tensor &, int64_t, const at::Tensor &, const at::Tensor &, c10::string_view, bool))
  Hpu_KERNEL(bitwise_and, "bitwise_and.Scalar", at::Tensor(const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(bitwise_and, "bitwise_and.Scalar_Tensor", at::Tensor(const at::Scalar &, const at::Tensor &))
  Hpu_KERNEL(bitwise_and, "bitwise_and.Tensor", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(bitwise_or, "bitwise_or.Scalar", at::Tensor(const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(bitwise_or, "bitwise_or.Scalar_Tensor", at::Tensor(const at::Scalar &, const at::Tensor &))
  Hpu_KERNEL(bitwise_or, "bitwise_or.Tensor", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(bitwise_xor, "bitwise_xor.Scalar", at::Tensor(const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(bitwise_xor, "bitwise_xor.Scalar_Tensor", at::Tensor(const at::Scalar &, const at::Tensor &))
  Hpu_KERNEL(bitwise_xor, "bitwise_xor.Tensor", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(bitwise_left_shift, "bitwise_left_shift.Tensor", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(bitwise_left_shift, "bitwise_left_shift.Tensor_Scalar", at::Tensor(const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(bitwise_left_shift, "bitwise_left_shift.Scalar_Tensor", at::Tensor(const at::Scalar &, const at::Tensor &))
  Hpu_KERNEL(bitwise_right_shift, "bitwise_right_shift.Tensor", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(bitwise_right_shift, "bitwise_right_shift.Tensor_Scalar", at::Tensor(const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(bitwise_right_shift, "bitwise_right_shift.Scalar_Tensor", at::Tensor(const at::Scalar &, const at::Tensor &))
  Hpu_KERNEL(addbmm, "addbmm", at::Tensor(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, const at::Scalar &))
  Hpu_KERNEL(diag, "diag", at::Tensor(const at::Tensor &, int64_t))
  Hpu_KERNEL(cross, "cross", at::Tensor(const at::Tensor &, const at::Tensor &, c10::optional<int64_t>))
  Hpu_KERNEL(triu, "triu", at::Tensor(const at::Tensor &, int64_t))
  Hpu_KERNEL(tril, "tril", at::Tensor(const at::Tensor &, int64_t))
  Hpu_KERNEL(tril_indices, "tril_indices", at::Tensor(int64_t, int64_t, int64_t, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(triu_indices, "triu_indices", at::Tensor(int64_t, int64_t, int64_t, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(trace, "trace", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(ne, "ne.Scalar", at::Tensor(const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(ne, "ne.Tensor", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(not_equal, "not_equal.Scalar", at::Tensor(const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(not_equal, "not_equal.Tensor", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(eq, "eq.Scalar", at::Tensor(const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(eq, "eq.Tensor", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(ge, "ge.Scalar", at::Tensor(const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(ge, "ge.Tensor", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(greater_equal, "greater_equal.Scalar", at::Tensor(const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(greater_equal, "greater_equal.Tensor", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(le, "le.Scalar", at::Tensor(const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(le, "le.Tensor", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(less_equal, "less_equal.Scalar", at::Tensor(const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(less_equal, "less_equal.Tensor", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(gt, "gt.Scalar", at::Tensor(const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(gt, "gt.Tensor", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(greater, "greater.Scalar", at::Tensor(const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(greater, "greater.Tensor", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(lt, "lt.Scalar", at::Tensor(const at::Tensor &, const at::Scalar &))
}

} // namespace
} // namespace autocast
} // namespace at

