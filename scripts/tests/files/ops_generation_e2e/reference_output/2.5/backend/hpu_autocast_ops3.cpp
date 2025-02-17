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
  Hpu_KERNEL(batch_norm_stats, "batch_norm_stats", tuple_2_tensors(const at::Tensor &, double))
  Hpu_KERNEL(batch_norm_elemt, "batch_norm_elemt", at::Tensor(const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const at::Tensor &, const at::Tensor &, double))
  Hpu_KERNEL(batch_norm_gather_stats, "batch_norm_gather_stats", tuple_2_tensors(const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, double, double, int64_t))
  Hpu_KERNEL(batch_norm_gather_stats_with_counts, "batch_norm_gather_stats_with_counts", tuple_2_tensors(const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, double, double, const at::Tensor &))
  Hpu_KERNEL(batch_norm_update_stats, "batch_norm_update_stats", tuple_2_tensors(const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, double))
  Hpu_KERNEL(_nnpack_spatial_convolution, "_nnpack_spatial_convolution", at::Tensor(const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, c10::IntArrayRef, c10::IntArrayRef))
  Hpu_KERNEL(ones, "ones.names", at::Tensor(at::IntArrayRef, c10::optional<at::DimnameList>, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(ones, "ones", at::Tensor(c10::IntArrayRef, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(ones_like, "ones_like", at::Tensor(const at::Tensor &, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<at::MemoryFormat>))
  Hpu_KERNEL(pairwise_distance, "pairwise_distance", at::Tensor(const at::Tensor &, const at::Tensor &, double, double, bool))
  Hpu_KERNEL(cdist, "cdist", at::Tensor(const at::Tensor &, const at::Tensor &, double, c10::optional<int64_t>))
  Hpu_KERNEL(_euclidean_dist, "_euclidean_dist", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(_cdist_forward, "_cdist_forward", at::Tensor(const at::Tensor &, const at::Tensor &, double, c10::optional<int64_t>))
  Hpu_KERNEL(pdist, "pdist", at::Tensor(const at::Tensor &, double))
  Hpu_KERNEL(_pdist_forward, "_pdist_forward", at::Tensor(const at::Tensor &, double))
  Hpu_KERNEL(cosine_similarity, "cosine_similarity", at::Tensor(const at::Tensor &, const at::Tensor &, int64_t, double))
  Hpu_KERNEL(permute, "permute", at::Tensor(const at::Tensor &, at::IntArrayRef))
  Hpu_KERNEL(movedim, "movedim.intlist", at::Tensor(const at::Tensor &, at::IntArrayRef, at::IntArrayRef))
  Hpu_KERNEL(movedim, "movedim.int", at::Tensor(const at::Tensor &, int64_t, int64_t))
  Hpu_KERNEL(moveaxis, "moveaxis.intlist", at::Tensor(const at::Tensor &, at::IntArrayRef, at::IntArrayRef))
  Hpu_KERNEL(moveaxis, "moveaxis.int", at::Tensor(const at::Tensor &, int64_t, int64_t))
  Hpu_KERNEL(adjoint, "adjoint", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(pixel_shuffle, "pixel_shuffle", at::Tensor(const at::Tensor &, int64_t))
  Hpu_KERNEL(pixel_unshuffle, "pixel_unshuffle", at::Tensor(const at::Tensor &, int64_t))
  Hpu_KERNEL(channel_shuffle, "channel_shuffle", at::Tensor(const at::Tensor &, int64_t))
  Hpu_KERNEL(native_channel_shuffle, "native_channel_shuffle", at::Tensor(const at::Tensor &, int64_t))
  Hpu_KERNEL(_pin_memory, "_pin_memory", at::Tensor(const at::Tensor &, c10::optional<c10::Device>))
  Hpu_KERNEL(pinverse, "pinverse", at::Tensor(const at::Tensor &, double))
  Hpu_KERNEL(poisson_nll_loss, "poisson_nll_loss", at::Tensor(const at::Tensor &, const at::Tensor &, bool, bool, double, int64_t))
  Hpu_KERNEL(rad2deg, "rad2deg", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(deg2rad, "deg2rad", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(scalar_tensor, "scalar_tensor", at::Tensor(const at::Scalar &, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(rand, "rand.names", at::Tensor(c10::IntArrayRef, c10::optional<at::DimnameList>, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(rand, "rand.generator_with_names", at::Tensor(c10::IntArrayRef, c10::optional<at::Generator>, c10::optional<at::DimnameList>, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(rand, "rand", at::Tensor(c10::IntArrayRef, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(rand, "rand.generator", at::Tensor(c10::IntArrayRef, c10::optional<at::Generator>, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(rand_like, "rand_like", at::Tensor(const at::Tensor &, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<at::MemoryFormat>))
  Hpu_KERNEL(randint, "randint", at::Tensor(int64_t, c10::IntArrayRef, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(randint, "randint.generator", at::Tensor(int64_t, c10::IntArrayRef, c10::optional<at::Generator>, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(randint, "randint.low", at::Tensor(int64_t, int64_t, c10::IntArrayRef, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(randint, "randint.low_generator", at::Tensor(int64_t, int64_t, c10::IntArrayRef, c10::optional<at::Generator>, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(randint_like, "randint_like", at::Tensor(const at::Tensor &, int64_t, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<at::MemoryFormat>))
  Hpu_KERNEL(randint_like, "randint_like.low_dtype", at::Tensor(const at::Tensor &, int64_t, int64_t, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<at::MemoryFormat>))
  Hpu_KERNEL(randn, "randn", at::Tensor(c10::IntArrayRef, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(randn, "randn.generator", at::Tensor(c10::IntArrayRef, c10::optional<at::Generator>, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(randn, "randn.names", at::Tensor(c10::IntArrayRef, c10::optional<at::DimnameList>, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(randn, "randn.generator_with_names", at::Tensor(c10::IntArrayRef, c10::optional<at::Generator>, c10::optional<at::DimnameList>, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(randn_like, "randn_like", at::Tensor(const at::Tensor &, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<at::MemoryFormat>))
  Hpu_KERNEL(randperm, "randperm", at::Tensor(int64_t, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(randperm, "randperm.generator", at::Tensor(int64_t, c10::optional<at::Generator>, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(range, "range.step", at::Tensor(const at::Scalar &, const at::Scalar &, const at::Scalar &, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(range, "range", at::Tensor(const at::Scalar &, const at::Scalar &, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<c10::Device>, c10::optional<bool>))
  Hpu_KERNEL(ravel, "ravel", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(reciprocal, "reciprocal", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(neg, "neg", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(negative, "negative", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(repeat_interleave, "repeat_interleave.Tensor", at::Tensor(const at::Tensor &, c10::optional<int64_t>))
  Hpu_KERNEL(repeat_interleave, "repeat_interleave.self_Tensor", at::Tensor(const at::Tensor &, const at::Tensor &, c10::optional<int64_t>, c10::optional<int64_t>))
  Hpu_KERNEL(repeat_interleave, "repeat_interleave.self_int", at::Tensor(const at::Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>))
  Hpu_KERNEL(reshape, "reshape", at::Tensor(const at::Tensor &, c10::IntArrayRef))
  Hpu_KERNEL(_reshape_copy, "_reshape_copy", at::Tensor(const at::Tensor &, c10::IntArrayRef))
  Hpu_KERNEL(_reshape_alias, "_reshape_alias", at::Tensor(const at::Tensor &, c10::IntArrayRef, c10::IntArrayRef))
  Hpu_KERNEL(_mkldnn_reshape, "_mkldnn_reshape", at::Tensor(const at::Tensor &, at::IntArrayRef))
  Hpu_KERNEL(round, "round", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(round, "round.decimals", at::Tensor(const at::Tensor &, int64_t))
  Hpu_KERNEL(rrelu, "rrelu", at::Tensor(const at::Tensor &, const at::Scalar &, const at::Scalar &, bool, c10::optional<at::Generator>))
  Hpu_KERNEL(relu, "relu", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(relu6, "relu6", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(prelu, "prelu", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(_prelu_kernel, "_prelu_kernel", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(gelu, "gelu", at::Tensor(const at::Tensor &, c10::string_view))
  Hpu_KERNEL(hardshrink, "hardshrink", at::Tensor(const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(rsqrt, "rsqrt", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(select, "select.Dimname", at::Tensor(const at::Tensor &, at::Dimname, int64_t))
  Hpu_KERNEL(select, "select.int", at::Tensor(const at::Tensor &, int64_t, int64_t))
  Hpu_KERNEL(selu, "selu", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(celu, "celu", at::Tensor(const at::Tensor &, const at::Scalar &))
  Hpu_KERNEL(silu, "silu", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(mish, "mish", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(sigmoid, "sigmoid", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(logit, "logit", at::Tensor(const at::Tensor &, c10::optional<double>))
  Hpu_KERNEL(sin, "sin", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(sinc, "sinc", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(sinh, "sinh", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(detach, "detach", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(size, "size.int", int64_t(const at::Tensor &, int64_t))
  Hpu_KERNEL(size, "size.Dimname", int64_t(const at::Tensor &, at::Dimname))
  Hpu_KERNEL(slice, "slice.Tensor", at::Tensor(const at::Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, int64_t))
  Hpu_KERNEL(slice_scatter, "slice_scatter", at::Tensor(const at::Tensor &, const at::Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, int64_t))
  Hpu_KERNEL(select_scatter, "select_scatter", at::Tensor(const at::Tensor &, const at::Tensor &, int64_t, int64_t))
  Hpu_KERNEL(diagonal_scatter, "diagonal_scatter", at::Tensor(const at::Tensor &, const at::Tensor &, int64_t, int64_t, int64_t))
  Hpu_KERNEL(as_strided_scatter, "as_strided_scatter", at::Tensor(const at::Tensor &, const at::Tensor &, c10::IntArrayRef, c10::IntArrayRef, c10::optional<int64_t>))
  Hpu_KERNEL(smm, "smm", at::Tensor(const at::Tensor &, const at::Tensor &))
  Hpu_KERNEL(softmax, "softmax.int", at::Tensor(const at::Tensor &, int64_t, c10::optional<at::ScalarType>))
  Hpu_KERNEL(softmax, "softmax.Dimname", at::Tensor(const at::Tensor &, at::Dimname, c10::optional<at::ScalarType>))
  Hpu_KERNEL(_softmax, "_softmax", at::Tensor(const at::Tensor &, int64_t, bool))
  Hpu_KERNEL(unsafe_split, "unsafe_split.Tensor", ::std::vector<at::Tensor>(const at::Tensor &, int64_t, int64_t))
  Hpu_KERNEL(split, "split.Tensor", ::std::vector<at::Tensor>(const at::Tensor &, int64_t, int64_t))
  Hpu_KERNEL(split, "split.sizes", ::std::vector<at::Tensor>(const at::Tensor &, c10::IntArrayRef, int64_t))
  Hpu_KERNEL(unsafe_split_with_sizes, "unsafe_split_with_sizes", ::std::vector<at::Tensor>(const at::Tensor &, c10::IntArrayRef, int64_t))
  Hpu_KERNEL(split_with_sizes, "split_with_sizes", ::std::vector<at::Tensor>(const at::Tensor &, c10::IntArrayRef, int64_t))
  Hpu_KERNEL(hsplit, "hsplit.int", ::std::vector<at::Tensor>(const at::Tensor &, int64_t))
  Hpu_KERNEL(hsplit, "hsplit.array", ::std::vector<at::Tensor>(const at::Tensor &, at::IntArrayRef))
  Hpu_KERNEL(vsplit, "vsplit.int", ::std::vector<at::Tensor>(const at::Tensor &, int64_t))
  Hpu_KERNEL(vsplit, "vsplit.array", ::std::vector<at::Tensor>(const at::Tensor &, at::IntArrayRef))
  Hpu_KERNEL(dsplit, "dsplit.int", ::std::vector<at::Tensor>(const at::Tensor &, int64_t))
  Hpu_KERNEL(dsplit, "dsplit.array", ::std::vector<at::Tensor>(const at::Tensor &, at::IntArrayRef))
  Hpu_KERNEL(squeeze, "squeeze", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(squeeze, "squeeze.dim", at::Tensor(const at::Tensor &, int64_t))
  Hpu_KERNEL(squeeze, "squeeze.dimname", at::Tensor(const at::Tensor &, at::Dimname))
  Hpu_KERNEL(squeeze, "squeeze.dims", at::Tensor(const at::Tensor &, at::IntArrayRef))
  Hpu_KERNEL(sspaddmm, "sspaddmm", at::Tensor(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, const at::Scalar &))
  Hpu_KERNEL(stack, "stack", at::Tensor(at::TensorList, int64_t))
  Hpu_KERNEL(_stack, "_stack", at::Tensor(at::TensorList, int64_t))
  Hpu_KERNEL(hstack, "hstack", at::Tensor(at::TensorList))
  Hpu_KERNEL(vstack, "vstack", at::Tensor(at::TensorList))
  Hpu_KERNEL(dstack, "dstack", at::Tensor(at::TensorList))
  Hpu_KERNEL(stft, "stft", at::Tensor(const at::Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const c10::optional<at::Tensor> &, bool, c10::optional<bool>, c10::optional<bool>))
  Hpu_KERNEL(stft, "stft.center", at::Tensor(const at::Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const c10::optional<at::Tensor> &, bool, c10::string_view, bool, c10::optional<bool>, c10::optional<bool>))
  Hpu_KERNEL(istft, "istft", at::Tensor(const at::Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const c10::optional<at::Tensor> &, bool, bool, c10::optional<bool>, c10::optional<int64_t>, bool))
  Hpu_KERNEL(stride, "stride.int", int64_t(const at::Tensor &, int64_t))
  Hpu_KERNEL(stride, "stride.Dimname", int64_t(const at::Tensor &, at::Dimname))
  Hpu_KERNEL(sum, "sum", at::Tensor(const at::Tensor &, c10::optional<at::ScalarType>))
  Hpu_KERNEL(sum, "sum.dim_IntList", at::Tensor(const at::Tensor &, at::OptionalIntArrayRef, bool, c10::optional<at::ScalarType>))
  Hpu_KERNEL(sum, "sum.dim_DimnameList", at::Tensor(const at::Tensor &, at::DimnameList, bool, c10::optional<at::ScalarType>))
  Hpu_KERNEL(nansum, "nansum", at::Tensor(const at::Tensor &, at::OptionalIntArrayRef, bool, c10::optional<at::ScalarType>))
  Hpu_KERNEL(sqrt, "sqrt", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(square, "square", at::Tensor(const at::Tensor &))
  Hpu_KERNEL(std, "std", at::Tensor(const at::Tensor &, bool))
  Hpu_KERNEL(std, "std.dim", at::Tensor(const at::Tensor &, at::OptionalIntArrayRef, bool, bool))
  Hpu_KERNEL(std, "std.correction", at::Tensor(const at::Tensor &, at::OptionalIntArrayRef, const c10::optional<at::Scalar> &, bool))
  Hpu_KERNEL(std_mean, "std_mean", tuple_2_tensors(const at::Tensor &, bool))
  Hpu_KERNEL(std_mean, "std_mean.dim", tuple_2_tensors(const at::Tensor &, at::OptionalIntArrayRef, bool, bool))
  Hpu_KERNEL(std_mean, "std_mean.correction", tuple_2_tensors(const at::Tensor &, at::OptionalIntArrayRef, const c10::optional<at::Scalar> &, bool))
  Hpu_KERNEL(std_mean, "std_mean.names_dim", tuple_2_tensors(const at::Tensor &, at::DimnameList, bool, bool))
  Hpu_KERNEL(std_mean, "std_mean.correction_names", tuple_2_tensors(const at::Tensor &, at::DimnameList, const c10::optional<at::Scalar> &, bool))
}

} // namespace
} // namespace autocast
} // namespace at

