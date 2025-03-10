/**
 * Copyright (c) 2025 Intel Corporation
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

// This file contains selected native_functions that can be registered to
// and the schema string that they should be registered with

at::Tensor & __ilshift__(at::Tensor & self, const at::Scalar & other); // {"schema": "aten::__ilshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)", "dispatch": "True", "default": "False"}
void _foreach_add_(at::TensorList self, const at::Scalar & scalar); // {"schema": "aten::_foreach_add_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()", "dispatch": "True", "default": "True"}
::std::tuple<at::Tensor,at::Tensor> _fused_dropout(const at::Tensor & self, double p, ::std::optional<at::Generator> generator); // {"schema": "aten::_fused_dropout(Tensor self, float p, Generator? generator=None) -> (Tensor, Tensor)", "dispatch": "True", "default": "False"}
::std::tuple<at::Tensor,at::Tensor> native_dropout(const at::Tensor & input, double p, ::std::optional<bool> train); // {"schema": "aten::native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)", "dispatch": "True", "default": "False"}
at::Tensor as_strided(const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, ::std::optional<c10::SymInt> storage_offset); // {"schema": "aten::as_strided(Tensor(a) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a)", "dispatch": "True", "default": "False"}
at::Tensor addbmm(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha); // {"schema": "aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor", "dispatch": "True", "default": "False"}
at::Tensor bucketize(const at::Scalar & self, const at::Tensor & boundaries, bool out_int32, bool right); // {"schema": "aten::bucketize.Scalar(Scalar self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor", "dispatch": "True", "default": "False"}
at::Tensor elu(const at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale); // {"schema": "aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor", "dispatch": "True", "default": "True"}
at::Tensor & prod_out(const at::Tensor & self, int64_t dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out); // {"schema": "aten::prod.int_out(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
at::Tensor clone(const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format); // {"schema": "aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor", "dispatch": "True", "default": "True"}
at::Tensor & mul_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out); // {"schema": "aten::mul.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "True"}
::std::tuple<at::Tensor &,at::Tensor &> sort_out(const at::Tensor & self, ::std::optional<bool> stable, int64_t dim, bool descending, at::Tensor & values, at::Tensor & indices); // {"schema": "aten::sort.values_stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)", "dispatch": "True", "default": "False"}
at::Tensor squeeze(const at::Tensor & self, at::IntArrayRef dim); // {"schema": "aten::squeeze.dims(Tensor(a) self, int[] dim) -> Tensor(a)", "dispatch": "True", "default": "True"}
at::Tensor & eq_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out); // {"schema": "aten::eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
at::Tensor isfinite(const at::Tensor & self); // {"schema": "aten::isfinite(Tensor self) -> Tensor", "dispatch": "False", "default": "True"}
at::Tensor upsample_bicubic2d(const at::Tensor & input, at::OptionalSymIntArrayRef output_size, bool align_corners, ::std::optional<at::ArrayRef<double>> scale_factors); // {"schema": "aten::upsample_bicubic2d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor", "dispatch": "False", "default": "True"}
at::Tensor matmul(const at::Tensor & self, const at::Tensor & other); // {"schema": "aten::matmul(Tensor self, Tensor other) -> Tensor", "dispatch": "True", "default": "True"}
at::Tensor _reshape_alias(const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride); // {"schema": "aten::_reshape_alias(Tensor(a) self, SymInt[] size, SymInt[] stride) -> Tensor(a)", "dispatch": "True", "default": "False"}
at::Tensor dropout(const at::Tensor & input, double p, bool train); // {"schema": "aten::dropout(Tensor input, float p, bool train) -> Tensor", "dispatch": "False", "default": "True"}
at::Tensor bitwise_left_shift(const at::Tensor & self, const at::Scalar & other); // {"schema": "aten::bitwise_left_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor", "dispatch": "True", "default": "True"}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _native_batch_norm_legit(const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, at::Tensor & running_mean, at::Tensor & running_var, bool training, double momentum, double eps); // {"schema": "aten::_native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)", "dispatch": "True", "default": "False"}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> convolution_backward_overrideable(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, c10::SymInt groups, ::std::array<bool,3> output_mask); // {"schema": "aten::convolution_backward_overrideable(Tensor grad_output, Tensor input, Tensor weight, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)", "dispatch": "True", "default": "True"}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_group_norm(const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, c10::SymInt N, c10::SymInt C, c10::SymInt HxW, int64_t group, double eps); // {"schema": "aten::native_group_norm(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps) -> (Tensor, Tensor, Tensor)", "dispatch": "True", "default": "True"}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> linear_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, ::std::array<bool,3> output_mask); // {"schema": "aten::linear_backward(Tensor self, Tensor grad_output, Tensor weight, bool[3] output_mask) -> (Tensor, Tensor, Tensor)", "dispatch": "True", "default": "False"}
