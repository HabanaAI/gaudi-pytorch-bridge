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

Tensor & __ilshift__(Tensor & self, const Scalar & other); // {"schema": "aten::__ilshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)", "dispatch": "True", "default": "False"}
void _foreach_add_(TensorList self, const Scalar & scalar); // {"schema": "aten::_foreach_add_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()", "dispatch": "True", "default": "False"}
::std::tuple<Tensor,Tensor> _fused_dropout(const Tensor & self, double p, c10::optional<Generator> generator); // {"schema": "aten::_fused_dropout(Tensor self, float p, Generator? generator=None) -> (Tensor, Tensor)", "dispatch": "True", "default": "False"}
::std::tuple<Tensor,Tensor> native_dropout(const Tensor & input, double p, c10::optional<bool> train); // {"schema": "aten::native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)", "dispatch": "True", "default": "False"}
Tensor as_strided(const Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, c10::optional<c10::SymInt> storage_offset); // {"schema": "aten::as_strided(Tensor(a) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a)", "dispatch": "True", "default": "False"}
Tensor addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha); // {"schema": "aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor", "dispatch": "True", "default": "False"}
Tensor bucketize(const Scalar & self, const Tensor & boundaries, bool out_int32, bool right); // {"schema": "aten::bucketize.Scalar(Scalar self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor", "dispatch": "True", "default": "False"}
Tensor elu(const Tensor & self, const Scalar & alpha, const Scalar & scale, const Scalar & input_scale); // {"schema": "aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor", "dispatch": "True", "default": "True"}
Tensor & prod_out(const Tensor & self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype, Tensor & out); // {"schema": "aten::prod.int_out(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
Tensor clone(const Tensor & self, c10::optional<MemoryFormat> memory_format); // {"schema": "aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor", "dispatch": "True", "default": "True"}
Tensor & mul_out(const Tensor & self, const Scalar & other, Tensor & out); // {"schema": "aten::mul.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "True"}
::std::tuple<Tensor &,Tensor &> sort_out(const Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending, Tensor & values, Tensor & indices); // {"schema": "aten::sort.values_stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)", "dispatch": "True", "default": "False"}
Tensor squeeze(const Tensor & self, IntArrayRef dim); // {"schema": "aten::squeeze.dims(Tensor(a) self, int[] dim) -> Tensor(a)", "dispatch": "True", "default": "True"}
Tensor & eq_out(const Tensor & self, const Scalar & other, Tensor & out); // {"schema": "aten::eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
Tensor isfinite(const Tensor & self); // {"schema": "aten::isfinite(Tensor self) -> Tensor", "dispatch": "False", "default": "True"}
Tensor upsample_bicubic2d(const Tensor & input, OptionalSymIntArrayRef output_size, bool align_corners, c10::optional<ArrayRef<double>> scale_factors); // {"schema": "aten::upsample_bicubic2d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor", "dispatch": "False", "default": "True"}
Tensor matmul(const Tensor & self, const Tensor & other); // {"schema": "aten::matmul(Tensor self, Tensor other) -> Tensor", "dispatch": "True", "default": "True"}
Tensor _reshape_alias(const Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride); // {"schema": "aten::_reshape_alias(Tensor(a) self, SymInt[] size, SymInt[] stride) -> Tensor(a)", "dispatch": "True", "default": "False"}
Tensor dropout(const Tensor & input, double p, bool train); // {"schema": "aten::dropout(Tensor input, float p, bool train) -> Tensor", "dispatch": "False", "default": "True"}
Tensor bitwise_left_shift(const Tensor & self, const Scalar & other); // {"schema": "aten::bitwise_left_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor", "dispatch": "True", "default": "True"}
::std::tuple<Tensor,Tensor,Tensor> _native_batch_norm_legit(const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, Tensor & running_mean, Tensor & running_var, bool training, double momentum, double eps); // {"schema": "aten::_native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)", "dispatch": "True", "default": "False"}
::std::tuple<Tensor,Tensor,Tensor> convolution_backward_overrideable(const Tensor & grad_output, const Tensor & input, const Tensor & weight, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, c10::SymInt groups, ::std::array<bool,3> output_mask); // {"schema": "aten::convolution_backward_overrideable(Tensor grad_output, Tensor input, Tensor weight, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)", "dispatch": "True", "default": "True"}
::std::tuple<Tensor,Tensor,Tensor> native_group_norm(const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, c10::SymInt N, c10::SymInt C, c10::SymInt HxW, int64_t group, double eps); // {"schema": "aten::native_group_norm(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps) -> (Tensor, Tensor, Tensor)", "dispatch": "True", "default": "True"}
::std::tuple<Tensor,Tensor,Tensor> linear_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, ::std::array<bool,3> output_mask); // {"schema": "aten::linear_backward(Tensor self, Tensor grad_output, Tensor weight, bool[3] output_mask) -> (Tensor, Tensor, Tensor)", "dispatch": "True", "default": "False"}
