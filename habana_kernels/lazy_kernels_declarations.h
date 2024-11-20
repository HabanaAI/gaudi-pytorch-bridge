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

#pragma once
#include <ATen/ExpandUtils.h>
#include <pthread.h>
#include <torch/csrc/api/include/torch/version.h>
#include <torch/library.h>
#include <torch/script.h>
#include <torch/version.h>
#include "backend/habana_operator.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

using OptionalIntArrayRef = at::OptionalIntArrayRef;

namespace habana_lazy {
at::Tensor bincount_hpu_lazy(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& weights,
    int64_t minlength);
at::Tensor _copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking);
at::Tensor& set_source_Storage(at::Tensor& self, at::Storage source);
at::Tensor& set_source_Tensor(at::Tensor& self, const at::Tensor& source);
at::Tensor& set_(at::Tensor& self);
at::Tensor& copy_hpu_lazy_(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking);
at::Tensor as_strided_hpu_lazy(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset);
at::Tensor as_strided_hpu(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::optional<c10::SymInt> storage_offset);
at::Tensor alias_hpu_lazy(const at::Tensor& self);
void strided_insert_hpu_lazy(
    const at::Tensor&,
    const at::Tensor&,
    bool is_flush = true);
const at::Tensor& as_strided_hpu_lazy_(
    const at::Tensor& self,
    at::SymIntArrayRef size,
    at::SymIntArrayRef stride,
    c10::optional<c10::SymInt> storage_offset);
at::Tensor& set_source_Storage_storage_offset(
    at::Tensor& self,
    at::Storage source,
    at::SymInt storage_offset,
    at::SymIntArrayRef size,
    at::SymIntArrayRef stride);
at::Tensor view_hpu(const at::Tensor& self, at::SymIntArrayRef size);
at::Tensor view_dtype_hpu(const at::Tensor& self_, c10::ScalarType dtype);
at::Tensor add_tensor_hpu_lazy(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha);
at::Tensor add_scalar_hpu_lazy(
    const at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha);
at::Tensor& add_scalar_hpu_lazy_(
    at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha);
at::Tensor& add_tensor_hpu_lazy_(
    at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha);
at::Tensor& mul_out_hpu_lazy(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out);
at::Tensor floor_divide_tensor_hpu_lazy(
    const at::Tensor& self,
    const at::Tensor& other);
at::Tensor complex_hpu(const at::Tensor& real, const at::Tensor& imag);
at::Tensor constant_pad_hpu_lazy(
    const at::Tensor& self,
    at::SymIntArrayRef pad,
    const at::Scalar& value);
at::Tensor embedding_bag_sum_hpu_lazy(
    const at::Tensor& input,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    const at::Tensor& valid_count,
    int64_t kernel_mode);
at::Tensor& embedding_bag_sum_bwd_out_kernel_mode_hpu_lazy(
    at::Tensor& out,
    const at::Tensor& input,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    const at::Tensor& valid_count,
    int64_t kernel_mode);
at::Tensor& embedding_bag_sum_bwd_out_hpu_lazy(
    at::Tensor& out,
    const at::Tensor& input,
    const at::Tensor& indices_bwd,
    const at::Tensor& offsets_bwd,
    const at::Tensor& valid_count_bwd);
at::Tensor& fill_hpu_lazy_(at::Tensor& self, const at::Scalar& value);
at::Tensor& scatter_add_inplace_src_hpu_lazy(
    at::Tensor& self,
    int64_t dim_,
    const at::Tensor& index,
    const at::Tensor& src);
at::Tensor& _index_put_impl_hpu_lazy_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value,
    const bool accumulate,
    const bool unsafe);
at::Tensor& index_add_hpu_lazy_(
    at::Tensor& self,
    int64_t dim_,
    const at::Tensor& indices,
    const at::Tensor& source,
    const at::Scalar& alpha);
at::Tensor& index_add_hpu_lazy_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& indices,
    const at::Tensor& source,
    const at::Scalar& alpha,
    at::Tensor& out);
at::Tensor index_put_hpu_lazy(
    const at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value,
    bool accumulate);
at::Tensor& index_put_hpu_lazy_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value,
    bool accumulate);
at::Tensor slice_hpu_lazy(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<int64_t> start,
    c10::optional<int64_t> end,
    int64_t step);
at::Tensor slice_backward_hpu_lazy(
    const at::Tensor& grad_out,
    c10::SymIntArrayRef input_sizes,
    int64_t dim,
    c10::SymInt start,
    c10::SymInt end,
    c10::SymInt step);
at::Tensor select_hpu_lazy(
    const at::Tensor& self,
    int64_t dim,
    c10::SymInt index);
at::Tensor masked_select_hpu_lazy(
    const at::Tensor& self,
    const at::Tensor& mask);
at::Tensor& masked_select_out_hpu_lazy(
    const at::Tensor& self,
    const at::Tensor& mask,
    at::Tensor& out);
at::Tensor nonzero_hpu_lazy(const at::Tensor& self);
at::Tensor& nonzero_out_hpu_lazy(const at::Tensor& self, at::Tensor& out);
at::Tensor kl_div_hpu_lazy(
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction,
    bool log_target);
std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_hpu_lazy(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    bool training,
    double momentum,
    double eps);
std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_legit_hpu_lazy(
    const at::Tensor& input_,
    const c10::optional<at::Tensor>& weight_tensor,
    const c10::optional<at::Tensor>& bias_tensor,
    at::Tensor& running_mean_,
    at::Tensor& running_var,
    bool training,
    double momentum,
    double eps);
std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_bwd_hpu_lazy(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    const c10::optional<at::Tensor>& save_mean,
    const c10::optional<at::Tensor>& save_invstd,
    bool train,
    double eps,
    std::array<bool, 3> output_mask);
::std::tuple<at::Tensor, at::Tensor> batch_norm_stats_lazy(
    const at::Tensor& input,
    double eps);
at::Tensor batch_norm_elemt_lazy(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    double eps);
at::Tensor batch_norm_backward_elemt_lazy(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const ::std::optional<at::Tensor>& weight,
    const at::Tensor& mean_dy,
    const at::Tensor& mean_dy_xmu,
    const at::Tensor& count);
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
batch_norm_backward_reduce_lazy(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const c10::optional<at::Tensor>& weight,
    bool input_g,
    bool weight_g,
    bool bias_g);
::std::tuple<at::Tensor, at::Tensor> batch_norm_gather_stats_with_counts_lazy(
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    double momentum,
    double eps,
    const at::Tensor& counts);
std::tuple<at::Tensor, at::Tensor, at::Tensor> instance_norm_hpu_lazy(
    const at::Tensor& input,
    const at::Tensor& weight_opt,
    const at::Tensor& bias_opt,
    double eps);
std::tuple<at::Tensor, at::Tensor, at::Tensor> instance_norm_backward_hpu_lazy(
    const at::Tensor& input,
    const at::Tensor& grad_in,
    const at::Tensor& mean,
    const at::Tensor& istd,
    const at::Tensor& gamma);
at::Tensor& uniform_hpu_lazy(
    at::Tensor& self,
    double from = 0,
    double to = 1,
    c10::optional<at::Generator> gen = c10::nullopt);
at::Tensor& normal_hpu_lazy(
    at::Tensor& self,
    double mean = 0,
    double std = 1,
    c10::optional<at::Generator> gen = c10::nullopt);
#if IS_PYTORCH_OLDER_THAN(2, 1)
at::Tensor& randperm_hpu_lazy(
    int64_t n,
    c10::optional<at::Generator> gen,
    at::Tensor& output);
#else
at::Tensor& randperm_hpu_lazy(
    c10::SymInt n,
    c10::optional<at::Generator> gen,
    at::Tensor& output);
at::Tensor randperm_nogen_hpu_lazy(
    c10::SymInt n,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory);
#endif
at::Tensor repeat_hpu(const at::Tensor& self, c10::SymIntArrayRef repeats);
at::Tensor repeat_inlv_hpu_lazy(
    const at::Tensor& self,
    c10::optional<int64_t> output_size);
std::tuple<at::Tensor, at::Tensor> _unique_hpu_lazy(
    const at::Tensor& self,
    bool sorted,
    bool return_inverse);
std::tuple<at::Tensor, at::Tensor, at::Tensor> unique2_hpu_lazy(
    const at::Tensor& self,
    bool sorted,
    bool return_inverse,
    bool return_counts);
std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_dim_hpu_lazy(
    const at::Tensor& self,
    int64_t dim,
    bool sorted,
    bool return_inverse,
    bool return_counts);
at::Tensor _copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst);
at::Tensor empty_hpu_lazy(
    at::IntArrayRef size,
    const at::TensorOptions& options,
    c10::optional<at::MemoryFormat> optional_memory_format,
    bool create_storage = true,
    synTensorType tensor_type = DATA_TENSOR,
    c10::optional<std::reference_wrapper<const at::Tensor>> base_view =
        c10::nullopt,
    bool is_strided = false);
at::Tensor empty_strided_hpu_lazy(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    const at::TensorOptions& options,
    bool create_storage = true,
    synTensorType tensor_type = DATA_TENSOR,
    int64_t storage_offset = 0,
    c10::optional<std::reference_wrapper<const at::Tensor>> base_view =
        c10::nullopt,
    bool is_strided = false);
at::Tensor transpose_hpu_lazy(
    const at::Tensor& self,
    int64_t dim0_,
    int64_t dim1_);
at::Tensor t_hpu_lazy(const at::Tensor& self);
at::Tensor squeeze_hpu_lazy(const at::Tensor& self, const int64_t dim);
at::Tensor squeeze_self_hpu_lazy(const at::Tensor& self);
at::Tensor squeeze_dim_hpu_lazy(const at::Tensor& self, const int64_t dim);
at::Tensor squeeze_dims_hpu_lazy(
    const at::Tensor& self,
    const at::IntArrayRef dims);
at::Tensor& squeeze_hpu_lazy_(at::Tensor& self);
at::Tensor& squeeze_dim_hpu_lazy_(at::Tensor& self, int64_t dim);
at::Tensor unsqueeze_hpu_lazy(const at::Tensor& self, const int64_t dim);
at::Tensor& unsqueeze_hpu_lazy_(at::Tensor& self, const int64_t dim);
std::vector<at::Tensor> unbind_hpu_lazy_(const at::Tensor& self, int64_t dim);
at::Tensor permute_hpu_lazy(const at::Tensor& self, at::IntArrayRef dims_);
at::Tensor permute_cl_hpu_lazy(const at::Tensor& self, at::IntArrayRef dims_);
at::Tensor expand_hpu_lazy(
    const at::Tensor& self,
    at::SymIntArrayRef size,
    bool implicit);

std::vector<at::Tensor> split_with_sizes_hpu_lazy(
    const at::Tensor& self,
    at::IntArrayRef split_sizes,
    int64_t dim);
std::tuple<at::Tensor, at::Tensor> sort_hpu_lazy(
    const at::Tensor& self,
    int64_t dim,
    bool descending);
at::Scalar _local_scalar_dense_hpu(const at::Tensor& self);
std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_sgd_with_valid_count_hpu_lazy(
    const at::Tensor& gradients,
    at::Tensor& weights_in,
    at::Tensor& moments_in,
    const at::Tensor& indices,
    const at::Tensor& learning_rate,
    const at::Tensor& valid_count_tensor,
    float mom,
    bool nesterov);
std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_adagrad_with_valid_count_hpu_lazy(
    const at::Tensor& gradients,
    at::Tensor& weights_in,
    at::Tensor& moments_in,
    const at::Tensor& indices,
    const at::Tensor& learning_rate,
    const at::Tensor& valid_count_tensor);
void optimizer_adamw_hpu_lazy(
    const at::TensorList gradient_vec,
    at::TensorList weight_vec,
    at::TensorList exp_avg_vec,
    at::TensorList exp_avg_sq_vec,
    const at::Tensor& neg_step_t,
    const double beta1,
    const double beta2,
    const double epsilon,
    const double weight_decay,
    c10::optional<at::TensorList> exp_avg_scales = c10::nullopt,
    c10::optional<at::TensorList> exp_avg_sq_scales = c10::nullopt);
at::Tensor fused_norm_hpu_lazy(
    std::vector<at::Tensor>& grad,
    const at::Tensor& max_norm,
    float norm_type = 2.0);
at::Tensor optimizer_lamb_norm_hpu_lazy(
    const std::vector<at::Tensor>& grad,
    double max_grad_norm);
void optimizer_lamb_phase1(
    const at::TensorList gradients,
    const at::TensorList weights,
    at::TensorList exp_avg,
    at::TensorList exp_avg_sq,
    at::TensorList out_weight_norms,
    at::TensorList out_adam_norms,
    at::TensorList out_adam_steps,
    const at::Tensor& clip_global_grad_norm,
    const int64_t grad_averaging,
    const double beta1,
    const double beta2,
    const double epsilon,
    const at::Tensor& bias_correction1,
    const at::Tensor& bias_correction2,
    const double weight_decay);
void optimizer_lamb_phase2(
    at::TensorList weights,
    const at::TensorList adam_norms,
    const at::TensorList weight_norms,
    const at::TensorList adam_steps,
    const at::Tensor& neg_step,
    const double weight_decay,
    const bool use_lamb);
void optimizer_ema_hpu_lazy(
    const at::TensorList model_inputs,
    at::TensorList updated_ema,
    const at::Tensor& decay);
void optimizer_adagrad_hpu_lazy(
    const at::TensorList& gradients,
    at::TensorList& weights,
    at::TensorList& variances,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float lrd,
    const float epsilon);
void optimizer_sgd_hpu_lazy(
    const at::TensorList& gradients,
    at::TensorList& weights,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov);
void optimizer_sgd_momentum_hpu_lazy(
    const at::TensorList& gradients,
    at::TensorList& weights,
    at::TensorList& momentum,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const at::Tensor& mom,
    const float wd,
    const float damp,
    const bool nesterov);
void optimizer_lars_hpu_lazy(
    const at::TensorList params,
    at::TensorList grads,
    const std::vector<int64_t> skipMasks,
    const float eeta,
    const float weight_decay,
    const float eps,
    const float lr);
at::Tensor ones_like_hpu_lazy(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> memory_format);
at::Tensor matmul_hpu_lazy(
    const at::Tensor& self,
    const at::Tensor& other,
    c10::optional<at::ScalarType> dtype = c10::nullopt);
std::tuple<at::Tensor, at::Tensor> matmul_backward_hpu_lazy(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& other,
    c10::optional<at::ScalarType> dtype = c10::nullopt);
at::Tensor habana_nms_hpu_lazy(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    float iou_threshold);
at::Tensor batched_nms_hpu_lazy(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    const at::Tensor& indexes,
    float iou_threshold);
at::Tensor roi_align_fwd_hpu_lazy(
    const at::Tensor& images,
    const at::Tensor& rois,
    const at::Tensor& num_rois,
    int output_h,
    int output_w,
    int mode,
    int sampling_ratio,
    float spatial_scale,
    bool aligned);
at::Tensor roi_align_bwd_hpu_lazy(
    const at::Tensor& grad_out,
    const at::Tensor& rois,
    const at::Tensor& num_rois,
    int bs,
    int ch,
    int h,
    int w,
    int sampling_ratio,
    float spatial_scale,
    bool aligned);
at::Tensor& broadcast_hpu_lazy_(
    at::Tensor& tensor,
    int64_t root_rank,
    int64_t comm_id);
at::Tensor& allreduce_hpu_lazy_(
    at::Tensor& tensor,
    uint8_t reduce_op,
    int64_t comm_id);
at::Tensor& reduce_hpu_lazy_(
    at::Tensor& tensor,
    int64_t dst_rank,
    uint8_t reduce_op,
    int64_t comm_id);
at::Tensor& alltoall_hpu_lazy_out(
    const at::Tensor& input_tensor,
    int64_t comm_id,
    at::Tensor& output_tensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes);
at::Tensor& allgather_hpu_lazy_out(
    const at::Tensor& inputTensor,
    int64_t comm_id,
    at::Tensor& output_tensor);
at::Tensor& reduce_scatter_hpu_lazy_out(
    const at::Tensor& input_tensor,
    uint8_t reduce_op,
    int64_t comm_id,
    at::Tensor& output_tensor);
at::Tensor& send_hpu_lazy_(
    at::Tensor& tensor,
    int64_t dst_rank,
    int64_t tag,
    int64_t comm_id);
at::Tensor& recv_hpu_lazy_(
    at::Tensor& tensor,
    int64_t src_rank,
    int64_t tag,
    int64_t comm_id);
std::tuple<at::Tensor&, at::Tensor&> cast_to_fp8_lazy(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& amax);
std::tuple<at::Tensor, at::Tensor> cast_to_fp8_v2_lazy(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    bool is_amax,
    at::ScalarType dtype,
    OptionalIntArrayRef scale_shape);
std::tuple<at::Tensor, at::Tensor> cast_to_fp8_v2_scalar_lazy(
    const at::Tensor& input,
    double scale,
    bool stochastic_rounding,
    bool is_amax,
    at::ScalarType dtype,
    OptionalIntArrayRef scale_shape);
std::tuple<at::Tensor, at::Tensor> cast_to_fp8_v2_scalar_list_lazy(
    const at::Tensor& input,
    c10::ArrayRef<double> scale,
    bool stochastic_rounding,
    bool is_amax,
    at::ScalarType dtype,
    OptionalIntArrayRef scale_shape);
at::Tensor cast_from_fp8_lazy(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    at::ScalarType out_dtype,
    OptionalIntArrayRef scale_shape);
at::Tensor cast_from_fp8_scalar_lazy(
    const at::Tensor& input,
    double scale,
    at::ScalarType out_dtype,
    OptionalIntArrayRef scale_shape);
at::Tensor cast_from_fp8_scalar_list_lazy(
    const at::Tensor& input,
    c10::ArrayRef<double> scale,
    at::ScalarType out_dtype,
    OptionalIntArrayRef scale_shape);
at::Tensor convert_from_int4_lazy(
    const at::Tensor& input,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& zero_point,
    at::ScalarType out_dtype);
at::Tensor convert_from_uint4_lazy(
    const at::Tensor& input,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& zero_point,
    at::ScalarType out_dtype);
at::Tensor& fp8_gemm_lazy(
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    const at::Tensor& D,
    at::ScalarType out_dtype,
    const c10::optional<at::Tensor>& A_scale_inv,
    const c10::optional<at::Tensor>& B_scale_inv,
    const c10::optional<at::Tensor>& bias,
    bool accumulate,
    at::Tensor& out);
at::Tensor fp8_gemm_v2_lazy(
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    const c10::optional<at::Tensor>& D,
    at::ScalarType out_dtype,
    const c10::optional<at::Tensor>& A_scale_inv,
    const c10::optional<at::Tensor>& B_scale_inv,
    const c10::optional<at::Tensor>& bias,
    bool accumulate,
    OptionalIntArrayRef B_scale_shape);
at::Tensor fp8_gemm_v2_lazy_scalar(
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    const c10::optional<at::Tensor>& D,
    at::ScalarType out_dtype,
    double A_scale_inv,
    double B_scale_inv,
    const c10::optional<at::Tensor>& bias,
    bool accumulate,
    OptionalIntArrayRef B_scale_shape);
at::Tensor fp8_gemm_v2_lazy_scalar_list(
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    const c10::optional<at::Tensor>& D,
    at::ScalarType out_dtype,
    c10::ArrayRef<double> A_scale_inv,
    c10::ArrayRef<double> B_scale_inv,
    const c10::optional<at::Tensor>& bias,
    bool accumulate,
    OptionalIntArrayRef B_scale_shape);
std::tuple<at::Tensor, at::Tensor, at::Tensor>
native_group_norm_backward_hpu_lazy(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const c10::optional<at::Tensor>& weight,
    c10::SymInt N,
    c10::SymInt C,
    c10::SymInt HxW,
    int64_t group,
    std::array<bool, 3> output_mask);
at::Tensor habana_random_seed_lazy(const at::Tensor& input);
std::vector<at::Tensor> habana_permute_1D_sparse_data_lazy(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& weights);
std::vector<at::Tensor> habana_permute_2D_sparse_data_lazy(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& weights);
at::Tensor habana_expand_into_jagged_permute_lazy(
    const at::Tensor& permute,
    const at::Tensor& input_offsets,
    const at::Tensor& output_offsets,
    int64_t output_size);
at::Tensor mixture_of_experts_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max);
at::Tensor mixture_of_experts_fused_weights_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max);
at::Tensor habana_split_permute_cat_lazy(
    const at::Tensor& input,
    const at::Tensor& indices,
    int64_t batch_size,
    int64_t num_features,
    int64_t dims);
at::Tensor _ragged_softmax(
    const at::Tensor& self,
    int64_t dim,
    bool half_to_float,
    const at::Tensor& valid_count);
at::Tensor scaled_masked_softmax_lazy(
    const at::Tensor& input,
    const at::Tensor& mask,
    double scale);
at::Tensor custom_softmax_lazy(const at::Tensor& input, int64_t flavor);
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>
habana_bounds_check_indices_lazy(
    at::Tensor& indices,
    at::Tensor& offsets,
    at::Tensor& warning,
    const at::Tensor& rows_per_table,
    int64_t bounds_check_mode,
    const c10::optional<at::Tensor>& weights);
at::Tensor rotary_pos_embedding_lazy(
    const at::Tensor& input,
    const at::Tensor& sin,
    const at::Tensor& cos,
    const c10::optional<at::Tensor>& position_ids,
    const int64_t offset,
    const int64_t mode);
at::Tensor rotary_pos_embedding_backward_lazy(
    const at::Tensor& grad_in,
    const at::Tensor& sin,
    const at::Tensor& cos,
    const c10::optional<at::Tensor>& position_ids,
    const int64_t offset,
    const int64_t mode);
std::tuple<at::Tensor, at::Tensor> ctc_loss_custom_lazy(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    int64_t blank,
    int64_t reduction,
    bool zero_infinity);
at::Tensor ctc_loss_custom_backward_lazy(
    const at::Tensor& grad,
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    const at::Tensor& neg_log_likelihood,
    const at::Tensor& log_alpha,
    int64_t blank,
    int64_t reduction,
    bool zero_infinity);
at::Tensor masked_batch_gemm_lazy(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& mask_a,
    const at::Tensor& mask_b,
    bool trans_a,
    bool trans_b);
std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_fwd_lazy(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    c10::string_view softmax_mode,
    const c10::optional<at::Tensor>& valid_seq_len,
    c10::string_view seq_padding_type);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fp8_sdpa_fwd_lazy(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    c10::string_view softmax_mode,
    const c10::optional<at::Tensor>& d_scale_q,
    const c10::optional<at::Tensor>& d_scale_k,
    const c10::optional<at::Tensor>& d_scale_v,
    const c10::optional<at::Tensor>& q_scale_s,
    const c10::optional<at::Tensor>& q_scale_o,
    const c10::optional<at::Tensor>& d_scale_s,
    const bool is_amax_s,
    const c10::optional<at::Tensor>& valid_seq_len,
    c10::string_view seq_padding_type);

std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_bwd_lazy(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& P,
    const c10::optional<at::Tensor>& dm,
    const bool is_causal,
    const double p,
    const double scale,
    const at::Tensor& fwd_out);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fp8_sdpa_bwd_lazy(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& P,
    const c10::optional<at::Tensor>& dm,
    const bool is_causal,
    const double p,
    const double scale,
    const c10::optional<at::Tensor>& d_scale_q,
    const c10::optional<at::Tensor>& d_scale_k,
    const c10::optional<at::Tensor>& d_scale_v,
    const c10::optional<at::Tensor>& d_scale_s,
    const c10::optional<at::Tensor>& d_scale_do,
    const c10::optional<at::Tensor>& d_scale_ds,
    const c10::optional<at::Tensor>& q_scale_s,
    const c10::optional<at::Tensor>& q_scale_ds,
    const bool is_amax_ds,
    const at::Tensor& fwd_out);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> sdpa_recomp_fwd_lazy(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    const bool requires_backward,
    c10::string_view softmax_mode,
    const c10::optional<at::Tensor>& valid_seq_len,
    c10::string_view seq_padding_type);
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
fp8_sdpa_recomp_fwd_lazy(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    const bool requires_backward,
    c10::string_view softmax_mode,
    const c10::optional<at::Tensor> d_scale_q,
    const c10::optional<at::Tensor> d_scale_k,
    const c10::optional<at::Tensor> d_scale_v,
    const c10::optional<at::Tensor> q_scale_s,
    const c10::optional<at::Tensor> q_scale_o,
    const c10::optional<at::Tensor> d_scale_s,
    const bool is_amax_s,
    const bool is_amax_o,
    const c10::optional<at::Tensor>& valid_seq_len,
    c10::string_view seq_padding_type);

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
fp8_sdpa_recomp_fwd_scalar_lazy(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    const bool requires_backward,
    c10::string_view softmax_mode,
    const double d_scale_q,
    const double d_scale_k,
    const double d_scale_v,
    const double q_scale_s,
    const double q_scale_o,
    const double d_scale_s,
    const bool is_amax_s,
    const bool is_amax_o,
    const c10::optional<at::Tensor>& valid_seq_len,
    c10::string_view seq_padding_type);

std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_recomp_bwd_lazy(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const at::Tensor& m,
    const at::Tensor& linv,
    const c10::optional<at::Tensor>& seed,
    const bool is_causal,
    const double p,
    const double scale,
    const c10::string_view softmax_mode,
    const at::Tensor& fwd_out);
at::Tensor scaled_triangular_softmax_lazy(
    const at::Tensor& self,
    double inv_scale_attn,
    const c10::optional<at::Tensor>& exp_sum_recpr,
    const c10::optional<at::Tensor>& max);
std::tuple<at::Tensor, at::Tensor, at::Tensor>
scaled_triangular_softmax_retain_lazy(
    const at::Tensor& self,
    double inv_scale_attn);
at::Tensor& kv_reorder_lazy(
    at::Tensor& self,
    const at::Tensor start,
    const at::Tensor end,
    const at::Tensor beam_idx);
at::Tensor& in_place_interleave_lazy(at::Tensor& self);
at::Tensor scaled_masked_triangular_softmax_lazy(
    const at::Tensor& self,
    const at::Tensor& start_end,
    double inv_scale_attn,
    int64_t grouped_batch_size,
    bool use_max,
    int64_t mode,
    c10::optional<at::ScalarType> out_dtype);

} // namespace habana_lazy
