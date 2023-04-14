/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#if IS_PYTORCH_OLDER_THAN(1, 13)
const at::Tensor& as_strided_hpu_lazy_(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset);
#else
const at::Tensor& as_strided_hpu_lazy_(
    const at::Tensor& self,
    at::SymIntArrayRef size,
    at::SymIntArrayRef stride,
    c10::optional<c10::SymInt> storage_offset);
#endif
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
at::Tensor baddbmm_hpu_lazy(
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& beta,
    const at::Scalar& alpha);
at::Tensor& baddbmm_hpu_lazy_(
    at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& beta,
    const at::Scalar& alpha);
at::Tensor& mul_out_hpu_lazy(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out);
at::Tensor floor_divide_tensor_hpu_lazy(
    const at::Tensor& self,
    const at::Tensor& other);
#if IS_PYTORCH_OLDER_THAN(1, 14)
at::Tensor constant_pad_hpu_lazy(
    const at::Tensor& self,
    at::IntArrayRef pad,
    const at::Scalar& value);
#else
at::Tensor constant_pad_hpu_lazy(
    const at::Tensor& self,
    at::SymIntArrayRef pad,
    const at::Scalar& value);
#endif
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
at::Tensor embedding_bag_sum_fwd_hpu_lazy(
    const at::Tensor& input,
    const at::Tensor& indices_fwd,
    const at::Tensor& offsets_fwd,
    const at::Tensor& valid_count,
    const at::Tensor& indices_bwd,
    const at::Tensor& offsets_bwd,
    const at::Tensor& valid_count_bwd,
    const at::Tensor& grad_weight);
at::Tensor& embedding_bag_sum_bwd_out_hpu_lazy(
    at::Tensor& out,
    const at::Tensor& input,
    const at::Tensor& indices_bwd,
    const at::Tensor& offsets_bwd,
    const at::Tensor& valid_count_bwd);
at::Tensor& fill_hpu_lazy_(at::Tensor& self, const at::Scalar& value);
at::Tensor scatter_add_src_hpu_lazy(
    const at::Tensor& self,
    int64_t dim_,
    const at::Tensor& index,
    const at::Tensor& src);
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
at::Tensor& index_fill_hpu_lazy_(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Scalar& value);
at::Tensor& index_copy_hpu_lazy_(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& value);
at::Tensor slice_hpu_lazy(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<int64_t> start,
    c10::optional<int64_t> end,
    int64_t step);
at::Tensor select_hpu_lazy(
    const at::Tensor& self,
    int64_t dim,
#if IS_PYTORCH_OLDER_THAN(1, 14)
    int64_t
#else
    c10::SymInt
#endif
        index);
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
at::Tensor kl_div_backward_hpu_lazy(
    const at::Tensor& grad,
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
    const c10::optional<at::Tensor>& weight,
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
#endif
at::Tensor bernoulli_hpu_lazy(
    const at::Tensor& self,
    c10::optional<at::Generator> gen = c10::nullopt);
at::Tensor& bernoulli_scalar_hpu_lazy(
    at::Tensor& self,
    double p,
    c10::optional<at::Generator> gen = c10::nullopt);
#if IS_PYTORCH_OLDER_THAN(1, 13)
at::Tensor repeat_hpu_lazy(const at::Tensor& self, c10::IntArrayRef repeats);
#else
at::Tensor repeat_hpu_lazy(const at::Tensor& self, c10::SymIntArrayRef repeats);
#endif
at::Tensor repeat_inlv_hpu_lazy(
    const at::Tensor& self,
    c10::optional<int64_t> output_size);
#if IS_PYTORCH_OLDER_THAN(1, 13)
at::Tensor sum_dim_IntList_hpu_lazy(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype);
#else
at::Tensor sum_dim_IntList_hpu_lazy(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype);
#endif

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
at::Tensor clone_hpu_lazy(
    const at::Tensor& self,
    c10::optional<at::MemoryFormat> memory_format);
at::Tensor cat_hpu_lazy(const at::ITensorListRef& tensors, int64_t dim_ = 0);
at::Tensor transpose_hpu_lazy(
    const at::Tensor& self,
    int64_t dim0_,
    int64_t dim1_);
at::Tensor t_hpu_lazy(const at::Tensor& self);
at::Tensor squeeze_hpu_lazy(const at::Tensor& self, const int64_t dim);
at::Tensor squeeze_self_hpu_lazy(const at::Tensor& self);
at::Tensor squeeze_dim_hpu_lazy(const at::Tensor& self, const int64_t dim);
at::Tensor& squeeze_hpu_lazy_(at::Tensor& self);
at::Tensor& squeeze_dim_hpu_lazy_(at::Tensor& self, int64_t dim);
at::Tensor unsqueeze_hpu_lazy(const at::Tensor& self, const int64_t dim);
at::Tensor& unsqueeze_hpu_lazy_(at::Tensor& self, const int64_t dim);
std::vector<at::Tensor> unbind_hpu_lazy_(const at::Tensor& self, int64_t dim);
at::Tensor permute_hpu_lazy(const at::Tensor& self, at::IntArrayRef dims_);
at::Tensor permute_cl_hpu_lazy(const at::Tensor& self, at::IntArrayRef dims_);
#if IS_PYTORCH_OLDER_THAN(1, 13)
at::Tensor expand_hpu_lazy(
    const at::Tensor& self,
    at::IntArrayRef size,
    bool implicit);
#else
at::Tensor expand_hpu_lazy(
    const at::Tensor& self,
    at::SymIntArrayRef size,
    bool implicit);
#endif

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
    const at::TensorList& gradient_vec,
    at::TensorList& weight_vec,
    at::TensorList& exp_avg_vec,
    at::TensorList& exp_avg_sq_vec,
    at::Tensor& lr_t,
    at::Tensor& neg_step_t,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weight_decay);
at::Tensor fused_norm_hpu_lazy(
    std::vector<at::Tensor>& grad,
    const at::Tensor& max_norm,
    float norm_type = 2.0);
at::Tensor optimizer_lamb_fused_norm_hpu_lazy(
    const std::vector<at::Tensor>& grad,
    double max_grad_norm);
std::tuple<
    std::vector<at::Tensor>,
    std::vector<at::Tensor>,
    std::vector<at::Tensor>>
optimizer_lamb_phase1_hpu_lazy(
    const std::vector<at::Tensor>& gradients,
    std::vector<at::Tensor>& weights,
    std::vector<at::Tensor>& exp_avg,
    std::vector<at::Tensor>& exp_avg_sq,
    const at::Tensor& clip_global_grad_norm,
    const int grad_averaging,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const int step,
    const int bias_correction,
    const float weight_decay);
void optimizer_lamb_phase2_hpu_lazy(
    std::vector<at::Tensor>& weight_vec,
    const std::vector<at::Tensor>& adam_norm_vec,
    const std::vector<at::Tensor>& weight_norm_vec,
    const std::vector<at::Tensor>& adam_step_vec,
    const float step,
    const float weight_decay,
    const int use_lamb);
void optimizer_ema_hpu_lazy(
    const at::TensorList& model_inputs,
    at::TensorList& updated_ema,
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
    const at::TensorList& params,
    at::TensorList& grads,
    const std::vector<int64_t> skipMasks,
    const float eeta,
    const float weight_decay,
    const float eps,
    const float lr);
void optimizer_ResourceApplyMomentum_hpu_lazy(
    at::TensorList& params_momentum_buffer_list,
    const at::TensorList& d_p_list,
    const float momentum);
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

at::Tensor linear_non2d_hpu_lazy(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::ScalarType> dtype = c10::nullopt);
std::vector<at::Tensor> linear_non2d_bwd_hpu_lazy(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& bias_grad_opt = c10::nullopt,
    const c10::optional<at::ScalarType> dtype = c10::nullopt);
#if IS_PYTORCH_FORK_AT_LEAST(1, 0)
at::Tensor habana_cast_to_fp8_lazy(
    const at::Tensor& input,
    bool stochastic_rounding,
    int seed);
#endif
std::tuple<at::Tensor&, at::Tensor&> cast_to_fp8_lazy(
    const at::Tensor& input,
    const at::Tensor& scale,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& amax);
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> fp8_cast_transpose_lazy(
    const at::Tensor& input,
    const at::Tensor& scale,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& amax,
    at::Tensor& transposed);
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>
fp8_cast_transpose_bgrad_lazy(
    const at::Tensor& input,
    const at::Tensor& scale,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& amax,
    at::Tensor& transposed,
    at::Tensor& bgrad);
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>
fp8_cast_transpose_bgrad_dgelu_lazy(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& retain,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& amax,
    at::Tensor& transposed,
    at::Tensor& bgrad);
at::Tensor cast_from_fp8_lazy(
    const at::Tensor& input,
    const at::Tensor& scale,
    at::ScalarType out_dtype);
std::tuple<at::Tensor, at::Tensor, at::Tensor> fp8_dropout_lazy(
    const at::Tensor& input,
    double p,
    const at::Tensor& scale,
    bool stochastic_rounding);
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> fp8_gelu_lazy(
    const at::Tensor& input,
    const at::Tensor& scale,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& amax,
    at::Tensor& retain);
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>
fp8_layernorm_lazy(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double eps,
    const at::Tensor& scale,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& amax,
    at::Tensor& mean,
    at::Tensor& istd);
at::Tensor& fp8_gemm_lazy(
    const at::Tensor& A,
    const at::Tensor& A_scale_inv,
    bool trans_A,
    const at::Tensor& B,
    const at::Tensor& B_scale_inv,
    bool trans_B,
    const at::Tensor& D,
    at::ScalarType out_dtype,
    const c10::optional<at::Tensor>& bias,
    bool accumulate,
    at::Tensor& out);
at::Tensor& fp8_transpose_lazy(const at::Tensor& input, at::Tensor& out);
at::Tensor& fp8_permute_lazy(
    const at::Tensor& input,
    at::IntArrayRef dims,
    at::Tensor& out);
at::Tensor fp8_reshape_lazy(const at::Tensor& input, at::IntArrayRef shape);
::std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_bwd_hpu_lazy(
    const at::Tensor& self,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    ::std::array<bool, 3> output_mask);
std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm_hpu_lazy(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    c10::SymInt N,
    c10::SymInt C,
    c10::SymInt HxW,
    int64_t group,
    double eps);
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
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>
habana_bounds_check_indices_lazy(
    at::Tensor& indices,
    at::Tensor& offsets,
    at::Tensor& warning,
    const at::Tensor& rows_per_table,
    int64_t bounds_check_mode,
    const c10::optional<at::Tensor>& weights);
} // namespace habana_lazy
