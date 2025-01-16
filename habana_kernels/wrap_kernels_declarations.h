/**
 * Copyright (c) 2020-2024 Intel Corporation
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
#include <torch/script.h>
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

// Ops from hpu_wrap namespace moved to hpu_op.yaml

std::tuple<at::Tensor&, at::Tensor&>
optimizer_sparse_sgd_with_valid_count_hpu_wrap(
    const at::Tensor& gradients,
    at::Tensor& weights_in,
    at::Tensor& moments_in,
    const at::Tensor& indices,
    const at::Tensor& learning_rate,
    const at::Tensor& valid_count_tensor,
    float mom,
    bool nesterov);
std::tuple<at::Tensor&, at::Tensor&>
optimizer_sparse_adagrad_with_valid_count_hpu_wrap(
    const at::Tensor& gradients,
    at::Tensor& weights_in,
    at::Tensor& moments_in,
    const at::Tensor& indices,
    const at::Tensor& learning_rate,
    const at::Tensor& valid_count_tensor);
void optimizer_adamw_hpu_wrap(
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
at::Tensor fused_norm_hpu_wrap(
    std::vector<at::Tensor>& grad,
    const at::Tensor& max_norm,
    float norm_type = 2.0);
void optimizer_adagrad_hpu_wrap(
    const at::TensorList& gradients,
    at::TensorList& weights,
    at::TensorList& variances,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float lrd,
    const float epsilon);
void optimizer_ema_hpu_wrap(
    const at::TensorList model_inputs,
    at::TensorList updated_ema,
    const at::Tensor& decay);
void optimizer_sgd_hpu_wrap(
    const at::TensorList gradients,
    at::TensorList weights,
    at::Tensor& lr,
    double wd,
    double mom,
    double damp,
    bool nesterov);
void optimizer_sgd_momentum_hpu_wrap(
    const at::TensorList gradients,
    at::TensorList weights,
    at::TensorList momentum,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const at::Tensor& mom,
    double wd,
    double damp,
    bool nesterov);
at::Tensor embedding_bag_sum_hpu_wrap(
    const at::Tensor& input,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    const at::Tensor& valid_count,
    int64_t kernel_mode);
at::Tensor& embedding_bag_sum_bwd_out_kernel_mode_hpu_wrap(
    at::Tensor& out,
    const at::Tensor& input,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    const at::Tensor& valid_count,
    int64_t kernel_mode);
at::Tensor gather2d_hpu_wrap(
    const at::Tensor& input,
    const at::Tensor& indices,
    int64_t validCount);
void optimizer_lars_hpu_wrap(
    const at::TensorList params,
    at::TensorList grads,
    c10::ArrayRef<int64_t> skip_masks,
    const double eeta,
    const double weight_decay,
    const double eps,
    const at::Tensor& lr);
void optimizer_lars_hpu_wrap(
    const at::TensorList params,
    at::TensorList grads,
    const std::vector<int64_t> skipMasks,
    const float eeta,
    const float weight_decay,
    const float eps,
    const float lr);
void optimizer_resource_apply_momentum_hpu_wrap(
    at::TensorList params_momentum_buf_list,
    const at::TensorList dp_list,
    const double momentum);
at::Tensor batched_nms_hpu_wrap(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    const at::Tensor& indices,
    float iou_threshold);
at::Tensor torchvision_nms_hpu_wrap(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    double iou_threshold);
at::Tensor matmul_ex_wrap(
    const at::Tensor& self,
    const at::Tensor& other,
    at::ScalarType dtype);
std::tuple<at::Tensor, at::Tensor> matmul_ex_backward_wrap(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& other,
    at::ScalarType dtype);
at::Tensor habana_random_seed_wrap(const at::Tensor& input);
std::vector<at::Tensor> habana_permute_1D_sparse_data_wrap(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& weights);
std::vector<at::Tensor> habana_permute_2D_sparse_data_wrap(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& weights);
at::Tensor habana_expand_into_jagged_permute_wrap(
    const at::Tensor& permute,
    const at::Tensor& input_offsets,
    const at::Tensor& output_offsets,
    int64_t output_size);
at::Tensor habana_split_permute_cat_wrap(
    const at::Tensor& input,
    const at::Tensor& indices,
    int64_t batch_size,
    int64_t num_features,
    int64_t dims);
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>
habana_bounds_check_indices_wrap(
    at::Tensor& indices,
    at::Tensor& offsets,
    at::Tensor& warning,
    const at::Tensor& rows_per_table,
    int64_t bounds_check_mode,
    const c10::optional<at::Tensor>& weights);
std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_fwd_wrap(
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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fp8_sdpa_fwd_wrap(
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_bwd_wrap(
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
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fp8_sdpa_bwd_wrap(
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_recomp_bwd_wrap(
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

namespace vision {
namespace ops {

at::Tensor roi_align_fwd_wrap(
    const at::Tensor& images,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t output_h,
    int64_t output_w,
    int64_t sampling_ratio,
    bool aligned);
at::Tensor roi_align_bwd_wrap(
    const at::Tensor& grad_out,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t output_h,
    int64_t output_w,
    int64_t bs,
    int64_t ch,
    int64_t h,
    int64_t w,
    int64_t sampling_ratio,
    bool aligned);

} // namespace ops
} // namespace vision
