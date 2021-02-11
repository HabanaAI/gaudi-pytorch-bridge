/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include "habana_kernels/unary_kernels.h"

#include <ATen/ExpandUtils.h>
#include <torch/script.h>

using namespace torch;
using namespace at;

Tensor habana_d2d_memcpy(const Tensor& self);
Tensor habana_d2d_memcpy_other(const Tensor& self, Tensor& other);
Tensor& copy_hpu_(Tensor& self, const Tensor& src, bool non_blocking);
Tensor as_strided_hpu(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<int64_t> storage_offset);
Tensor& set_hpu_(
    Tensor& self,
    Storage source,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride);
Tensor view_hpu(const Tensor& self, IntArrayRef size);
Tensor addcmul_hpu(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha);
Tensor& addcmul_hpu_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha);
Tensor addcdiv_hpu(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha);
Tensor& addcdiv_hpu_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha);
Tensor add_tensor_hpu(const Tensor& self, const Tensor& other, Scalar alpha);
Tensor add_scalar_hpu(const Tensor& self, Scalar other, Scalar alpha);
Tensor& add_scalar_hpu_(Tensor& self, Scalar other, Scalar alpha);
Tensor& add_tensor_hpu_(Tensor& self, const Tensor& other, Scalar alpha);
Tensor sub_tensor_hpu(const Tensor& self, const Tensor& other, Scalar alpha);
Tensor& sub_tensor_hpu_(Tensor& self, const Tensor& other, Scalar alpha);
Tensor sub_scalar_hpu(const Tensor& self, Scalar other, Scalar alpha);
Tensor& sub_scalar_hpu_(Tensor& self, Scalar other, Scalar alpha);
Tensor rsub_scalar_hpu(const Tensor& self, Scalar other, Scalar alpha);
Tensor& mul_tensor_hpu_(Tensor& self, const Tensor& other);
Tensor mul_tensor_hpu(const Tensor& self, const Tensor& other);
Tensor mul_scalar_hpu(const Tensor& self, Scalar other);
Tensor& mul_scalar_hpu_(Tensor& self, Scalar other);
Tensor div_tensor_hpu(const Tensor& self, const Tensor& other);
Tensor& div_tensor_hpu_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other);
Tensor& div_tensor_hpu_(Tensor& self, const Tensor& other);
Tensor div_scalar_hpu(const Tensor& self, Scalar other);
Tensor& div_scalar_hpu_(Tensor& self, Scalar other);
Tensor pow_tensor_tensor_hpu(const Tensor& self, const Tensor& other);
Tensor& pow_tensor_tensor_hpu_(Tensor& self, const Tensor& other);
Tensor pow_tensor_scalar_hpu(const Tensor& self, Scalar other);
Tensor& pow_tensor_scalar_hpu_(Tensor& self, Scalar other);
Tensor pow_scalar_tensor_hpu(Scalar other, const Tensor& self);
Tensor gt_tensor_hpu(Tensor& self, Tensor& other);
Tensor gt_scalar_hpu(Tensor& self, Scalar other);
void eq_tensor_out_hpu(Tensor& output, const Tensor& self, const Tensor& other);
Tensor eq_tensor_hpu(Tensor& self, Tensor& other);
Tensor eq_tensor_scalar_hpu(Tensor& self, Scalar other);
Tensor lt_scalar_hpu(Tensor& self, Scalar other);
Tensor lt_tensor_hpu(Tensor& self, Tensor& other);
Tensor ge_scalar_hpu(Tensor& self, Scalar other);
Tensor ge_tensor_hpu(Tensor& self, Tensor& other);
Tensor convolution_hpu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups);
std::tuple<Tensor, Tensor, Tensor> convolution_backward_hpu(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask);
std::tuple<Tensor, Tensor, Tensor, Tensor> embedding_bag_hpu(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    bool scale_grad_by_freq,
    int64_t mode,
    UNUSED bool sparse,
    Tensor& per_sample_weights,
    UNUSED bool include_last_offset);
Tensor embedding_bag_bwd_hpu(
    Tensor& grad,
    Tensor& indices,
    Tensor& offsets,
    UNUSED Tensor& offset2bag,
    UNUSED Tensor& bag_size,
    UNUSED Tensor& maximum_indices,
    int num_weights,
    bool scale_grad_by_freq,
    int mode,
    Tensor per_sample_weights);
Tensor constant_pad_hpu(const Tensor& self, IntArrayRef pad, Scalar value);
Tensor embedding_hpu(
    const Tensor& weight,
    const Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse);
Tensor embedding_dense_backward_hpu(
    const Tensor& grad,
    const Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq);
Tensor embedding_bag_sum_hpu(
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode);
Tensor& embedding_bag_sum_bwd_out_kernel_mode_hpu(
    Tensor& out,
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode);
Tensor embedding_bag_sum_fwd_hpu(
    const Tensor& input,
    const Tensor& indices_fwd,
    const Tensor& offsets_fwd,
    const Tensor& valid_count,
    const Tensor& indices_bwd,
    const Tensor& offsets_bwd,
    const Tensor& valid_count_bwd,
    const Tensor& grad_weight);
Tensor& embedding_bag_sum_bwd_out_hpu(
    Tensor& out,
    const Tensor& input,
    const Tensor& indices_bwd,
    const Tensor& offsets_bwd,
    const Tensor& valid_count_bwd);
Tensor& fill_hpu_(Tensor& self, Scalar value);
Tensor& masked_fill_hpu_(Tensor& self, const Tensor& mask, const Tensor& value);
Tensor& masked_fill_scalar_hpu_(Tensor& self, const Tensor& mask, Scalar value);
Tensor gather_src_hpu(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    bool sparse_grad);
Tensor& scatter_inplace_src_hpu(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src);
Tensor scatter_src_hpu(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src);
Tensor scatter_add_src_hpu(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src);
Tensor& scatter_add_inplace_src_hpu(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src);
Tensor& index_add_hpu_(
    Tensor& self,
    int64_t dim_,
    const Tensor& indices,
    const Tensor& source);
Tensor index_put_hpu(
    const Tensor& self,
    TensorList indices,
    const Tensor& value,
    bool accumulate);
Tensor& index_put_hpu_(
    Tensor& self,
    TensorList indices,
    const Tensor& value,
    bool accumulate);
Tensor index_select_hpu(const Tensor& self, int64_t dim, const Tensor& index);
Tensor gather2d_hpu(
    const Tensor& input,
    const Tensor& indices,
    int64_t validCount);
Tensor slice_hpu(
    const Tensor& self,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t step);
Tensor select_hpu(const Tensor& self, int64_t dim, int64_t index);
Tensor& arange_hpu(Tensor& output, Scalar start, Scalar end, Scalar step);
at::Tensor mm_hpu(const at::Tensor& mat1, const at::Tensor& mat2);
Tensor addmm_hpu(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha);
Tensor& batch_gemm_out_hpu(Tensor& out, const Tensor& self, const Tensor& mat2);
Tensor batch_gemm_hpu(const Tensor& self, const Tensor& mat2);
Tensor dot_hpu(const Tensor& self, const Tensor& other);
Tensor mv_hpu(const Tensor& self, const Tensor& other);
std::tuple<Tensor, Tensor> nll_loss_forward_hpu(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index);
Tensor nll_loss_backward_hpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    UNUSED const Tensor& total_weight);
Tensor mse_loss_forward_hpu(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction);
Tensor mse_loss_backward_hpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction);
Tensor binary_cross_entropy_hpu(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction);
Tensor binary_cross_entropy_backward_hpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction);
std::tuple<Tensor, Tensor, Tensor> batch_norm_hpu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool training,
    double momentum,
    double eps);
std::tuple<Tensor, Tensor, Tensor> batch_norm_bwd_hpu(
    Tensor& grad_out,
    Tensor& input,
    Tensor& weight,
    UNUSED Tensor& running_mean,
    UNUSED Tensor& running_var,
    Tensor& save_mean,
    Tensor& save_invstd,
    bool train,
    double eps,
    UNUSED std::array<bool, 3> output_mask);
std::tuple<Tensor, Tensor, Tensor> layer_norm_hpu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int64_t m,
    int64_t n,
    double eps);
std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_hpu(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    std::array<bool, 3> grad_input_mask);
Tensor norm_scalar_hpu(const Tensor& self, Scalar p);
std::tuple<Tensor, Tensor> max_pool2d_with_indices_hpu(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode);
Tensor& max_pool2d_with_indices_backward_out_hpu(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& indices,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode);
Tensor max_pool2d_with_indices_backward_hpu(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices);
Tensor avg_pool2d_hpu(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);
Tensor& avg_pool2d_backward_out_hpu(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);
Tensor avg_pool2d_backward_hpu(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);
void uniform_hpu(
    const Tensor& self,
    double from = 0,
    double to = 1,
    CPUGenerator* gen = nullptr);
void normal_hpu(
    const Tensor& self,
    double mean = 0,
    double std = 1,
    CPUGenerator* gen = nullptr);
Tensor bernoulli_hpu(const Tensor& self, CPUGenerator* gen = nullptr);
Tensor& bernoulli_scalar_hpu(
    Tensor& self,
    double p,
    CPUGenerator* gen = nullptr);
std::tuple<Tensor, Tensor> fused_dropout_hpu(
    const Tensor& self,
    double p,
    CPUGenerator* gen = nullptr);
Tensor sum_dim_IntList_hpu(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype);
Tensor& sum_IntList_out_hpu(
    Tensor& output,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype);
Tensor mean_dim_hpu(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype);
Tensor& mean_dim_out_hpu(
    Tensor& output,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype);
Tensor sum_hpu(const Tensor& self, c10::optional<ScalarType> dtype);
Tensor mean_hpu(const Tensor& self, c10::optional<ScalarType> dtype);
Tensor& any_dim_out_hpu(
    Tensor& output,
    const Tensor& self,
    int64_t dim,
    bool keepdim);
Tensor any_dim_hpu(const Tensor& self, int64_t dim, bool keepdim);
Tensor any_hpu(const Tensor& self);
namespace habana {
Tensor log_softmax_hpu(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float);
Tensor log_softmax_backward_hpu(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    const Tensor& input);
Tensor softmax_hpu(const Tensor& self, int64_t dim, const bool half_to_float);
Tensor softmax_backward_hpu(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    const Tensor& input);
} // namespace habana

namespace at {
namespace native {
Tensor empty_hpu(
    IntArrayRef size,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format);
Tensor empty_strided_hpu(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options);
} // namespace native
} // namespace at

Tensor clone_hpu(const Tensor& self, c10::optional<MemoryFormat> memory_format);
Tensor& zero_hpu(Tensor& self);
Tensor cat_hpu(const TensorList tensors, int64_t dim_ = 0);
Tensor& cat_hpu_out(Tensor& result, const TensorList tensors, int64_t dim_ = 0);
Tensor transpose_hpu(const Tensor& self, int64_t dim0_, int64_t dim1_);
Tensor& transpose_hpu_(Tensor& self, int64_t dim0_, int64_t dim1_);
Tensor t_hpu(const Tensor& self);
Tensor& t_hpu_(Tensor& self);
Tensor permute_hpu(const Tensor& self, IntArrayRef dims_);
Tensor expand_hpu(const Tensor& self, IntArrayRef size, bool implicit);
std::vector<Tensor> split_with_sizes_hpu(
    const Tensor& self,
    IntArrayRef split_sizes,
    int64_t dim);
Tensor threshold_backward_hpu(
    const Tensor& grad_output,
    const Tensor& self,
    Scalar threshold);
std::tuple<Tensor&, Tensor&> topk_out_hpu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool largest,
    bool sorted);
std::tuple<Tensor, Tensor> topk_hpu(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted);
std::tuple<Tensor, Tensor> sort_hpu(
    const Tensor& self,
    int64_t dim,
    bool descending);
Tensor unary_op_hpu(
    const Tensor& input,
    std::string& node_type,
    UnaryOperator* Op);
Tensor unary_backward_op_hpu(
    const Tensor& grad_in,
    const Tensor& input,
    std::string& node_type,
    UnaryBackwardOperator* Op);
Tensor relu_hpu(const Tensor& input);
Tensor& relu_hpu_(Tensor& self);
Tensor sigmoid_hpu(const Tensor& input);
Tensor sigmoid_backward_hpu(const Tensor& grad_in, const Tensor& input);
Tensor sqrt_hpu(const Tensor& input);
Tensor tanh_hpu(const Tensor& input);
Tensor& tanh_hpu_(Tensor& self);
Tensor& tanh_out_hpu(Tensor& out, Tensor& self);
Tensor tanh_backward_hpu(const Tensor& grad_in, const Tensor& input);
Tensor gelu_hpu(const Tensor& self);
Tensor gelu_backward_hpu(const Tensor& grad, const Tensor& self);
Tensor& erf_hpu_(Tensor& self);
Tensor erf_hpu(const Tensor& self);
Tensor& exp_hpu_(Tensor& self);
Tensor exp_hpu(const Tensor& self);
Tensor& neg_out_hpu(Tensor& result, const Tensor& input);
Tensor& reciprocal_hpu_(Tensor& self);
Tensor reciprocal_hpu(const Tensor& self);
Tensor& reciprocal_out_hpu(Tensor& result, const Tensor& self);
Tensor clamp_min_hpu(const Tensor& self, Scalar min);
Tensor& clamp_hpu_(
    Tensor& self,
    c10::optional<Scalar> min,
    c10::optional<Scalar> max);
Tensor clamp_hpu(
    const Tensor& self,
    c10::optional<Scalar> min,
    c10::optional<Scalar> max);
Tensor abs_hpu(const Tensor& self);
Tensor neg_hpu(const Tensor& self);
namespace at {
namespace native {
Scalar _local_scalar_dense_hpu(const Tensor& self);
}
} // namespace at
std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_sgd_with_valid_count_hpu(
    const Tensor& gradients,
    Tensor& weights_in,
    Tensor& moments_in,
    const Tensor& indices,
    const Tensor& learning_rate,
    const Tensor& valid_count_tensor,
    float mom,
    bool nesterov);
std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_adagrad_with_valid_count_hpu(
    const Tensor& gradients,
    Tensor& weights_in,
    Tensor& moments_in,
    const Tensor& indices,
    const Tensor& learning_rate,
    const Tensor& valid_count_tensor);
Tensor ones_like_hpu(
    const Tensor& self,
    const TensorOptions& options,
    c10::optional<c10::MemoryFormat> optional_memory_format);
Tensor matmul_hpu(const at::Tensor& tensor1, const at::Tensor& tensor2);
std::tuple<at::Tensor, at::Tensor> matmul_backward_hpu(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& other);
Tensor masked_scale_hpu(const Tensor& self, const Tensor& mask, double scale);
void optimizer_adamw_hpu(
    const std::vector<at::Tensor>& gradient_vec,
    std::vector<at::Tensor>& weight_vec,
    std::vector<at::Tensor>& exp_avg_vec,
    std::vector<at::Tensor>& exp_avg_sq_vec,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const int step,
    const int bias_correction,
    const float weight_decay);
Tensor fused_norm_hpu(
    std::vector<Tensor>& grad,
    const Tensor& max_norm,
    float norm_type = 2.0);
std::tuple<std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>>
optimizer_lamb_phase1_hpu(
    const std::vector<at::Tensor>& gradient_vec,
    std::vector<at::Tensor>& weight_vec,
    std::vector<at::Tensor>& exp_avg_vec,
    std::vector<at::Tensor>& exp_avg_sq_vec,
    const Tensor& clip_global_grad_norm,
    const int grad_averaging,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const int step,
    const int bias_correction,
    const float weight_decay);
void optimizer_lamb_phase2_hpu(
    std::vector<at::Tensor>& weight_vec,
    const std::vector<at::Tensor>& adam_norm_vec,
    const std::vector<at::Tensor>& weight_norm_vec,
    const std::vector<at::Tensor>& adam_step_vec,
    const std::vector<at::Tensor>& trust_ratio_vec,
    const float step,
    const float weight_decay,
    const int use_lamb);
Tensor optimizer_lamb_fused_norm_hpu(
    const std::vector<at::Tensor>& grad,
    float max_grad_norm);
Tensor& optimizer_adagrad_hpu(
    const TensorList& gradient_vec,
    TensorList& weight_vec,
    TensorList& variance_vec,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float lrd,
    const float epsilon);
