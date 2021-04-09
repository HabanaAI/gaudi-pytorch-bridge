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
#include <ATen/ExpandUtils.h>
#include <pthread.h>
#include <torch/script.h>

#define HPU_LAZY_FUNC_DECL(op_code) \
  Tensor op_code##_hpu_lazy(const Tensor& self);

using namespace torch;
using namespace at;

#include "habana_kernels/unary_kernels.h"

Tensor& copy_hpu_lazy_(Tensor& self, const Tensor& src, bool non_blocking);
Tensor as_strided_hpu_lazy(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<int64_t> storage_offset);
Tensor asin_hpu_lazy(const Tensor& self);
HPU_LAZY_FUNC_DECL(acos)
Tensor& set_hpu_lazy_(
    Tensor& self,
    Storage source,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride);
Tensor view_hpu_lazy(const Tensor& self, IntArrayRef size);
Tensor addcmul_hpu_lazy(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha);
Tensor& addcmul_hpu_lazy_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha);
Tensor addcdiv_hpu_lazy(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha);
Tensor& addcdiv_hpu_lazy_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha);
Tensor add_tensor_hpu_lazy(
    const Tensor& self,
    const Tensor& other,
    Scalar alpha);
Tensor add_scalar_hpu_lazy(const Tensor& self, Scalar other, Scalar alpha);
Tensor& add_scalar_hpu_lazy_(Tensor& self, Scalar other, Scalar alpha);
Tensor& add_tensor_hpu_lazy_(Tensor& self, const Tensor& other, Scalar alpha);
Tensor sub_tensor_hpu_lazy(
    const Tensor& self,
    const Tensor& other,
    Scalar alpha);
Tensor& sub_tensor_hpu_lazy_(Tensor& self, const Tensor& other, Scalar alpha);
Tensor sub_scalar_hpu_lazy(const Tensor& self, Scalar other, Scalar alpha);
Tensor& sub_scalar_hpu_lazy_(Tensor& self, Scalar other, Scalar alpha);
Tensor rsub_scalar_hpu_lazy(const Tensor& self, Scalar other, Scalar alpha);
Tensor& mul_tensor_hpu_lazy_(Tensor& self, const Tensor& other);
Tensor& mul_out_hpu_lazy(Tensor& out, const Tensor& self, const Tensor& other);
Tensor mul_tensor_hpu_lazy(const Tensor& self, const Tensor& other);
Tensor mul_scalar_hpu_lazy(const Tensor& self, Scalar other);
Tensor& mul_scalar_hpu_lazy_(Tensor& self, Scalar other);
Tensor where_tensor_hpu_lazy(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other);
Tensor div_tensor_hpu_lazy(const Tensor& self, const Tensor& other);
Tensor& div_tensor_hpu_lazy_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other);
Tensor& div_tensor_hpu_lazy_(Tensor& self, const Tensor& other);
Tensor div_scalar_hpu_lazy(const Tensor& self, Scalar other);
Tensor& div_scalar_hpu_lazy_(Tensor& self, Scalar other);
Tensor pow_tensor_tensor_hpu_lazy(const Tensor& self, const Tensor& other);
Tensor& pow_tensor_tensor_hpu_lazy_(Tensor& self, const Tensor& other);
Tensor pow_tensor_scalar_hpu_lazy(const Tensor& self, Scalar other);
Tensor& pow_tensor_scalar_hpu_lazy_(Tensor& self, Scalar other);
Tensor pow_scalar_tensor_hpu_lazy(Scalar other, const Tensor& self);
Tensor maximum_hpu_lazy(const Tensor& self, const Tensor& other);
Tensor minimum_hpu_lazy(const Tensor& self, const Tensor& other);
Tensor gt_tensor_hpu_lazy(const Tensor& self, const Tensor& other);
Tensor gt_scalar_hpu_lazy(const Tensor& self, Scalar other);
Tensor& eq_tensor_out_hpu_lazy(
    Tensor& output,
    const Tensor& self,
    const Tensor& other);
Tensor ne_scalar_hpu_lazy(const Tensor& self, Scalar other);
Tensor ne_tensor_hpu_lazy(const Tensor& self, const Tensor& other);
Tensor all_hpu_lazy(const Tensor& self);
Tensor all_dim_hpu_lazy(const Tensor& self, int64_t dim, bool keepdim);
Tensor ge_scalar_hpu_lazy(const Tensor& self, Scalar other);
Tensor ge_tensor_hpu_lazy(const Tensor& self, const Tensor& other);
Tensor eq_tensor_hpu_lazy(const Tensor& self, const Tensor& other);
Tensor eq_tensor_scalar_hpu_lazy(const Tensor& self, Scalar other);
Tensor lt_scalar_hpu_lazy(const Tensor& self, Scalar other);
Tensor lt_tensor_hpu_lazy(const Tensor& self, const Tensor& other);
Tensor upsample_nearest2d_hpu_lazy(
    const Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    c10::optional<at::ArrayRef<double>> scale_factors);
Tensor upsample_nearest2d_backward_hpu_lazy(
    const Tensor& grad_output,
    c10::optional<at::IntArrayRef> output_size,
    at::IntArrayRef input_size,
    c10::optional<at::ArrayRef<double>> scale_factors);
Tensor convolution_hpu_lazy(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups);
std::tuple<Tensor, Tensor, Tensor> convolution_backward_hpu_lazy(
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
Tensor constant_pad_hpu_lazy(const Tensor& self, IntArrayRef pad, Scalar value);
Tensor embedding_hpu_lazy(
    const Tensor& weight,
    const Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse);
Tensor embedding_dense_backward_hpu_lazy(
    const Tensor& grad,
    const Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq);
Tensor embedding_bag_sum_hpu_lazy(
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode);
Tensor& embedding_bag_sum_bwd_out_kernel_mode_hpu_lazy(
    Tensor& out,
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode);
Tensor embedding_bag_sum_fwd_hpu_lazy(
    const Tensor& input,
    const Tensor& indices_fwd,
    const Tensor& offsets_fwd,
    const Tensor& valid_count,
    const Tensor& indices_bwd,
    const Tensor& offsets_bwd,
    const Tensor& valid_count_bwd,
    const Tensor& grad_weight);
Tensor& embedding_bag_sum_bwd_out_hpu_lazy(
    Tensor& out,
    const Tensor& input,
    const Tensor& indices_bwd,
    const Tensor& offsets_bwd,
    const Tensor& valid_count_bwd);
Tensor& fill_hpu_lazy_(Tensor& self, Scalar value);
Tensor& masked_fill_hpu_lazy_(
    Tensor& self,
    const Tensor& mask,
    const Tensor& value);
Tensor& masked_fill_scalar_hpu_lazy_(
    Tensor& self,
    const Tensor& mask,
    Scalar value);
Tensor gather_src_hpu_lazy(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    bool sparse_grad);
Tensor& scatter_inplace_src_hpu_lazy(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src);
Tensor scatter_src_hpu_lazy(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src);
Tensor& scatter_inplace_value_hpu_lazy(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    Scalar value);
Tensor scatter_add_src_hpu_lazy(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src);
Tensor& scatter_add_inplace_src_hpu_lazy(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src);
Tensor index_hpu_lazy(const Tensor& self, TensorList indices);
Tensor& index_add_hpu_lazy_(
    Tensor& self,
    int64_t dim_,
    const Tensor& indices,
    const Tensor& source);
Tensor index_put_hpu_lazy(
    const Tensor& self,
    TensorList indices,
    const Tensor& value,
    bool accumulate);
Tensor& index_put_hpu_lazy_(
    Tensor& self,
    TensorList indices,
    const Tensor& value,
    bool accumulate);
Tensor index_select_hpu_lazy(
    const Tensor& self,
    int64_t dim,
    const Tensor& index);
Tensor gather2d_hpu_lazy(
    const Tensor& input,
    const Tensor& indices,
    int64_t validCount);
Tensor slice_hpu_lazy(
    const Tensor& self,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t step);
Tensor select_hpu_lazy(const Tensor& self, int64_t dim, int64_t index);
Tensor& arange_hpu_lazy(Tensor& output, Scalar start, Scalar end, Scalar step);
at::Tensor mm_hpu_lazy(const at::Tensor& mat1, const at::Tensor& mat2);
Tensor addmm_hpu_lazy(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha);
Tensor& batch_gemm_out_hpu_lazy(
    Tensor& out,
    const Tensor& self,
    const Tensor& mat2);
Tensor batch_gemm_hpu_lazy(const Tensor& self, const Tensor& mat2);
Tensor dot_hpu_lazy(const Tensor& self, const Tensor& other);
Tensor mv_hpu_lazy(const Tensor& self, const Tensor& other);
at::Tensor one_hot_hpu_lazy(const at::Tensor& self, int64_t num_classes);
std::tuple<Tensor, Tensor> nll_loss_forward_hpu_lazy(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index);
Tensor nll_loss_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    UNUSED const Tensor& total_weight);
Tensor mse_loss_forward_hpu_lazy(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction);
Tensor mse_loss_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction);
Tensor binary_cross_entropy_hpu_lazy(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction);
Tensor binary_cross_entropy_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction);
Tensor binary_cross_entropy_with_logits_hpu_lazy(
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight,
    const c10::optional<Tensor>& pos_weight,
    int64_t reduction);
std::tuple<Tensor, Tensor, Tensor> batch_norm_hpu_lazy(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool training,
    double momentum,
    double eps);
std::tuple<Tensor, Tensor, Tensor> batch_norm_bwd_hpu_lazy(
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& save_mean,
    const Tensor& save_invstd,
    bool train,
    double eps,
    std::array<bool, 3> output_mask);
std::tuple<Tensor, Tensor, Tensor> layer_norm_hpu_lazy(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int64_t m,
    int64_t n,
    double eps);
std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_hpu_lazy(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    std::array<bool, 3> grad_input_mask);
Tensor norm_scalar_hpu_lazy(const Tensor& self, Scalar p);
std::tuple<Tensor, Tensor> max_pool2d_with_indices_hpu_lazy(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode);
Tensor& max_pool2d_with_indices_backward_out_hpu_lazy(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& indices,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode);
Tensor max_pool2d_with_indices_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices);
Tensor avg_pool2d_hpu_lazy(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);
Tensor& avg_pool2d_backward_out_hpu_lazy(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);
Tensor avg_pool2d_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);
Tensor& uniform_hpu_lazy(
    Tensor& self,
    double from = 0,
    double to = 1,
    c10::optional<Generator> gen = c10::nullopt);
Tensor& normal_hpu_lazy(
    Tensor& self,
    double mean = 0,
    double std = 1,
    c10::optional<Generator> gen = c10::nullopt);
Tensor bernoulli_hpu_lazy(
    const Tensor& self,
    c10::optional<Generator> gen = c10::nullopt);
Tensor& bernoulli_scalar_hpu_lazy(
    Tensor& self,
    double p,
    c10::optional<Generator> gen = c10::nullopt);
std::tuple<Tensor, Tensor> fused_dropout_hpu_lazy(
    const Tensor& self,
    double p,
    c10::optional<Generator> gen = c10::nullopt);
at::Tensor repeat_hpu_lazy(const at::Tensor& self, at::IntArrayRef repeats);
Tensor sum_dim_IntList_hpu_lazy(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype);
Tensor& sum_IntList_out_hpu_lazy(
    Tensor& output,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype);
Tensor mean_dim_hpu_lazy(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype);
Tensor& mean_dim_out_hpu_lazy(
    Tensor& output,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype);
Tensor sum_hpu_lazy(const Tensor& self, c10::optional<ScalarType> dtype);
Tensor mean_hpu_lazy(const Tensor& self, c10::optional<ScalarType> dtype);
Tensor prod_dim_hpu_lazy(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    c10::optional<ScalarType> dtype);
Tensor prod_hpu_lazy(const Tensor& self, c10::optional<ScalarType> dtype);
std::tuple<at::Tensor, at::Tensor> max_dim_hpu_lazy(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim);
at::Tensor max_hpu_lazy(const at::Tensor& self);
Tensor& any_dim_out_hpu_lazy(
    Tensor& output,
    const Tensor& self,
    int64_t dim,
    bool keepdim);
Tensor any_dim_hpu_lazy(const Tensor& self, int64_t dim, bool keepdim);
Tensor any_hpu_lazy(const Tensor& self);
Tensor argmax_hpu_lazy(
    const Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim);
Tensor& bitwise_and_out_hpu_lazy(
    Tensor& out,
    const Tensor& self,
    const Tensor& other);
Tensor& bitwise_and_out_hpu_lazy(Tensor& out, const Tensor& self, Scalar other);
namespace habana {
Tensor log_softmax_hpu_lazy(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float);
Tensor log_softmax_backward_hpu_lazy(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    const Tensor& input);
Tensor softmax_hpu_lazy(
    const Tensor& self,
    int64_t dim,
    const bool half_to_float);
Tensor softmax_backward_hpu_lazy(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    const Tensor& input);
} // namespace habana

namespace at {
namespace native {
Tensor empty_hpu_lazy(
    IntArrayRef size,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format,
    bool create_storage = true);
Tensor empty_strided_hpu_lazy(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options,
    bool create_storage = true);
} // namespace native
} // namespace at

Tensor clone_hpu_lazy(
    const Tensor& self,
    c10::optional<MemoryFormat> memory_format);
Tensor& zero_hpu_lazy(Tensor& self);
Tensor cat_hpu_lazy(const TensorList tensors, int64_t dim_ = 0);
Tensor& cat_hpu_lazy_out(
    Tensor& result,
    const TensorList tensors,
    int64_t dim_ = 0);
Tensor transpose_hpu_lazy(const Tensor& self, int64_t dim0_, int64_t dim1_);
Tensor& transpose_hpu_lazy_(Tensor& self, int64_t dim0_, int64_t dim1_);
Tensor t_hpu_lazy(const Tensor& self);
Tensor& t_hpu_lazy_(Tensor& self);
Tensor permute_hpu_lazy(const Tensor& self, IntArrayRef dims_);
Tensor permute_cl_hpu_lazy(const Tensor& self, IntArrayRef dims_);
Tensor expand_hpu_lazy(const Tensor& self, IntArrayRef size, bool implicit);
std::vector<Tensor> split_with_sizes_hpu_lazy(
    const Tensor& self,
    IntArrayRef split_sizes,
    int64_t dim);
Tensor threshold_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    Scalar threshold);
std::tuple<Tensor&, Tensor&> topk_out_hpu_lazy(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool largest,
    bool sorted);
std::tuple<Tensor, Tensor> topk_hpu_lazy(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted);
std::tuple<Tensor, Tensor> sort_hpu_lazy(
    const Tensor& self,
    int64_t dim,
    bool descending);
Tensor unary_op_hpu_lazy(
    const Tensor& input,
    std::string& node_type,
    UnaryOperator* Op);
Tensor unary_backward_op_hpu_lazy(
    const Tensor& grad_in,
    const Tensor& input,
    std::string& node_type,
    UnaryBackwardOperator* Op);
Tensor relu_hpu_lazy(const Tensor& input);
Tensor& relu_hpu_lazy_(Tensor& self);
at::Tensor& leaky_relu_lazy_(at::Tensor& self, at::Scalar negative_slope);
at::Tensor leaky_relu_backward_lazy(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Scalar negative_slope,
    bool self_is_result);
at::Tensor leaky_relu_lazy(const at::Tensor& self, at::Scalar negative_slope);
Tensor leaky_relu_backward(
    const Tensor& grad_output,
    const Tensor& self,
    Scalar negative_slope,
    bool self_is_result);

Tensor sigmoid_hpu_lazy(const Tensor& input);
Tensor sigmoid_backward_hpu_lazy(const Tensor& grad_in, const Tensor& input);
Tensor sqrt_hpu_lazy(const Tensor& input);
Tensor sqrt_hpu_lazy_(Tensor& input);
Tensor tanh_hpu_lazy(const Tensor& input);
Tensor& tanh_hpu_lazy_(Tensor& self);
Tensor& tanh_out_hpu_lazy(Tensor& out, const Tensor& self);
Tensor tanh_backward_hpu_lazy(const Tensor& grad_in, const Tensor& input);
Tensor gelu_hpu_lazy(const Tensor& self);
Tensor gelu_backward_hpu_lazy(const Tensor& grad, const Tensor& self);
Tensor& erf_hpu_lazy_(Tensor& self);
Tensor erf_hpu_lazy(const Tensor& self);
Tensor& exp_hpu_lazy_(Tensor& self);
Tensor exp_hpu_lazy(const Tensor& self);
Tensor& neg_out_hpu_lazy(Tensor& result, const Tensor& input);
Tensor& reciprocal_hpu_lazy_(Tensor& self);
Tensor reciprocal_hpu_lazy(const Tensor& self);
Tensor& reciprocal_out_hpu_lazy(Tensor& result, const Tensor& self);
Tensor clamp_min_hpu_lazy(const Tensor& self, Scalar min);
Tensor sign_hpu_lazy(const Tensor& input);
Tensor& sign_hpu_lazy_(Tensor& self);
Tensor sgn_hpu_lazy(const Tensor& input);
Tensor& sgn_hpu_lazy_(Tensor& self);
Tensor floor_hpu_lazy(const Tensor& input);
Tensor& floor_hpu_lazy_(Tensor& self);
Tensor log_hpu_lazy(const Tensor& input);
Tensor& log_hpu_lazy_(Tensor& self);
Tensor log2_hpu_lazy(const Tensor& input);
Tensor& log2_hpu_lazy_(Tensor& self);
Tensor& clamp_hpu_lazy_(
    Tensor& self,
    c10::optional<Scalar> min,
    c10::optional<Scalar> max);
Tensor clamp_hpu_lazy(
    const Tensor& self,
    c10::optional<Scalar> min,
    c10::optional<Scalar> max);
Tensor abs_hpu_lazy(const Tensor& input);
Tensor& abs_hpu_lazy_(Tensor& self);
Tensor round_hpu_lazy(const Tensor& self);
Tensor& round_hpu_lazy_(Tensor& self);
Tensor rsqrt_hpu_lazy(const Tensor& self);
Tensor& rsqrt_hpu_lazy_(Tensor& self);
Tensor isfinite_hpu_lazy(const Tensor& self);
Tensor neg_hpu_lazy(const Tensor& self);
namespace at {
namespace native {
Scalar _local_scalar_dense_hpu_lazy(const Tensor& self);
}
} // namespace at
std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_sgd_with_valid_count_hpu_lazy(
    const Tensor& gradients,
    Tensor& weights_in,
    Tensor& moments_in,
    const Tensor& indices,
    const Tensor& learning_rate,
    const Tensor& valid_count_tensor,
    float mom,
    bool nesterov);
std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_adagrad_with_valid_count_hpu_lazy(
    const Tensor& gradients,
    Tensor& weights_in,
    Tensor& moments_in,
    const Tensor& indices,
    const Tensor& learning_rate,
    const Tensor& valid_count_tensor);
void optimizer_adamw_hpu_lazy(
    const TensorList& gradient_vec,
    TensorList& weight_vec,
    TensorList& exp_avg_vec,
    TensorList& exp_avg_sq_vec,
    Tensor& lr_t,
    Tensor& neg_step_t,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weight_decay);
Tensor fused_norm_hpu_lazy(
    std::vector<Tensor>& grad,
    const Tensor& max_norm,
    float norm_type = 2.0);
Tensor& optimizer_adagrad_hpu_lazy(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& variances,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float lrd,
    const float epsilon);
Tensor& optimizer_sgd_hpu_lazy(
    const TensorList& gradients,
    TensorList& weights,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov);
Tensor& optimizer_sgd_momentum_hpu_lazy(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& momentum,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov);
Tensor ones_like_hpu_lazy(
    const Tensor& self,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> memory_format);
Tensor masked_scale_hpu_lazy(
    const Tensor& self,
    const Tensor& mask,
    double scale);
