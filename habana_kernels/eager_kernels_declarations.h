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
#include <torch/script.h>

using OptionalIntArrayRef = at::OptionalIntArrayRef;

at::Tensor& copy_hpu_(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking);
at::Tensor as_strided_hpu(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset);
at::Tensor& set_hpu_(
    at::Tensor& self,
    at::Storage source,
    int64_t storage_offset,
    at::IntArrayRef size,
    at::IntArrayRef stride);
at::Tensor view_hpu(const at::Tensor& self, at::IntArrayRef size);
at::Tensor addcmul_hpu(
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Scalar alpha);
at::Tensor& addcmul_hpu_(
    at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Scalar alpha);
at::Tensor addcdiv_hpu(
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& alpha);
at::Tensor& addcdiv_hpu_(
    at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& alpha);
at::Tensor add_tensor_hpu(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha);
at::Tensor add_scalar_hpu(
    const at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha);
at::Tensor& add_scalar_hpu_(
    at::Tensor& self,
    at::Scalar other,
    at::Scalar alpha);
at::Tensor& add_tensor_hpu_(
    at::Tensor& self,
    const at::Tensor& other,
    at::Scalar alpha);
at::Tensor sub_tensor_hpu(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha);
at::Tensor& sub_tensor_hpu_(
    at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha);
at::Tensor sub_scalar_hpu(
    const at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha);
at::Tensor& sub_scalar_hpu_(
    at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha);
at::Tensor rsub_scalar_hpu(
    const at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha);
at::Tensor& mul_tensor_hpu_(at::Tensor& self, const at::Tensor& other);
at::Tensor mul_tensor_hpu(const at::Tensor& self, const at::Tensor& other);
at::Tensor where_tensor_hpu(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other);
at::Tensor& mul_out_hpu(
    at::Tensor& out,
    const at::Tensor& self,
    const at::Tensor& other);
at::Tensor mul_scalar_hpu(const at::Tensor& self, const at::Scalar& other);
at::Tensor& mul_scalar_hpu_(at::Tensor& self, const at::Scalar& other);
at::Tensor div_tensor_hpu(const at::Tensor& self, const at::Tensor& other);
at::Tensor& div_tensor_hpu_out(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other);
at::Tensor& div_tensor_hpu_(at::Tensor& self, const at::Tensor& other);
at::Tensor div_scalar_hpu(const at::Tensor& self, const at::Scalar& other);
at::Tensor& div_scalar_hpu_(at::Tensor& self, const at::Scalar& other);
at::Tensor pow_tensor_tensor_hpu(
    const at::Tensor& self,
    const at::Tensor& other);
at::Tensor& pow_tensor_tensor_hpu_(at::Tensor& self, const at::Tensor& other);
at::Tensor pow_tensor_scalar_hpu(
    const at::Tensor& self,
    const at::Scalar& other);
at::Tensor random_shuffle_tensor_hpu(
    const at::Tensor& self,
    const at::Tensor& seed);
at::Tensor& pow_tensor_scalar_hpu_(at::Tensor& self, at::Scalar other);
at::Tensor pow_scalar_tensor_hpu(at::Scalar other, const at::Tensor& self);
at::Tensor maximum_hpu(const at::Tensor& self, const at::Tensor& other);
at::Tensor minimum_hpu(const at::Tensor& self, const at::Tensor& other);
at::Tensor ge_scalar_hpu(const at::Tensor& self, const at::Scalar& other);
at::Tensor ge_tensor_hpu(const at::Tensor& self, const at::Tensor& other);
at::Tensor le_scalar_hpu(const at::Tensor& self, const at::Scalar& other);
at::Tensor le_tensor_hpu(const at::Tensor& self, const at::Tensor& other);
at::Tensor ne_scalar_hpu(const at::Tensor& self, at::Scalar other);
at::Tensor ne_tensor_hpu(const at::Tensor& self, const at::Tensor& other);
at::Tensor all_hpu(const at::Tensor& self);
at::Tensor all_dim_hpu(const at::Tensor& self, int64_t dim, bool keepdim);
at::Tensor gt_tensor_hpu(const at::Tensor& self, const at::Tensor& other);
at::Tensor gt_scalar_hpu(const at::Tensor& self, at::Scalar other);
at::Tensor& eq_tensor_out_hpu(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& other);
at::Tensor eq_tensor_hpu(const at::Tensor& self, const at::Tensor& other);
at::Tensor eq_tensor_scalar_hpu(
    const at::Tensor& self,
    const at::Scalar& other);
at::Tensor lt_scalar_hpu(const at::Tensor& self, at::Scalar other);
at::Tensor lt_tensor_hpu(const at::Tensor& self, const at::Tensor& other);
at::Tensor remainder_tensor_hpu(
    const at::Tensor& self,
    const at::Tensor& other);
at::Tensor& remainder_tensor_hpu_(at::Tensor& self, const at::Tensor& other);
at::Tensor remainder_scalar_hpu(
    const at::Tensor& self,
    const at::Scalar& other);
at::Tensor& remainder_scalar_hpu_(at::Tensor& self, const at::Scalar& other);
at::Tensor& remainder_tensor_hpu_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result);
at::Tensor& remainder_scalar_hpu_out(
    const at::Tensor& self,
    at::Scalar other,
    at::Tensor& result);
at::Tensor upsample_nearest2d_hpu(
    const at::Tensor& input,
    OptionalIntArrayRef output_size,
    c10::optional<at::ArrayRef<double>> scale_factors);
at::Tensor upsample_nearest2d_backward_hpu(
    const at::Tensor& grad_output,
    OptionalIntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<at::ArrayRef<double>> scale_factors);
at::Tensor upsample_nearest3d_hpu(
    const at::Tensor& input,
    OptionalIntArrayRef output_size,
    c10::optional<at::ArrayRef<double>> scale_factors);
at::Tensor upsample_nearest3d_backward_hpu(
    const at::Tensor& grad_output,
    OptionalIntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<at::ArrayRef<double>> scale_factors);
at::Tensor convolution_hpu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups);
std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward_hpu(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask);
at::Tensor constant_pad_hpu(
    const at::Tensor& self,
    at::IntArrayRef pad,
    at::Scalar value);
at::Tensor embedding_hpu(
    const at::Tensor& weight,
    const at::Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse);
at::Tensor embedding_dense_backward_hpu(
    const at::Tensor& grad,
    const at::Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq);
at::Tensor embedding_bag_sum_hpu(
    const at::Tensor& input,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    const at::Tensor& valid_count,
    int64_t kernel_mode);
at::Tensor& embedding_bag_sum_bwd_out_kernel_mode_hpu(
    at::Tensor& out,
    const at::Tensor& input,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    const at::Tensor& valid_count,
    int64_t kernel_mode);
at::Tensor embedding_bag_sum_fwd_hpu(
    const at::Tensor& input,
    const at::Tensor& indices_fwd,
    const at::Tensor& offsets_fwd,
    const at::Tensor& valid_count,
    const at::Tensor& indices_bwd,
    const at::Tensor& offsets_bwd,
    const at::Tensor& valid_count_bwd,
    const at::Tensor& grad_weight);
at::Tensor& embedding_bag_sum_bwd_out_hpu(
    at::Tensor& out,
    const at::Tensor& input,
    const at::Tensor& indices_bwd,
    const at::Tensor& offsets_bwd,
    const at::Tensor& valid_count_bwd);
at::Tensor& fill_hpu_(at::Tensor& self, const at::Scalar& value);
at::Tensor& masked_fill_hpu_(
    at::Tensor& self,
    const at::Tensor& mask,
    const at::Tensor& value);
at::Tensor& masked_fill_scalar_hpu_(
    at::Tensor& self,
    const at::Tensor& mask,
    const at::Scalar& value);
at::Tensor nonzero_hpu(const at::Tensor& self);
at::Tensor gather_src_hpu(
    const at::Tensor& self,
    int64_t dim_,
    const at::Tensor& index,
    bool sparse_grad);
at::Tensor& scatter_inplace_src_hpu(
    at::Tensor& self,
    int64_t dim_,
    const at::Tensor& index,
    const at::Tensor& src);
at::Tensor& scatter_inplace_value_hpu(
    at::Tensor& self,
    int64_t dim_,
    const at::Tensor& index,
    const at::Scalar& value);
at::Tensor scatter_src_hpu(
    const at::Tensor& self,
    int64_t dim_,
    const at::Tensor& index,
    const at::Tensor& src);
at::Tensor scatter_value_hpu(
    const at::Tensor& self,
    int64_t dim_,
    const at::Tensor& index,
    at::Scalar value);
at::Tensor scatter_add_src_hpu(
    const at::Tensor& self,
    int64_t dim_,
    const at::Tensor& index,
    const at::Tensor& src);
at::Tensor& scatter_add_inplace_src_hpu(
    at::Tensor& self,
    int64_t dim_,
    const at::Tensor& index,
    const at::Tensor& src);
at::Tensor& index_add_hpu_(
    at::Tensor& self,
    int64_t dim_,
    const at::Tensor& indices,
    const at::Tensor& source);
at::Tensor index_put_hpu(
    const at::Tensor& self,
    at::TensorList indices,
    const at::Tensor& value,
    bool accumulate);
at::Tensor& index_put_hpu_(
    at::Tensor& self,
    at::TensorList indices,
    const at::Tensor& value,
    bool accumulate);
at::Tensor index_select_hpu(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index);
at::Tensor index_hpu(const at::Tensor& self, at::TensorList indices);
at::Tensor gather2d_hpu(
    const at::Tensor& input,
    const at::Tensor& indices,
    int64_t validCount);
at::Tensor slice_hpu(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<int64_t> start,
    c10::optional<int64_t> end,
    int64_t step);
at::Tensor select_hpu(const at::Tensor& self, int64_t dim, int64_t index);
at::Tensor& arange_hpu(
    at::Tensor& output,
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step);
at::Tensor mm_hpu(const at::Tensor& mat1, const at::Tensor& mat2);
at::Tensor addmm_hpu(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    at::Scalar beta,
    at::Scalar alpha);
at::Tensor& batch_gemm_out_hpu(
    at::Tensor& out,
    const at::Tensor& self,
    const at::Tensor& mat2);
at::Tensor batch_gemm_hpu(const at::Tensor& self, const at::Tensor& mat2);
at::Tensor dot_hpu(const at::Tensor& self, const at::Tensor& other);
at::Tensor mv_hpu(const at::Tensor& self, const at::Tensor& other);
std::tuple<at::Tensor, at::Tensor> nll_loss_forward_hpu(
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t reduction,
    int64_t ignore_index);
std::tuple<at::Tensor, at::Tensor> nll_loss2d_forward_hpu(
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t reduction,
    int64_t ignore_index);
at::Tensor nll_loss_backward_hpu(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight);
at::Tensor nll_loss2d_backward_hpu(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight);
at::Tensor mse_loss_forward_hpu(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction);
at::Tensor mse_loss_backward_hpu(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction);
at::Tensor binary_cross_entropy_hpu(
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t reduction);
at::Tensor binary_cross_entropy_backward_hpu(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t reduction);
at::Tensor binary_cross_entropy_with_logits_hpu(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& pos_weight,
    int64_t reduction);
at::Tensor kl_div_hpu(
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction,
    bool log_target);
at::Tensor kl_div_backward_hpu(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction,
    bool log_target);
std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_hpu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool training,
    double momentum,
    double eps);
std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_bwd_hpu(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const at::Tensor& save_mean,
    const at::Tensor& save_invstd,
    bool train,
    double eps,
    std::array<bool, 3> output_mask);
std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_hpu(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    double eps);
std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_backward_hpu(
    const at::Tensor& dY,
    const at::Tensor& X,
    at::IntArrayRef normalized_shape,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    std::array<bool, 3> grad_input_mask);
at::Tensor norm_scalar_hpu(const at::Tensor& self, at::Scalar p);
std::tuple<at::Tensor, at::Tensor> max_pool2d_with_indices_hpu(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode);
at::Tensor& max_pool2d_with_indices_backward_out_hpu(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& indices,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode);
at::Tensor max_pool2d_with_indices_backward_hpu(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor& indices);
at::Tensor avg_pool2d_hpu(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);
at::Tensor& avg_pool2d_backward_out_hpu(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);
at::Tensor avg_pool2d_backward_hpu(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);
at::Tensor adaptive_avg_pool2d_hpu(
    const at::Tensor& input,
    at::IntArrayRef output_size);
at::Tensor adaptive_avg_pool2d_backward_hpu(
    const at::Tensor& grad_output,
    const at::Tensor& input);
at::Tensor& uniform_hpu(
    at::Tensor& self,
    double from = 0,
    double to = 1,
    c10::optional<at::Generator> gen = c10::nullopt);
at::Tensor& normal_hpu(
    at::Tensor& self,
    double mean = 0,
    double std = 1,
    c10::optional<at::Generator> gen = c10::nullopt);
at::Tensor bernoulli_hpu(
    const at::Tensor& self,
    c10::optional<at::Generator> gen = c10::nullopt);
at::Tensor& bernoulli_scalar_hpu(
    at::Tensor& self,
    double p,
    c10::optional<at::Generator> gen = c10::nullopt);
at::Tensor& randperm_hpu(
    at::Tensor& out,
    int64_t n,
    c10::optional<at::Generator> gen = c10::nullopt);
std::tuple<at::Tensor, at::Tensor> fused_dropout_hpu(
    const at::Tensor& self,
    double p,
    c10::optional<at::Generator> gen = c10::nullopt);
at::Tensor sum_dim_IntList_hpu(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype);
at::Tensor mean_dim_hpu(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype);
at::Tensor sum_hpu(const at::Tensor& self, c10::optional<at::ScalarType> dtype);
at::Tensor mean_hpu(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype);
at::Tensor prod_dim_hpu(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype);
std::tuple<at::Tensor, at::Tensor, at::Tensor> unique2_hpu(
    const at::Tensor& self,
    bool sorted,
    bool return_inverse,
    bool return_counts);
std::tuple<at::Tensor, at::Tensor> max_dim_hpu(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim);
at::Tensor max_hpu(const at::Tensor& self);
at::Tensor min_hpu(const at::Tensor& self);
at::Tensor& any_dim_out_hpu(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    at::Tensor& output);
at::Tensor any_dim_hpu(const at::Tensor& self, int64_t dim, bool keepdim);
at::Tensor any_hpu(const at::Tensor& self);
at::Tensor& bitwise_and_out_hpu(
    at::Tensor& out,
    const at::Tensor& self,
    at::Scalar other);
at::Tensor& bitwise_and_out_hpu(
    at::Tensor& out,
    const at::Tensor& self,
    const at::Tensor& other);
at::Tensor& bitwise_or_out_hpu(
    at::Tensor& out,
    const at::Tensor& self,
    const at::Tensor& other);
at::Tensor& bitwise_xor_out_hpu(
    at::Tensor& out,
    const at::Tensor& self,
    const at::Tensor& other);
at::Tensor& bitwise_not_out_hpu(at::Tensor& out, const at::Tensor& self);
at::Tensor log_softmax_hpu(
    const at::Tensor& self,
    const int64_t dim,
    const bool half_to_float);
at::Tensor log_softmax_backward_hpu(
    const at::Tensor& grad,
    const at::Tensor& output,
    int64_t dim,
    const at::Tensor& input);
at::Tensor softmax_hpu(
    const at::Tensor& self,
    int64_t dim,
    const bool half_to_float);
at::Tensor softmax_backward_hpu(
    const at::Tensor& grad,
    const at::Tensor& output,
    int64_t dim,
    const at::Tensor& input);
at::Tensor empty_hpu(
    at::IntArrayRef size,
    const at::TensorOptions& options,
    c10::optional<at::MemoryFormat> optional_memory_format);
at::Tensor empty_strided_hpu(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    const at::TensorOptions& options);

at::Tensor clone_hpu(
    const at::Tensor& self,
    c10::optional<at::MemoryFormat> memory_format);
at::Tensor& zero_hpu(at::Tensor& self);
at::Tensor cat_hpu(const at::TensorList tensors, int64_t dim_ = 0);
at::Tensor& cat_hpu_out(
    at::Tensor& result,
    const at::TensorList tensors,
    int64_t dim_ = 0);
at::Tensor transpose_hpu(const at::Tensor& self, int64_t dim0_, int64_t dim1_);
at::Tensor t_hpu(const at::Tensor& self);
at::Tensor permute_hpu(const at::Tensor& self, at::IntArrayRef dims_);
at::Tensor expand_hpu(
    const at::Tensor& self,
    at::IntArrayRef size,
    bool implicit);
std::vector<at::Tensor> split_with_sizes_hpu(
    const at::Tensor& self,
    at::IntArrayRef split_sizes,
    int64_t dim);
at::Tensor threshold_backward_hpu(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Scalar threshold);
std::tuple<at::Tensor&, at::Tensor&> topk_out_hpu(
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t k,
    int64_t dim_,
    bool largest,
    bool sorted);
std::tuple<at::Tensor, at::Tensor> topk_hpu(
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted);
std::tuple<at::Tensor, at::Tensor> sort_hpu(
    const at::Tensor& self,
    int64_t dim,
    bool descending);
at::Tensor relu_hpu(const at::Tensor& input);
at::Tensor& relu_hpu_(at::Tensor& self);
at::Tensor& leaky_relu_hpu_(at::Tensor& self, const at::Scalar& negative_slope);
at::Tensor leaky_relu_hpu(const at::Tensor& self, at::Scalar negative_slope);
at::Tensor leaky_relu_backward_hpu(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& negative_slope,
    bool self_is_result);
at::Tensor sigmoid_hpu(const at::Tensor& input);
at::Tensor sigmoid_backward_hpu(
    const at::Tensor& grad_in,
    const at::Tensor& input);
at::Tensor sqrt_hpu(const at::Tensor& input);
at::Tensor tanh_hpu(const at::Tensor& input);
at::Tensor& tanh_hpu_(at::Tensor& self);
at::Tensor& tanh_out_hpu(at::Tensor& out, const at::Tensor& self);
at::Tensor tanh_backward_hpu(
    const at::Tensor& grad_in,
    const at::Tensor& input);
at::Tensor gelu_hpu(const at::Tensor& self);
at::Tensor gelu_backward_hpu(const at::Tensor& grad, const at::Tensor& self);
std::tuple<at::Tensor, at::Tensor> gelu2_hpu(const at::Tensor& self);
at::Tensor gelu2_backward_hpu(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& saved);

at::Tensor& idop_hpu_(at::Tensor& self);
at::Tensor idop_hpu(const at::Tensor& self);

at::Tensor& erf_hpu_(at::Tensor& self);
at::Tensor erf_hpu(const at::Tensor& self);
at::Tensor& exp_hpu_(at::Tensor& self);
at::Tensor exp_hpu(const at::Tensor& self);
at::Tensor& neg_out_hpu(at::Tensor& result, const at::Tensor& input);
at::Tensor& reciprocal_hpu_(at::Tensor& self);
at::Tensor reciprocal_hpu(const at::Tensor& self);
at::Tensor& reciprocal_out_hpu(at::Tensor& result, const at::Tensor& self);
at::Tensor clamp_min_hpu(const at::Tensor& self, at::Scalar min);
at::Tensor floor_hpu(const at::Tensor& input);
at::Tensor& floor_hpu_(at::Tensor& self);
at::Tensor log_hpu(const at::Tensor& input);
at::Tensor& log_hpu_(at::Tensor& self);
at::Tensor log2_hpu(const at::Tensor& input);
at::Tensor& log2_hpu_(at::Tensor& self);
at::Tensor& clamp_hpu_(
    at::Tensor& self,
    const c10::optional<at::Scalar>& min,
    const c10::optional<at::Scalar>& max);
at::Tensor clamp_hpu(
    const at::Tensor& self,
    const c10::optional<at::Scalar>& min,
    const c10::optional<at::Scalar>& max);
at::Tensor abs_hpu(const at::Tensor& self);
at::Tensor& abs_hpu_(at::Tensor& self);
at::Tensor round_hpu(const at::Tensor& self);
at::Tensor& round_hpu_(at::Tensor& self);
at::Tensor rsqrt_hpu(const at::Tensor& self);
at::Tensor& rsqrt_hpu_(at::Tensor& self);
at::Tensor isfinite_hpu(const at::Tensor& self);
at::Tensor isnan_hpu(const at::Tensor& self);
at::Tensor neg_hpu(const at::Tensor& self);
at::Tensor sin_hpu(const at::Tensor& self);
at::Tensor cos_hpu(const at::Tensor& self);
at::Tensor argmax_hpu(
    const at::Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim);
at::Scalar _local_scalar_dense_hpu(const at::Tensor& self);
std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_sgd_with_valid_count_hpu(
    const at::Tensor& gradients,
    at::Tensor& weights_in,
    at::Tensor& moments_in,
    const at::Tensor& indices,
    const at::Tensor& learning_rate,
    const at::Tensor& valid_count_tensor,
    float mom,
    bool nesterov);
std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_adagrad_with_valid_count_hpu(
    const at::Tensor& gradients,
    at::Tensor& weights_in,
    at::Tensor& moments_in,
    const at::Tensor& indices,
    const at::Tensor& learning_rate,
    const at::Tensor& valid_count_tensor);
at::Tensor ones_like_hpu(
    const at::Tensor& self,
    const at::TensorOptions& options,
    c10::optional<c10::MemoryFormat> optional_memory_format);
at::Tensor matmul_hpu(const at::Tensor& tensor1, const at::Tensor& tensor2);
std::tuple<at::Tensor, at::Tensor> matmul_backward_hpu(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& other);
at::Tensor masked_scale_hpu(
    const at::Tensor& self,
    const at::Tensor& mask,
    double scale);
void optimizer_adamw_hpu(
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
at::Tensor fused_norm_hpu(
    std::vector<at::Tensor>& grad,
    const at::Tensor& max_norm,
    float norm_type = 2.0);

void optimizer_lamb_phase1_hpu(
    const std::vector<at::Tensor>& gradient_vec,
    std::vector<at::Tensor>& hl_adam_step_vec,
    std::vector<at::Tensor>& hl_adam_norm_vec,
    std::vector<at::Tensor>& hl_weight_norm_vec,
    std::vector<at::Tensor>& weight_vec,
    std::vector<at::Tensor>& exp_avg_vec,
    std::vector<at::Tensor>& exp_avg_sq_vec,
    const at::Tensor& clip_global_grad_norm,
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
at::Tensor optimizer_lamb_fused_norm_hpu(
    const std::vector<at::Tensor>& grad,
    float max_grad_norm);
at::Tensor& optimizer_adagrad_hpu(
    const at::TensorList& gradient_vec,
    at::TensorList& weight_vec,
    at::TensorList& variance_vec,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float lrd,
    const float epsilon);
at::Tensor& optimizer_sgd_hpu(
    const at::TensorList& gradient_vec,
    at::TensorList& weight_vec,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov);
at::Tensor& optimizer_sgd_momentum_hpu(
    const at::TensorList& gradient_vec,
    at::TensorList& weight_vec,
    at::TensorList& momentum_vec,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const at::Tensor& mom,
    const float wd,
    const float damp,
    const bool nesterov);
at::Tensor habana_nms_hpu(
    const at::Tensor& scores,
    const at::Tensor& boxes,
    float iou_threshold,
    float score_threshold);
at::Tensor silu_hpu(const at::Tensor& self);
at::Tensor repeat_hpu(const at::Tensor& self, at::IntArrayRef repeats);
at::Tensor cumsum_hpu(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype);
at::Tensor flip_hpu(const at::Tensor& self, at::IntArrayRef dims);
at::Tensor& linspace_out_hpu(
    const at::Scalar& start,
    const at::Scalar& end,
    int64_t steps,
    at::Tensor& out);
at::Tensor diag_hpu(const at::Tensor& self, int64_t diagonal);
at::Tensor& diag_hpu_out(
    const at::Tensor& self,
    int64_t diagonal,
    at::Tensor& result);
bool is_pinned_hpu(const at::Tensor& self, c10::optional<at::Device> device);
at::Tensor pin_memory_hpu(
    const at::Tensor& self,
    c10::optional<at::Device> device);
