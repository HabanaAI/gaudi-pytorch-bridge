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
  at::Tensor op_code##_hpu_lazy(const at::Tensor& self);
#define HPU_LAZY_FUNC_DECL_INPLACE(op_code) \
  at::Tensor& op_code##hpu_lazy_(at::Tensor& self);

namespace habana_lazy {
at::Tensor& copy_hpu_lazy_(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking);
at::Tensor as_strided_hpu_lazy(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset);

at::Tensor asin_hpu_lazy(const at::Tensor& self);
at::Tensor acosh_hpu_lazy(const at::Tensor& self);
at::Tensor asinh_hpu_lazy(const at::Tensor& self);
at::Tensor atan_hpu_lazy(const at::Tensor& self);
at::Tensor atanh_hpu_lazy(const at::Tensor& self);
at::Tensor cosh_hpu_lazy(const at::Tensor& self);

at::Tensor& acosh_hpu_lazy_(at::Tensor& self);
at::Tensor& asinh_hpu_lazy_(at::Tensor& self);
at::Tensor& atan_hpu_lazy_(at::Tensor& self);
at::Tensor& atanh_hpu_lazy_(at::Tensor& self);
at::Tensor& cos_hpu_lazy_(at::Tensor& self);
at::Tensor& cosh_hpu_lazy_(at::Tensor& self);
at::Tensor& tanh_hpu_lazy_(at::Tensor& self);

HPU_LAZY_FUNC_DECL(acos)
HPU_LAZY_FUNC_DECL_INPLACE(acos_)
at::Tensor& set_hpu_lazy_(
    at::Tensor& self,
    at::Storage source,
    int64_t storage_offset,
    at::IntArrayRef size,
    at::IntArrayRef stride);
at::Tensor view_hpu_lazy(const at::Tensor& self, at::IntArrayRef size);
at::Tensor addcmul_hpu_lazy(
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Scalar alpha);
at::Tensor& addcmul_hpu_lazy_(
    at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Scalar alpha);
at::Tensor addcdiv_hpu_lazy(
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Scalar alpha);
at::Tensor& addcdiv_hpu_lazy_(
    at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Scalar alpha);
at::Tensor add_tensor_hpu_lazy(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Scalar alpha);
at::Tensor add_scalar_hpu_lazy(
    const at::Tensor& self,
    at::Scalar other,
    at::Scalar alpha);
at::Tensor& add_scalar_hpu_lazy_(
    at::Tensor& self,
    at::Scalar other,
    at::Scalar alpha);
at::Tensor& add_tensor_hpu_lazy_(
    at::Tensor& self,
    const at::Tensor& other,
    at::Scalar alpha);
at::Tensor sub_tensor_hpu_lazy(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Scalar alpha);
at::Tensor& sub_tensor_hpu_lazy_(
    at::Tensor& self,
    const at::Tensor& other,
    at::Scalar alpha);
at::Tensor sub_scalar_hpu_lazy(
    const at::Tensor& self,
    at::Scalar other,
    at::Scalar alpha);
at::Tensor& sub_scalar_hpu_lazy_(
    at::Tensor& self,
    at::Scalar other,
    at::Scalar alpha);
at::Tensor rsub_scalar_hpu_lazy(
    const at::Tensor& self,
    at::Scalar other,
    at::Scalar alpha);
at::Tensor& mul_tensor_hpu_lazy_(at::Tensor& self, const at::Tensor& other);
at::Tensor& mul_out_hpu_lazy(
    at::Tensor& out,
    const at::Tensor& self,
    const at::Tensor& other);
at::Tensor mul_tensor_hpu_lazy(const at::Tensor& self, const at::Tensor& other);
at::Tensor mul_scalar_hpu_lazy(const at::Tensor& self, at::Scalar other);
at::Tensor& mul_scalar_hpu_lazy_(at::Tensor& self, at::Scalar other);
at::Tensor where_tensor_hpu_lazy(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other);
at::Tensor div_tensor_hpu_lazy(const at::Tensor& self, const at::Tensor& other);
at::Tensor& div_tensor_hpu_lazy_out(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other);
at::Tensor& div_tensor_hpu_lazy_(at::Tensor& self, const at::Tensor& other);
at::Tensor div_scalar_hpu_lazy(const at::Tensor& self, at::Scalar other);
at::Tensor& div_scalar_hpu_lazy_(at::Tensor& self, at::Scalar other);
at::Tensor pow_tensor_tensor_hpu_lazy(
    const at::Tensor& self,
    const at::Tensor& other);
at::Tensor& pow_tensor_tensor_hpu_lazy_(
    at::Tensor& self,
    const at::Tensor& other);
at::Tensor pow_tensor_scalar_hpu_lazy(const at::Tensor& self, at::Scalar other);
at::Tensor& pow_tensor_scalar_hpu_lazy_(at::Tensor& self, at::Scalar other);
at::Tensor pow_scalar_tensor_hpu_lazy(at::Scalar other, const at::Tensor& self);
at::Tensor maximum_hpu_lazy(const at::Tensor& self, const at::Tensor& other);
at::Tensor minimum_hpu_lazy(const at::Tensor& self, const at::Tensor& other);
at::Tensor gt_tensor_hpu_lazy(const at::Tensor& self, const at::Tensor& other);
at::Tensor gt_scalar_hpu_lazy(const at::Tensor& self, at::Scalar other);
at::Tensor& eq_tensor_out_hpu_lazy(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& other);
at::Tensor ne_scalar_hpu_lazy(const at::Tensor& self, at::Scalar other);
at::Tensor ne_tensor_hpu_lazy(const at::Tensor& self, const at::Tensor& other);
at::Tensor all_hpu_lazy(const at::Tensor& self);
at::Tensor all_dim_hpu_lazy(const at::Tensor& self, int64_t dim, bool keepdim);
at::Tensor ge_scalar_hpu_lazy(const at::Tensor& self, at::Scalar other);
at::Tensor ge_tensor_hpu_lazy(const at::Tensor& self, const at::Tensor& other);
at::Tensor eq_tensor_hpu_lazy(const at::Tensor& self, const at::Tensor& other);
at::Tensor eq_tensor_scalar_hpu_lazy(const at::Tensor& self, at::Scalar other);
at::Tensor lt_scalar_hpu_lazy(const at::Tensor& self, at::Scalar other);
at::Tensor lt_tensor_hpu_lazy(const at::Tensor& self, const at::Tensor& other);
at::Tensor upsample_nearest2d_hpu_lazy(
    const at::Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    c10::optional<at::ArrayRef<double>> scale_factors);
at::Tensor upsample_nearest2d_backward_hpu_lazy(
    const at::Tensor& grad_output,
    c10::optional<at::IntArrayRef> output_size,
    at::IntArrayRef input_size,
    c10::optional<at::ArrayRef<double>> scale_factors);
at::Tensor convolution_hpu_lazy(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups);
std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward_hpu_lazy(
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
at::Tensor constant_pad_hpu_lazy(
    const at::Tensor& self,
    at::IntArrayRef pad,
    at::Scalar value);
at::Tensor embedding_hpu_lazy(
    const at::Tensor& weight,
    const at::Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse);
at::Tensor embedding_dense_backward_hpu_lazy(
    const at::Tensor& grad,
    const at::Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq);
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
at::Tensor& fill_hpu_lazy_(at::Tensor& self, at::Scalar value);
at::Tensor& masked_fill_hpu_lazy_(
    at::Tensor& self,
    const at::Tensor& mask,
    const at::Tensor& value);
at::Tensor& masked_fill_scalar_hpu_lazy_(
    at::Tensor& self,
    const at::Tensor& mask,
    at::Scalar value);
at::Tensor gather_src_hpu_lazy(
    const at::Tensor& self,
    int64_t dim_,
    const at::Tensor& index,
    bool sparse_grad);
at::Tensor& scatter_inplace_src_hpu_lazy(
    at::Tensor& self,
    int64_t dim_,
    const at::Tensor& index,
    const at::Tensor& src);
at::Tensor scatter_src_hpu_lazy(
    const at::Tensor& self,
    int64_t dim_,
    const at::Tensor& index,
    const at::Tensor& src);
at::Tensor& scatter_inplace_value_hpu_lazy(
    at::Tensor& self,
    int64_t dim_,
    const at::Tensor& index,
    at::Scalar value);
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
at::Tensor index_hpu_lazy(const at::Tensor& self, at::TensorList indices);
at::Tensor& _index_put_impl_hpu_lazy_(
    at::Tensor& self,
    at::TensorList indices,
    const at::Tensor& value,
    const bool accumulate,
    const bool unsafe);
at::Tensor& index_add_hpu_lazy_(
    at::Tensor& self,
    int64_t dim_,
    const at::Tensor& indices,
    const at::Tensor& source);
at::Tensor index_put_hpu_lazy(
    const at::Tensor& self,
    at::TensorList indices,
    const at::Tensor& value,
    bool accumulate);
at::Tensor& index_put_hpu_lazy_(
    at::Tensor& self,
    at::TensorList indices,
    const at::Tensor& value,
    bool accumulate);
at::Tensor index_select_hpu_lazy(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index);
at::Tensor gather2d_hpu_lazy(
    const at::Tensor& input,
    const at::Tensor& indices,
    int64_t validCount);
at::Tensor slice_hpu_lazy(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<int64_t> start,
    c10::optional<int64_t> end,
    int64_t step);
at::Tensor select_hpu_lazy(const at::Tensor& self, int64_t dim, int64_t index);
at::Tensor& arange_hpu_lazy(
    at::Tensor& output,
    at::Scalar start,
    at::Scalar end,
    at::Scalar step);
at::Tensor nonzero_hpu_lazy(const at::Tensor& self);
at::Tensor mm_hpu_lazy(const at::Tensor& mat1, const at::Tensor& mat2);
at::Tensor addmm_hpu_lazy(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    at::Scalar beta,
    at::Scalar alpha);
at::Tensor& batch_gemm_out_hpu_lazy(
    at::Tensor& out,
    const at::Tensor& self,
    const at::Tensor& mat2);
at::Tensor batch_gemm_hpu_lazy(const at::Tensor& self, const at::Tensor& mat2);
at::Tensor dot_hpu_lazy(const at::Tensor& self, const at::Tensor& other);
at::Tensor mv_hpu_lazy(const at::Tensor& self, const at::Tensor& other);
at::Tensor one_hot_hpu_lazy(const at::Tensor& self, int64_t num_classes);
std::tuple<at::Tensor, at::Tensor> nll_loss_forward_hpu_lazy(
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t reduction,
    int64_t ignore_index);
at::Tensor nll_loss_backward_hpu_lazy(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight);
at::Tensor mse_loss_forward_hpu_lazy(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction);
at::Tensor mse_loss_backward_hpu_lazy(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction);
at::Tensor binary_cross_entropy_hpu_lazy(
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t reduction);
at::Tensor binary_cross_entropy_backward_hpu_lazy(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t reduction);
at::Tensor binary_cross_entropy_with_logits_hpu_lazy(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& pos_weight,
    int64_t reduction);
std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_hpu_lazy(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool training,
    double momentum,
    double eps);
std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_bwd_hpu_lazy(
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
std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_hpu_lazy(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    double eps);
std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_backward_hpu_lazy(
    const at::Tensor& dY,
    const at::Tensor& X,
    at::IntArrayRef normalized_shape,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    std::array<bool, 3> grad_input_mask);
at::Tensor norm_scalar_hpu_lazy(const at::Tensor& self, at::Scalar p);
std::tuple<at::Tensor, at::Tensor> max_pool2d_with_indices_hpu_lazy(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode);
at::Tensor& max_pool2d_with_indices_backward_out_hpu_lazy(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& indices,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode);
at::Tensor max_pool2d_with_indices_backward_hpu_lazy(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor& indices);
at::Tensor avg_pool2d_hpu_lazy(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);
at::Tensor& avg_pool2d_backward_out_hpu_lazy(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);
at::Tensor avg_pool2d_backward_hpu_lazy(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);
at::Tensor adaptive_avg_pool2d_hpu_lazy(
    const at::Tensor& input,
    at::IntArrayRef output_size);
at::Tensor adaptive_avg_pool2d_backward_hpu_lazy(
    const at::Tensor& grad_output,
    const at::Tensor& input);
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
at::Tensor& randperm_hpu_lazy(
    at::Tensor& output,
    int64_t n,
    c10::optional<at::Generator> gen);
at::Tensor bernoulli_hpu_lazy(
    const at::Tensor& self,
    c10::optional<at::Generator> gen = c10::nullopt);
at::Tensor& bernoulli_scalar_hpu_lazy(
    at::Tensor& self,
    double p,
    c10::optional<at::Generator> gen = c10::nullopt);
std::tuple<at::Tensor, at::Tensor> fused_dropout_hpu_lazy(
    const at::Tensor& self,
    double p,
    c10::optional<at::Generator> gen = c10::nullopt);
at::Tensor repeat_hpu_lazy(const at::Tensor& self, at::IntArrayRef repeats);
at::Tensor sum_dim_IntList_hpu_lazy(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype);
at::Tensor& sum_IntList_out_hpu_lazy(
    at::Tensor& output,
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype);
at::Tensor mean_dim_hpu_lazy(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype);
at::Tensor& mean_dim_out_hpu_lazy(
    at::Tensor& output,
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype);
at::Tensor sum_hpu_lazy(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype);
at::Tensor mean_hpu_lazy(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype);
at::Tensor prod_dim_hpu_lazy(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype);
at::Tensor prod_hpu_lazy(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype);
std::tuple<at::Tensor, at::Tensor, at::Tensor> unique2_hpu_lazy(
    const at::Tensor& self,
    bool sorted,
    bool return_inverse,
    bool return_counts);
std::tuple<at::Tensor, at::Tensor> max_dim_hpu_lazy(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim);
at::Tensor max_hpu_lazy(const at::Tensor& self);
at::Tensor& any_dim_out_hpu_lazy(
    at::Tensor& output,
    const at::Tensor& self,
    int64_t dim,
    bool keepdim);
at::Tensor any_dim_hpu_lazy(const at::Tensor& self, int64_t dim, bool keepdim);
at::Tensor any_hpu_lazy(const at::Tensor& self);
at::Tensor argmax_hpu_lazy(
    const at::Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim);
at::Tensor& bitwise_and_out_hpu_lazy(
    at::Tensor& out,
    const at::Tensor& self,
    const at::Tensor& other);
at::Tensor& bitwise_or_out_hpu_lazy(
    at::Tensor& out,
    const at::Tensor& self,
    const at::Tensor& other);
at::Tensor& bitwise_xor_out_hpu_lazy(
    at::Tensor& out,
    const at::Tensor& self,
    const at::Tensor& other);
at::Tensor& bitwise_not_out_hpu_lazy(at::Tensor& out, const at::Tensor& self);
at::Tensor log_softmax_hpu_lazy(
    const at::Tensor& self,
    const int64_t dim,
    const bool half_to_float);
at::Tensor log_softmax_backward_hpu_lazy(
    const at::Tensor& grad,
    const at::Tensor& output,
    int64_t dim,
    const at::Tensor& input);
at::Tensor softmax_hpu_lazy(
    const at::Tensor& self,
    int64_t dim,
    const bool half_to_float);
at::Tensor softmax_backward_hpu_lazy(
    const at::Tensor& grad,
    const at::Tensor& output,
    int64_t dim,
    const at::Tensor& input);

at::Tensor empty_hpu_lazy(
    at::IntArrayRef size,
    const at::TensorOptions& options,
    c10::optional<at::MemoryFormat> optional_memory_format,
    bool create_storage = true);
at::Tensor empty_strided_hpu_lazy(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    const at::TensorOptions& options,
    bool create_storage = true);

at::Tensor clone_hpu_lazy(
    const at::Tensor& self,
    c10::optional<at::MemoryFormat> memory_format);
at::Tensor& zero_hpu_lazy(at::Tensor& self);
at::Tensor cat_hpu_lazy(const at::TensorList tensors, int64_t dim_ = 0);
at::Tensor& cat_hpu_lazy_out(
    at::Tensor& result,
    const at::TensorList tensors,
    int64_t dim_);
at::Tensor transpose_hpu_lazy(
    const at::Tensor& self,
    int64_t dim0_,
    int64_t dim1_);
at::Tensor& transpose_hpu_lazy_(at::Tensor& self, int64_t dim0_, int64_t dim1_);
at::Tensor t_hpu_lazy(const at::Tensor& self);
at::Tensor& t_hpu_lazy_(at::Tensor& self);
at::Tensor permute_hpu_lazy(const at::Tensor& self, at::IntArrayRef dims_);
at::Tensor permute_cl_hpu_lazy(const at::Tensor& self, at::IntArrayRef dims_);
at::Tensor expand_hpu_lazy(
    const at::Tensor& self,
    at::IntArrayRef size,
    bool implicit);
std::vector<at::Tensor> split_with_sizes_hpu_lazy(
    const at::Tensor& self,
    at::IntArrayRef split_sizes,
    int64_t dim);
at::Tensor threshold_backward_hpu_lazy(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Scalar threshold);
std::tuple<at::Tensor&, at::Tensor&> topk_out_hpu_lazy(
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t k,
    int64_t dim_,
    bool largest,
    bool sorted);
std::tuple<at::Tensor, at::Tensor> topk_hpu_lazy(
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted);
std::tuple<at::Tensor, at::Tensor> sort_hpu_lazy(
    const at::Tensor& self,
    int64_t dim,
    bool descending);
at::Tensor elu_hpu_lazy(
    const at::Tensor& self,
    at::Scalar alpha,
    at::Scalar scale,
    at::Scalar input_scale);
at::Tensor& elu_hpu_lazy_(
    at::Tensor& self,
    at::Scalar alpha,
    at::Scalar scale,
    at::Scalar input_scale);
at::Tensor relu_hpu_lazy(const at::Tensor& input);
at::Tensor& relu_hpu_lazy_(at::Tensor& self);
at::Tensor& leaky_relu_lazy_(at::Tensor& self, at::Scalar negative_slope);
at::Tensor leaky_relu_backward_lazy(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Scalar negative_slope,
    bool self_is_result);
at::Tensor leaky_relu_lazy(const at::Tensor& self, at::Scalar negative_slope);
at::Tensor leaky_relu_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Scalar negative_slope,
    bool self_is_result);

at::Tensor sigmoid_hpu_lazy(const at::Tensor& input);
at::Tensor sigmoid_backward_hpu_lazy(
    const at::Tensor& grad_in,
    const at::Tensor& input);

at::Tensor& hardsigmoid_hpu_lazy_(at::Tensor& self);
at::Tensor hardsigmoid_hpu_lazy(const at::Tensor& self);
at::Tensor hardsigmoid_backward_hpu_lazy(
    const at::Tensor& grad_output,
    const at::Tensor& self);

at::Tensor sqrt_hpu_lazy(const at::Tensor& input);
at::Tensor sqrt_hpu_lazy_(at::Tensor& input);
at::Tensor tanh_hpu_lazy(const at::Tensor& input);
at::Tensor& tanh_hpu_lazy_(at::Tensor& self);
at::Tensor& tanh_out_hpu_lazy(at::Tensor& out, const at::Tensor& self);
at::Tensor tanh_backward_hpu_lazy(
    const at::Tensor& grad_in,
    const at::Tensor& input);
at::Tensor gelu_hpu_lazy(const at::Tensor& self);
at::Tensor gelu_backward_hpu_lazy(
    const at::Tensor& grad,
    const at::Tensor& self);
at::Tensor& erf_hpu_lazy_(at::Tensor& self);
at::Tensor erf_hpu_lazy(const at::Tensor& self);
at::Tensor& exp_hpu_lazy_(at::Tensor& self);
at::Tensor exp_hpu_lazy(const at::Tensor& self);
at::Tensor& neg_out_hpu_lazy(at::Tensor& result, const at::Tensor& input);
at::Tensor& reciprocal_hpu_lazy_(at::Tensor& self);
at::Tensor reciprocal_hpu_lazy(const at::Tensor& self);
at::Tensor& reciprocal_out_hpu_lazy(at::Tensor& result, const at::Tensor& self);
at::Tensor clamp_min_hpu_lazy(const at::Tensor& self, at::Scalar min);
at::Tensor sign_hpu_lazy(const at::Tensor& input);
at::Tensor& sign_hpu_lazy_(at::Tensor& self);
at::Tensor sgn_hpu_lazy(const at::Tensor& input);
at::Tensor& sgn_hpu_lazy_(at::Tensor& self);
at::Tensor floor_hpu_lazy(const at::Tensor& input);
at::Tensor& floor_hpu_lazy_(at::Tensor& self);
at::Tensor log_hpu_lazy(const at::Tensor& input);
at::Tensor& log_hpu_lazy_(at::Tensor& self);
at::Tensor log2_hpu_lazy(const at::Tensor& input);
at::Tensor& log2_hpu_lazy_(at::Tensor& self);
at::Tensor& clamp_hpu_lazy_(
    at::Tensor& self,
    c10::optional<at::Scalar> min,
    c10::optional<at::Scalar> max);
at::Tensor clamp_hpu_lazy(
    const at::Tensor& self,
    c10::optional<at::Scalar> min,
    c10::optional<at::Scalar> max);
at::Tensor abs_hpu_lazy(const at::Tensor& input);
at::Tensor& abs_hpu_lazy_(at::Tensor& self);
at::Tensor round_hpu_lazy(const at::Tensor& self);
at::Tensor& round_hpu_lazy_(at::Tensor& self);
at::Tensor rsqrt_hpu_lazy(const at::Tensor& self);
at::Tensor& rsqrt_hpu_lazy_(at::Tensor& self);
at::Tensor isfinite_hpu_lazy(const at::Tensor& self);
at::Tensor neg_hpu_lazy(const at::Tensor& self);
at::Scalar _local_scalar_dense_hpu_lazy(const at::Tensor& self);
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
    float max_grad_norm);
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
    const std::vector<at::Tensor>& trust_ratio_vec,
    const float step,
    const float weight_decay,
    const int use_lamb);
at::Tensor& optimizer_adagrad_hpu_lazy(
    const at::TensorList& gradients,
    at::TensorList& weights,
    at::TensorList& variances,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float lrd,
    const float epsilon);
at::Tensor& optimizer_sgd_hpu_lazy(
    const at::TensorList& gradients,
    at::TensorList& weights,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov);
at::Tensor& optimizer_sgd_momentum_hpu_lazy(
    const at::TensorList& gradients,
    at::TensorList& weights,
    at::TensorList& momentum,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov);
at::Tensor ones_like_hpu_lazy(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> memory_format);
at::Tensor masked_scale_hpu_lazy(
    const at::Tensor& self,
    const at::Tensor& mask,
    double scale);
at::Tensor matmul_hpu_lazy(const at::Tensor& self, const at::Tensor& other);
std::tuple<at::Tensor, at::Tensor> matmul_backward_hpu_lazy(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& other);
at::Tensor habana_nms_hpu_lazy(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    float iou_threshold,
    float score_threshold);
at::Tensor isnan_hpu_lazy(const at::Tensor& self);
at::Tensor silu_hpu_lazy(const at::Tensor& self);
} // namespace habana_lazy
