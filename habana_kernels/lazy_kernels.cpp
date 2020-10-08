/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "habana_kernels/eager_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "pytorch_helpers/habana_device/HPUAllocator.h"

// calling eager mode kernels as a temporary placeholder to avoid warnings
Tensor& copy_hpu_lazy_(Tensor& self, const Tensor& src, bool non_blocking) {
  return copy_hpu_(self, src, non_blocking);
};
Tensor as_strided_hpu_lazy(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  return as_strided_hpu(self, size, stride, storage_offset);
};
Tensor& set_hpu_lazy_(
    Tensor& self,
    Storage source,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride) {
  return set_hpu_(self, source, storage_offset, size, stride);
};
Tensor view_hpu_lazy(const Tensor& self, IntArrayRef size) {
  return view_hpu(self, size);
};
Tensor addcmul_hpu_lazy(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  return addcmul_hpu(self, tensor1, tensor2, alpha);
};
Tensor& addcmul_hpu_lazy_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  return addcmul_hpu_(self, tensor1, tensor2, alpha);
};
Tensor addcdiv_hpu_lazy(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  return addcdiv_hpu(self, tensor1, tensor2, alpha);
};
Tensor& addcdiv_hpu_lazy_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  return addcdiv_hpu_(self, tensor1, tensor2, alpha);
};
Tensor add_tensor_hpu_lazy(
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  auto hl_other = habana_lazy::GetOrCreateHbLazyTensor(other, c10::kHABANA);
  auto hl_alpha = habana_lazy::GetIrValueForScalar(alpha);

  int num_outputs = 1;
  auto node = habana_lazy::Node::Create(
      Symbol::fromQualString("aten::add"),
      {hl_self.GetIrValue(), hl_other.GetIrValue(), hl_alpha},
      num_outputs);
  at::Tensor result = add_tensor_hpu(self, other, alpha);
  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  return result;
}
Tensor add_scalar_hpu_lazy(const Tensor& self, Scalar other, Scalar alpha) {
  return add_scalar_hpu(self, other, alpha);
};
Tensor& add_scalar_hpu_lazy_(Tensor& self, Scalar other, Scalar alpha) {
  return add_scalar_hpu_(self, other, alpha);
};
Tensor& add_tensor_hpu_lazy_(Tensor& self, const Tensor& other, Scalar alpha) {
  return add_tensor_hpu_(self, other, alpha);
};
Tensor sub_tensor_hpu_lazy(
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
  return sub_tensor_hpu(self, other, alpha);
};
Tensor& sub_tensor_hpu_lazy_(Tensor& self, const Tensor& other, Scalar alpha) {
  return sub_tensor_hpu_(self, other, alpha);
};
Tensor sub_scalar_hpu_lazy(const Tensor& self, Scalar other, Scalar alpha) {
  return sub_scalar_hpu(self, other, alpha);
};
Tensor& sub_scalar_hpu_lazy_(Tensor& self, Scalar other, Scalar alpha) {
  return sub_scalar_hpu_(self, other, alpha);
};
Tensor rsub_scalar_hpu_lazy(const Tensor& self, Scalar other, Scalar alpha) {
  return rsub_scalar_hpu(self, other, alpha);
};
Tensor& mul_tensor_hpu_lazy_(Tensor& self, const Tensor& other) {
  return mul_tensor_hpu_(self, other);
};
Tensor mul_tensor_hpu_lazy(const Tensor& self, const Tensor& other) {
  return mul_tensor_hpu(self, other);
};
Tensor mul_scalar_hpu_lazy(const Tensor& self, Scalar other) {
  return mul_scalar_hpu(self, other);
};
Tensor& mul_scalar_hpu_lazy_(Tensor& self, Scalar other) {
  return mul_scalar_hpu_(self, other);
};
Tensor div_tensor_hpu_lazy(const Tensor& self, const Tensor& other) {
  return div_tensor_hpu(self, other);
};
Tensor& div_tensor_hpu_lazy_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  return div_tensor_hpu_out(result, self, other);
};
Tensor& div_tensor_hpu_lazy_(Tensor& self, const Tensor& other) {
  return div_tensor_hpu_(self, other);
};
Tensor div_scalar_hpu_lazy(const Tensor& self, Scalar other) {
  return div_scalar_hpu(self, other);
};
Tensor& div_scalar_hpu_lazy_(Tensor& self, Scalar other) {
  return div_scalar_hpu_(self, other);
};
Tensor pow_tensor_tensor_hpu_lazy(const Tensor& self, const Tensor& other) {
  return pow_tensor_tensor_hpu(self, other);
};
Tensor& pow_tensor_tensor_hpu_lazy_(Tensor& self, const Tensor& other) {
  return pow_tensor_tensor_hpu_(self, other);
};
Tensor pow_tensor_scalar_hpu_lazy(const Tensor& self, Scalar other) {
  return pow_tensor_scalar_hpu(self, other);
};
Tensor& pow_tensor_scalar_hpu_lazy_(Tensor& self, Scalar other) {
  return pow_tensor_scalar_hpu_(self, other);
};
Tensor pow_scalar_tensor_hpu_lazy(Scalar other, const Tensor& self) {
  return pow_scalar_tensor_hpu(other, self);
};
Tensor gt_hpu_lazy(Tensor& self, Tensor& other) {
  return gt_hpu(self, other);
};

void eq_tensor_out_hpu_lazy(
    Tensor& output,
    const Tensor& self,
    const Tensor& other) {
  return eq_tensor_out_hpu(output, self, other);
};
Tensor eq_tensor_hpu_lazy(Tensor& self, Tensor& other) {
  return eq_tensor_hpu(self, other);
};
Tensor eq_tensor_scalar_hpu_lazy(Tensor& self, Scalar other) {
  return eq_tensor_scalar_hpu(self, other);
};
Tensor lt_scalar_hpu_lazy(Tensor& self, Scalar other) {
  return lt_scalar_hpu(self, other);
};
Tensor lt_tensor_hpu_lazy(Tensor& self, Tensor& other) {
  return lt_tensor_hpu(self, other);
};
Tensor convolution_hpu_lazy(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups) {
  return convolution_hpu(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups);
};
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
    std::array<bool, 3> output_mask) {
  return convolution_backward_hpu(
      grad_output,
      input,
      weight,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      output_mask);
};
std::tuple<Tensor, Tensor, Tensor, Tensor> embedding_bag_hpu_lazy(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    bool scale_grad_by_freq,
    int64_t mode,
    UNUSED bool sparse,
    Tensor& per_sample_weights,
    UNUSED bool include_last_offset) {
  return embedding_bag_hpu(
      weight,
      indices,
      offsets,
      scale_grad_by_freq,
      mode,
      sparse,
      per_sample_weights,
      include_last_offset);
};
Tensor embedding_bag_bwd_hpu_lazy(
    Tensor& grad,
    Tensor& indices,
    Tensor& offsets,
    UNUSED Tensor& offset2bag,
    UNUSED Tensor& bag_size,
    UNUSED Tensor& maximum_indices,
    int num_weights,
    bool scale_grad_by_freq,
    int mode,
    Tensor per_sample_weights) {
  return embedding_bag_bwd_hpu(
      grad,
      indices,
      offsets,
      offset2bag,
      bag_size,
      maximum_indices,
      num_weights,
      scale_grad_by_freq,
      mode,
      per_sample_weights);
};
Tensor constant_pad_hpu_lazy(
    const Tensor& self,
    IntArrayRef pad,
    Scalar value) {
  return constant_pad_hpu(self, pad, value);
};
Tensor embedding_hpu_lazy(
    const Tensor& weight,
    const Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  return embedding_hpu(
      weight, indices, padding_idx, scale_grad_by_freq, sparse);
};
Tensor embedding_dense_backward_hpu_lazy(
    const Tensor& grad,
    const Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  return embedding_dense_backward_hpu(
      grad, indices, num_weights, padding_idx, scale_grad_by_freq);
};
Tensor embedding_bag_sum_hpu_lazy(
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode) {
  return embedding_bag_sum_hpu(
      input, indices, offsets, valid_count, kernel_mode);
};
Tensor embedding_bag_sum_fwd_hpu_lazy(
    const Tensor& input,
    const Tensor& indices_fwd,
    const Tensor& offsets_fwd,
    const Tensor& valid_count,
    const Tensor& indices_bwd,
    const Tensor& offsets_bwd,
    const Tensor& valid_count_bwd,
    const Tensor& grad_weight) {
  return embedding_bag_sum_fwd_hpu(
      input,
      indices_fwd,
      offsets_fwd,
      valid_count,
      indices_bwd,
      offsets_bwd,
      valid_count_bwd,
      grad_weight);
};
Tensor& embedding_bag_sum_bwd_out_hpu_lazy(
    Tensor& out,
    const Tensor& input,
    const Tensor& indices_bwd,
    const Tensor& offsets_bwd,
    const Tensor& valid_count_bwd) {
  return embedding_bag_sum_bwd_out_hpu(
      out, input, indices_bwd, offsets_bwd, valid_count_bwd);
};
Tensor& fill_hpu_lazy_(Tensor& self, Scalar value) {
  return fill_hpu_(self, value);
};
Tensor& masked_fill_hpu_lazy_(
    Tensor& self,
    const Tensor& mask,
    const Tensor& value) {
  return masked_fill_hpu_(self, mask, value);
};
Tensor& masked_fill_scalar_hpu_lazy_(
    Tensor& self,
    const Tensor& mask,
    Scalar value) {
  return masked_fill_scalar_hpu_(self, mask, value);
};
Tensor gather_src_hpu_lazy(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    bool sparse_grad) {
  return gather_src_hpu(self, dim_, index, sparse_grad);
};
Tensor& scatter_inplace_src_hpu_lazy(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  return scatter_inplace_src_hpu(self, dim_, index, src);
};
Tensor scatter_src_hpu_lazy(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  return scatter_src_hpu(self, dim_, index, src);
};
Tensor scatter_add_src_hpu_lazy(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  return scatter_add_src_hpu(self, dim_, index, src);
};
Tensor& scatter_add_inplace_src_hpu_lazy(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  return scatter_add_inplace_src_hpu(self, dim_, index, src);
};
Tensor& index_add_hpu_lazy_(
    Tensor& self,
    int64_t dim_,
    const Tensor& indices,
    const Tensor& source) {
  return index_add_hpu_(self, dim_, indices, source);
};
Tensor index_put_hpu_lazy(
    const Tensor& self,
    TensorList indices,
    const Tensor& value,
    bool accumulate) {
  return index_put_hpu(self, indices, value, accumulate);
};
Tensor& index_put_hpu_lazy_(
    Tensor& self,
    TensorList indices,
    const Tensor& value,
    bool accumulate) {
  return index_put_hpu_(self, indices, value, accumulate);
};
Tensor index_select_hpu_lazy(
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  return index_select_hpu(self, dim, index);
};
Tensor gather2d_hpu_lazy(
    const Tensor& input,
    const Tensor& indices,
    int64_t validCount) {
  return gather2d_hpu(input, indices, validCount);
};
Tensor slice_hpu_lazy(
    const Tensor& self,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t step) {
  return slice_hpu(self, dim, start, end, step);
};
Tensor select_hpu_lazy(const Tensor& self, int64_t dim, int64_t index) {
  return select_hpu(self, dim, index);
};
Tensor& arange_hpu_lazy(Tensor& output, Scalar start, Scalar end, Scalar step) {
  return arange_hpu(output, start, end, step);
};
Tensor mm_hpu_lazy(const at::Tensor& mat1, const at::Tensor& mat2) {
  return mm_hpu(mat1, mat2);
};
Tensor addmm_hpu_lazy(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha) {
  return addmm_hpu(self, mat1, mat2, beta, alpha);
};
Tensor& batch_gemm_out_hpu_lazy(
    Tensor& out,
    const Tensor& self,
    const Tensor& mat2) {
  return batch_gemm_out_hpu(out, self, mat2);
};
Tensor batch_gemm_hpu_lazy(const Tensor& self, const Tensor& mat2) {
  return batch_gemm_hpu(self, mat2);
};
Tensor dot_hpu_lazy(const Tensor& self, const Tensor& other) {
  return dot_hpu(self, other);
};
Tensor mv_hpu_lazy(const Tensor& self, const Tensor& other) {
  return mv_hpu(self, other);
};
std::tuple<Tensor, Tensor> nll_loss_forward_hpu_lazy(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  return nll_loss_forward_hpu(self, target, weight, reduction, ignore_index);
};
Tensor nll_loss_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    UNUSED const Tensor& total_weight) {
  return nll_loss_backward_hpu(
      grad_output, self, target, weight, reduction, ignore_index, total_weight);
};
Tensor mse_loss_forward_hpu_lazy(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  return mse_loss_forward_hpu(self, target, reduction);
};
Tensor mse_loss_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  return mse_loss_backward_hpu(grad_output, self, target, reduction);
};
Tensor binary_cross_entropy_hpu_lazy(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  return binary_cross_entropy_hpu(self, target, weight, reduction);
};
Tensor binary_cross_entropy_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  return binary_cross_entropy_backward_hpu(
      grad_output, self, target, weight, reduction);
};

std::tuple<Tensor, Tensor, Tensor> batch_norm_hpu_lazy(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool training,
    double momentum,
    double eps) {
  return batch_norm_hpu(
      input, weight, bias, running_mean, running_var, training, momentum, eps);
};
std::tuple<Tensor, Tensor, Tensor> batch_norm_bwd_hpu_lazy(
    Tensor& grad_out,
    Tensor& input,
    Tensor& weight,
    UNUSED Tensor& running_mean,
    UNUSED Tensor& running_var,
    Tensor& save_mean,
    Tensor& save_invstd,
    bool train,
    double eps,
    UNUSED std::array<bool, 3> output_mask) {
  return batch_norm_bwd_hpu(
      grad_out,
      input,
      weight,
      running_mean,
      running_var,
      save_mean,
      save_invstd,
      train,
      eps,
      output_mask);
};
std::tuple<Tensor, Tensor, Tensor> layer_norm_hpu_lazy(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int64_t m,
    int64_t n,
    double eps) {
  return layer_norm_hpu(input, weight, bias, m, n, eps);
};
std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_hpu_lazy(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    std::array<bool, 3> grad_input_mask) {
  return layer_norm_backward_hpu(
      dY, X, mean, rstd, gamma, M, N, grad_input_mask);
};
Tensor norm_scalar_hpu_lazy(const Tensor& self, Scalar p) {
  return norm_scalar_hpu(self, p);
};
std::tuple<Tensor, Tensor> max_pool2d_with_indices_hpu_lazy(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  return max_pool2d_with_indices_hpu(
      input, kernel_size, stride, padding, dilation, ceil_mode);
};
Tensor& max_pool2d_with_indices_backward_out_hpu_lazy(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& indices,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  return max_pool2d_with_indices_backward_out_hpu(
      grad_input,
      grad_output,
      input,
      indices,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);
};
Tensor max_pool2d_with_indices_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices) {
  return max_pool2d_with_indices_backward_hpu(
      grad_output,
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      indices);
};
Tensor avg_pool2d_hpu_lazy(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  return avg_pool2d_hpu(
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
};
Tensor& avg_pool2d_backward_out_hpu_lazy(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  return avg_pool2d_backward_out_hpu(
      grad_input,
      grad_output,
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
};
Tensor avg_pool2d_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  return avg_pool2d_backward_hpu(
      grad_output,
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
};
void uniform_hpu_lazy(
    const Tensor& self,
    double from,
    double to,
    CPUGenerator* gen) {
  uniform_hpu(self, from, to, gen);
};
void normal_hpu_lazy(
    const Tensor& self,
    double mean,
    double std,
    CPUGenerator* gen) {
  normal_hpu(self, mean, std, gen);
};
Tensor bernoulli_hpu_lazy(const Tensor& self, CPUGenerator* gen) {
  return bernoulli_hpu(self, gen);
};
Tensor& bernoulli_scalar_hpu_lazy(Tensor& self, double p, CPUGenerator* gen) {
  return bernoulli_scalar_hpu(self, p, gen);
};
Tensor sum_dim_IntList_hpu_lazy(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  return sum_dim_IntList_hpu(self, dim, keepdim, dtype);
};
Tensor& sum_IntList_out_hpu_lazy(
    Tensor& output,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  return sum_IntList_out_hpu(output, self, dim, keepdim, dtype);
};
Tensor mean_dim_hpu_lazy(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  return mean_dim_hpu(self, dim, keepdim, dtype);
};
Tensor& mean_dim_out_hpu_lazy(
    Tensor& output,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  return mean_dim_out_hpu(output, self, dim, keepdim, dtype);
};
Tensor sum_hpu_lazy(const Tensor& self, c10::optional<ScalarType> dtype) {
  return sum_hpu(self, dtype);
};
Tensor mean_hpu_lazy(const Tensor& self, c10::optional<ScalarType> dtype) {
  return mean_hpu(self, dtype);
};
Tensor& any_dim_out_hpu_lazy(
    Tensor& output,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  return any_dim_out_hpu(output, self, dim, keepdim);
};
Tensor any_dim_hpu_lazy(const Tensor& self, int64_t dim, bool keepdim) {
  return any_dim_hpu(self, dim, keepdim);
};
Tensor any_hpu_lazy(const Tensor& self) {
  return any_hpu(self);
};
namespace habana {
Tensor log_softmax_hpu_lazy(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  return log_softmax_hpu(self, dim, half_to_float);
};
Tensor log_softmax_backward_hpu_lazy(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    const Tensor& input) {
  return log_softmax_backward_hpu(grad, output, dim, input);
};
Tensor softmax_hpu_lazy(
    const Tensor& self,
    int64_t dim,
    const bool half_to_float) {
  return softmax_hpu(self, dim, half_to_float);
};
Tensor softmax_backward_hpu_lazy(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    const Tensor& input) {
  return softmax_backward_hpu(grad, output, dim, input);
};
} // namespace habana
namespace at {
namespace native {
Tensor empty_hpu_lazy(
    IntArrayRef size,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format,
    bool create_storage) {
  if (create_storage) {
    c10 ::Allocator* allocator;
    if (options.pinned_memory()) {
      TORCH_CHECK(false, "habana allocator doesn't supported pinned memory");
    } else {
      allocator = habana::getHABANADeviceAllocator();
    }
    int64_t nelements = prod_intlist(size);
    auto dtype = options.dtype();
    auto storage_impl = c10::make_intrusive<StorageImpl>(
        dtype,
        nelements,
        allocator->allocate(nelements * dtype.itemsize()),
        allocator,
        /*resizeable=*/true);
    habana_lazy::HbLazyTensor hb_tensor =
        habana_lazy::HbLazyTensor::CreateHbLazyTensor(
            size,
            0,
            options.device(),
            c10::typeMetaToScalarType(options.dtype()));
    Tensor at_tensor =
        habana_lazy::AtenFromHbLazyTensor(hb_tensor, std::move(storage_impl));
    // Setup the tensor sizes/strides, for now assuming contiguous
    at_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
    hb_tensor.SetTensorData(at_tensor);

    return at_tensor;

  } else {
    habana_lazy::HbLazyTensor hb_tensor =
        habana_lazy::HbLazyTensor::CreateHbLazyTensor(
            size,
            0,
            options.device(),
            c10::typeMetaToScalarType(options.dtype()));
    Tensor at_tensor = habana_lazy::AtenFromHbLazyTensor(hb_tensor);
    // Setup the tensor sizes/strides, for now assuming contiguous
    at_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
    hb_tensor.SetTensorData(at_tensor);

    return at_tensor;
  }
};

Tensor empty_strided_hpu_lazy(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options) {
  return empty_strided_hpu(size, stride, options);
};
} // namespace native
} // namespace at
Tensor clone_hpu_lazy(
    const Tensor& self,
    c10::optional<MemoryFormat> memory_format) {
  return clone_hpu(self, memory_format);
};
Tensor& zero_hpu_lazy(Tensor& self) {
  return zero_hpu(self);
};
Tensor cat_hpu_lazy(const TensorList tensors, int64_t dim_) {
  return cat_hpu(tensors, dim_);
};
Tensor& cat_hpu_lazy_out(
    Tensor& result,
    const TensorList tensors,
    int64_t dim_) {
  return cat_hpu_out(result, tensors, dim_);
};
Tensor transpose_hpu_lazy(const Tensor& self, int64_t dim0_, int64_t dim1_) {
  return transpose_hpu(self, dim0_, dim1_);
};
Tensor& transpose_hpu_lazy_(Tensor& self, int64_t dim0_, int64_t dim1_) {
  return transpose_hpu_(self, dim0_, dim1_);
};
Tensor t_hpu_lazy(const Tensor& self) {
  return t_hpu(self);
};
Tensor& t_hpu_lazy_(Tensor& self) {
  return t_hpu_(self);
};
Tensor permute_hpu_lazy(const Tensor& self, IntArrayRef dims_) {
  return permute_hpu(self, dims_);
};
Tensor expand_hpu_lazy(const Tensor& self, IntArrayRef size, bool implicit) {
  return expand_hpu(self, size, implicit);
};
std::vector<Tensor> split_with_sizes_hpu_lazy(
    const Tensor& self,
    IntArrayRef split_sizes,
    int64_t dim) {
  return split_with_sizes_hpu(self, split_sizes, dim);
};
Tensor threshold_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    Scalar threshold) {
  return threshold_backward_hpu(grad_output, self, threshold);
};
std::tuple<Tensor&, Tensor&> topk_out_hpu_lazy(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool largest,
    bool sorted) {
  return topk_out_hpu(values, indices, self, k, dim_, largest, sorted);
};
std::tuple<Tensor, Tensor> topk_hpu_lazy(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  return topk_hpu(self, k, dim, largest, sorted);
};
std::tuple<Tensor, Tensor> sort_hpu_lazy(
    const Tensor& self,
    int64_t dim,
    bool descending) {
  return sort_hpu(self, dim, descending);
};
Tensor unary_op_hpu_lazy(
    const Tensor& input,
    std::string& node_type,
    UnaryOperator* Op) {
  return unary_op_hpu(input, node_type, Op);
};
Tensor unary_backward_op_hpu_lazy(
    const Tensor& grad_in,
    const Tensor& input,
    std::string& node_type,
    UnaryBackwardOperator* Op) {
  return unary_backward_op_hpu(grad_in, input, node_type, Op);
};
Tensor relu_hpu_lazy(const Tensor& input) {
  return relu_hpu(input);
};
Tensor& relu_hpu_lazy_(Tensor& self) {
  return relu_hpu_(self);
};
Tensor sigmoid_hpu_lazy(const Tensor& input) {
  return sigmoid_hpu(input);
};
Tensor sigmoid_backward_hpu_lazy(const Tensor& grad_in, const Tensor& input) {
  return sigmoid_backward_hpu(grad_in, input);
};
Tensor sqrt_hpu_lazy(const Tensor& input) {
  return sqrt_hpu(input);
};
Tensor tanh_hpu_lazy(const Tensor& input) {
  return tanh_hpu(input);
};
Tensor& tanh_hpu_lazy_(Tensor& self) {
  return tanh_hpu_(self);
};
Tensor& tanh_out_hpu_lazy(Tensor& out, Tensor& self) {
  return tanh_out_hpu(out, self);
};
Tensor tanh_backward_hpu_lazy(const Tensor& grad_in, const Tensor& input) {
  return tanh_backward_hpu(grad_in, input);
};
Tensor gelu_hpu_lazy(const Tensor& self) {
  return gelu_hpu(self);
};
Tensor gelu_backward_hpu_lazy(const Tensor& grad, const Tensor& self) {
  return gelu_backward_hpu(grad, self);
};
Tensor& erf_hpu_lazy_(Tensor& self) {
  return erf_hpu_(self);
};
Tensor erf_hpu_lazy(const Tensor& self) {
  return erf_hpu(self);
};
Tensor& exp_hpu_lazy_(Tensor& self) {
  return exp_hpu_(self);
};
Tensor exp_hpu_lazy(const Tensor& self) {
  return exp_hpu(self);
};
Tensor& neg_out_hpu_lazy(Tensor& result, const Tensor& input) {
  return neg_out_hpu(result, input);
};
Tensor& reciprocal_hpu_lazy_(Tensor& self) {
  return reciprocal_hpu_(self);
};
Tensor reciprocal_hpu_lazy(const Tensor& self) {
  return reciprocal_hpu(self);
};
Tensor& reciprocal_out_hpu_lazy(Tensor& result, const Tensor& self) {
  return reciprocal_out_hpu(result, self);
};
Tensor clamp_min_hpu_lazy(const Tensor& self, Scalar min) {
  return clamp_min_hpu(self, min);
};
Tensor& clamp_hpu_lazy_(
    Tensor& self,
    c10::optional<Scalar> min,
    c10::optional<Scalar> max) {
  return clamp_hpu_(self, min, max);
};
Tensor clamp_hpu_lazy(
    const Tensor& self,
    c10::optional<Scalar> min,
    c10::optional<Scalar> max) {
  return clamp_hpu(self, min, max);
};
Tensor abs_hpu_lazy(const Tensor& self) {
  return abs_hpu(self);
};
Tensor neg_hpu_lazy(const Tensor& self) {
  return neg_hpu(self);
};
namespace at {
namespace native {
Scalar _local_scalar_dense_hpu_lazy(const Tensor& self) {
  return _local_scalar_dense_hpu(self);
}
} // namespace native
} // namespace at