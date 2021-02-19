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
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_lazy/lazy_executor.h"

using namespace torch;
using namespace at;

Tensor& copy_hpu_wrap_(Tensor& self, const Tensor& src, bool non_blocking) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = copy_hpu_lazy_(self, src, non_blocking);
    return t;
  } else {
    return copy_hpu_(self, src, non_blocking);
  }
};
Tensor as_strided_hpu_wrap(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = as_strided_hpu_lazy(self, size, stride, storage_offset);

    return t;
  } else {
    return as_strided_hpu(self, size, stride, storage_offset);
  }
};
Tensor& set_hpu_wrap_(
    Tensor& self,
    Storage source,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = set_hpu_lazy_(self, source, storage_offset, size, stride);

    return t;
  } else {
    return set_hpu_(self, source, storage_offset, size, stride);
  }
};
Tensor view_hpu_wrap(const Tensor& self, IntArrayRef size) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = view_hpu_lazy(self, size);

    return t;
  } else {
    return view_hpu(self, size);
  }
};
Tensor addcmul_hpu_wrap(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = addcmul_hpu_lazy(self, tensor1, tensor2, alpha);

    return t;
  } else {
    return addcmul_hpu(self, tensor1, tensor2, alpha);
  }
};
Tensor& addcmul_hpu_wrap_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = addcmul_hpu_lazy_(self, tensor1, tensor2, alpha);

    return t;
  } else {
    return addcmul_hpu_(self, tensor1, tensor2, alpha);
  }
};
Tensor addcdiv_hpu_wrap(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = addcdiv_hpu_lazy(self, tensor1, tensor2, alpha);

    return t;
  } else {
    return addcdiv_hpu(self, tensor1, tensor2, alpha);
  }
};
Tensor& addcdiv_hpu_wrap_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = addcdiv_hpu_lazy_(self, tensor1, tensor2, alpha);

    return t;
  } else {
    return addcdiv_hpu_(self, tensor1, tensor2, alpha);
  }
};
Tensor add_tensor_hpu_wrap(
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return add_tensor_hpu_lazy(self, other, alpha);
  } else {
    return add_tensor_hpu(self, other, alpha);
  }
};
Tensor add_scalar_hpu_wrap(const Tensor& self, Scalar other, Scalar alpha) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = add_scalar_hpu_lazy(self, other, alpha);

    return t;
  } else {
    return add_scalar_hpu(self, other, alpha);
  }
};
Tensor& add_scalar_hpu_wrap_(Tensor& self, Scalar other, Scalar alpha) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = add_scalar_hpu_lazy_(self, other, alpha);

    return t;
  } else {
    return add_scalar_hpu_(self, other, alpha);
  }
};
Tensor& add_tensor_hpu_wrap_(Tensor& self, const Tensor& other, Scalar alpha) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return add_tensor_hpu_lazy_(self, other, alpha);
  } else {
    return add_tensor_hpu_(self, other, alpha);
  }
};
Tensor sub_tensor_hpu_wrap(
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = sub_tensor_hpu_lazy(self, other, alpha);

    return t;
  } else {
    return sub_tensor_hpu(self, other, alpha);
  }
};
Tensor& sub_tensor_hpu_wrap_(Tensor& self, const Tensor& other, Scalar alpha) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = sub_tensor_hpu_lazy_(self, other, alpha);

    return t;
  } else {
    return sub_tensor_hpu_(self, other, alpha);
  }
};
Tensor sub_scalar_hpu_wrap(const Tensor& self, Scalar other, Scalar alpha) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = sub_scalar_hpu_lazy(self, other, alpha);

    return t;
  } else {
    return sub_scalar_hpu(self, other, alpha);
  }
};
Tensor& sub_scalar_hpu_wrap_(Tensor& self, Scalar other, Scalar alpha) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = sub_scalar_hpu_lazy_(self, other, alpha);

    return t;
  } else {
    return sub_scalar_hpu_(self, other, alpha);
  }
};
Tensor rsub_scalar_hpu_wrap(const Tensor& self, Scalar other, Scalar alpha) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = rsub_scalar_hpu_lazy(self, other, alpha);

    return t;
  } else {
    return rsub_scalar_hpu(self, other, alpha);
  }
};
Tensor& mul_tensor_hpu_wrap_(Tensor& self, const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return mul_tensor_hpu_lazy_(self, other);
  } else {
    return mul_tensor_hpu_(self, other);
  }
};
Tensor mul_tensor_hpu_wrap(const Tensor& self, const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return mul_tensor_hpu_lazy(self, other);
  } else {
    return mul_tensor_hpu(self, other);
  }
};
Tensor mul_scalar_hpu_wrap(const Tensor& self, Scalar other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = mul_scalar_hpu_lazy(self, other);

    return t;
  } else {
    return mul_scalar_hpu(self, other);
  }
};
Tensor& mul_scalar_hpu_wrap_(Tensor& self, Scalar other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = mul_scalar_hpu_lazy_(self, other);

    return t;
  } else {
    return mul_scalar_hpu_(self, other);
  }
};
Tensor div_tensor_hpu_wrap(const Tensor& self, const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = div_tensor_hpu_lazy(self, other);

    return t;
  } else {
    return div_tensor_hpu(self, other);
  }
};
Tensor& div_tensor_hpu_wrap_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = div_tensor_hpu_lazy_out(result, self, other);

    return t;
  } else {
    return div_tensor_hpu_out(result, self, other);
  }
};
Tensor& div_tensor_hpu_wrap_(Tensor& self, const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = div_tensor_hpu_lazy_(self, other);

    return t;
  } else {
    return div_tensor_hpu_(self, other);
  }
};
Tensor div_scalar_hpu_wrap(const Tensor& self, Scalar other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = div_scalar_hpu_lazy(self, other);

    return t;
  } else {
    return div_scalar_hpu(self, other);
  }
};
Tensor& div_scalar_hpu_wrap_(Tensor& self, Scalar other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = div_scalar_hpu_lazy_(self, other);

    return t;
  } else {
    return div_scalar_hpu_(self, other);
  }
};
Tensor pow_tensor_tensor_hpu_wrap(const Tensor& self, const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = pow_tensor_tensor_hpu_lazy(self, other);

    return t;
  } else {
    return pow_tensor_tensor_hpu(self, other);
  }
};
Tensor& pow_tensor_tensor_hpu_wrap_(Tensor& self, const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = pow_tensor_tensor_hpu_lazy_(self, other);

    return t;
  } else {
    return pow_tensor_tensor_hpu_(self, other);
  }
};
Tensor pow_tensor_scalar_hpu_wrap(const Tensor& self, Scalar other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = pow_tensor_scalar_hpu_lazy(self, other);

    return t;
  } else {
    return pow_tensor_scalar_hpu(self, other);
  }
};
Tensor& pow_tensor_scalar_hpu_wrap_(Tensor& self, Scalar other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = pow_tensor_scalar_hpu_lazy_(self, other);

    return t;
  } else {
    return pow_tensor_scalar_hpu_(self, other);
  }
};
Tensor pow_scalar_tensor_hpu_wrap(Scalar other, const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = pow_scalar_tensor_hpu_lazy(other, self);

    return t;
  } else {
    return pow_scalar_tensor_hpu(other, self);
  }
};
Tensor gt_tensor_hpu_wrap(Tensor& self, Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = gt_tensor_hpu_lazy(self, other);

    return t;
  } else {
    return gt_tensor_hpu(self, other);
  }
};
Tensor gt_scalar_hpu_wrap(Tensor& self, Scalar other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = gt_scalar_hpu_lazy(self, other);

    return t;
  } else {
    return gt_scalar_hpu(self, other);
  }
};
void eq_tensor_out_hpu_wrap(
    Tensor& output,
    const Tensor& self,
    const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    eq_tensor_out_hpu_lazy(output, self, other);
  } else {
    eq_tensor_out_hpu(output, self, other);
  }
};
Tensor eq_tensor_hpu_wrap(Tensor& self, Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = eq_tensor_hpu_lazy(self, other);

    return t;
  } else {
    return eq_tensor_hpu(self, other);
  }
};
Tensor eq_tensor_scalar_hpu_wrap(Tensor& self, Scalar other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = eq_tensor_scalar_hpu_lazy(self, other);

    return t;
  } else {
    return eq_tensor_scalar_hpu(self, other);
  }
};
Tensor lt_scalar_hpu_wrap(Tensor& self, Scalar other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = lt_scalar_hpu_lazy(self, other);

    return t;
  } else {
    return lt_scalar_hpu(self, other);
  }
};
Tensor lt_tensor_hpu_wrap(Tensor& self, Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = lt_tensor_hpu_lazy(self, other);

    return t;
  } else {
    return lt_tensor_hpu(self, other);
  }
};
Tensor ge_scalar_hpu_wrap(const Tensor& self, Scalar other) {
  if (!habana_lazy::isDeviceInLoweringMode(self.device().index()) &&
      std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = ge_scalar_hpu_lazy(self, other);

    return t;
  } else {
    return ge_scalar_hpu(self, other);
  }
};
Tensor ge_tensor_hpu_wrap(const Tensor& self, const Tensor& other) {
  if (!habana_lazy::isDeviceInLoweringMode(self.device().index()) &&
      std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = ge_tensor_hpu_lazy(self, other);

    return t;
  } else {
    return ge_tensor_hpu(self, other);
  }
};
Tensor ne_scalar_hpu_wrap(const Tensor& self, Scalar other) {
  if (!habana_lazy::isDeviceInLoweringMode(self.device().index()) &&
      std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = ne_scalar_hpu_lazy(self, other);

    return t;
  } else {
    return ne_scalar_hpu(self, other);
  }
};
Tensor ne_tensor_hpu_wrap(const Tensor& self, const Tensor& other) {
  if (!habana_lazy::isDeviceInLoweringMode(self.device().index()) &&
      std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = ne_tensor_hpu_lazy(self, other);

    return t;
  } else {
    return ne_tensor_hpu(self, other);
  }
};
Tensor convolution_hpu_wrap(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return convolution_hpu_lazy(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups);
  } else {
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
  }
};
std::tuple<Tensor, Tensor, Tensor> convolution_backward_hpu_wrap(
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
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = convolution_backward_hpu_lazy(
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

    return t;
  } else {
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
  }
};
std::tuple<Tensor, Tensor, Tensor, Tensor> embedding_bag_hpu_wrap(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    bool scale_grad_by_freq,
    int64_t mode,
    UNUSED bool sparse,
    Tensor& per_sample_weights,
    UNUSED bool include_last_offset) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = embedding_bag_hpu_lazy(
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset);

    return t;
  } else {
    return embedding_bag_hpu(
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset);
  }
};
Tensor embedding_bag_bwd_hpu_wrap(
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
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = embedding_bag_bwd_hpu_lazy(
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

    return t;
  } else {
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
  }
};
Tensor constant_pad_hpu_wrap(
    const Tensor& self,
    IntArrayRef pad,
    Scalar value) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = constant_pad_hpu_lazy(self, pad, value);

    return t;
  } else {
    return constant_pad_hpu(self, pad, value);
  }
};
Tensor embedding_hpu_wrap(
    const Tensor& weight,
    const Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = embedding_hpu_lazy(
        weight, indices, padding_idx, scale_grad_by_freq, sparse);

    return t;
  } else {
    return embedding_hpu(
        weight, indices, padding_idx, scale_grad_by_freq, sparse);
  }
};
Tensor embedding_dense_backward_hpu_wrap(
    const Tensor& grad,
    const Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = embedding_dense_backward_hpu_lazy(
        grad, indices, num_weights, padding_idx, scale_grad_by_freq);

    return t;
  } else {
    return embedding_dense_backward_hpu(
        grad, indices, num_weights, padding_idx, scale_grad_by_freq);
  }
};
Tensor embedding_bag_sum_hpu_wrap(
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = embedding_bag_sum_hpu_lazy(
        input, indices, offsets, valid_count, kernel_mode);

    return t;
  } else {
    return embedding_bag_sum_hpu(
        input, indices, offsets, valid_count, kernel_mode);
  }
};
Tensor embedding_bag_sum_fwd_hpu_wrap(
    const Tensor& input,
    const Tensor& indices_fwd,
    const Tensor& offsets_fwd,
    const Tensor& valid_count,
    const Tensor& indices_bwd,
    const Tensor& offsets_bwd,
    const Tensor& valid_count_bwd,
    const Tensor& grad_weight) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = embedding_bag_sum_fwd_hpu_lazy(
        input,
        indices_fwd,
        offsets_fwd,
        valid_count,
        indices_bwd,
        offsets_bwd,
        valid_count_bwd,
        grad_weight);

    return t;
  } else {
    return embedding_bag_sum_fwd_hpu(
        input,
        indices_fwd,
        offsets_fwd,
        valid_count,
        indices_bwd,
        offsets_bwd,
        valid_count_bwd,
        grad_weight);
  }
};
Tensor& embedding_bag_sum_bwd_out_hpu_wrap(
    Tensor& out,
    const Tensor& input,
    const Tensor& indices_bwd,
    const Tensor& offsets_bwd,
    const Tensor& valid_count_bwd) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = embedding_bag_sum_bwd_out_hpu_lazy(
        out, input, indices_bwd, offsets_bwd, valid_count_bwd);

    return t;
  } else {
    return embedding_bag_sum_bwd_out_hpu(
        out, input, indices_bwd, offsets_bwd, valid_count_bwd);
  }
};
Tensor& embedding_bag_sum_bwd_out_kernel_mode_hpu_wrap(
    Tensor& out,
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = embedding_bag_sum_bwd_out_kernel_mode_hpu_lazy(
        out, input, indices, offsets, valid_count, kernel_mode);

    return t;
  } else {
    return embedding_bag_sum_bwd_out_kernel_mode_hpu(
        out, input, indices, offsets, valid_count, kernel_mode);
  }
};
Tensor& fill_hpu_wrap_(Tensor& self, Scalar value) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = fill_hpu_lazy_(self, value);

    return t;
  } else {
    return fill_hpu_(self, value);
  }
};
Tensor& masked_fill_hpu_wrap_(
    Tensor& self,
    const Tensor& mask,
    const Tensor& value) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = masked_fill_hpu_lazy_(self, mask, value);

    return t;
  } else {
    return masked_fill_hpu_(self, mask, value);
  }
};
Tensor& masked_fill_scalar_hpu_wrap_(
    Tensor& self,
    const Tensor& mask,
    Scalar value) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = masked_fill_scalar_hpu_lazy_(self, mask, value);

    return t;
  } else {
    return masked_fill_scalar_hpu_(self, mask, value);
  }
};
Tensor gather_src_hpu_wrap(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    bool sparse_grad) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = gather_src_hpu_lazy(self, dim_, index, sparse_grad);

    return t;
  } else {
    return gather_src_hpu(self, dim_, index, sparse_grad);
  }
};
Tensor& scatter_inplace_src_hpu_wrap(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = scatter_inplace_src_hpu_lazy(self, dim_, index, src);

    return t;
  } else {
    return scatter_inplace_src_hpu(self, dim_, index, src);
  }
};
Tensor scatter_src_hpu_wrap(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = scatter_src_hpu_lazy(self, dim_, index, src);

    return t;
  } else {
    return scatter_src_hpu(self, dim_, index, src);
  }
};
Tensor scatter_add_src_hpu_wrap(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = scatter_add_src_hpu_lazy(self, dim_, index, src);

    return t;
  } else {
    return scatter_add_src_hpu(self, dim_, index, src);
  }
};
Tensor& scatter_add_inplace_src_hpu_wrap(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = scatter_add_inplace_src_hpu_lazy(self, dim_, index, src);

    return t;
  } else {
    return scatter_add_inplace_src_hpu(self, dim_, index, src);
  }
};
Tensor& index_add_hpu_wrap_(
    Tensor& self,
    int64_t dim_,
    const Tensor& indices,
    const Tensor& source) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = index_add_hpu_lazy_(self, dim_, indices, source);

    return t;
  } else {
    return index_add_hpu_(self, dim_, indices, source);
  }
};
Tensor index_put_hpu_wrap(
    const Tensor& self,
    TensorList indices,
    const Tensor& value,
    bool accumulate) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = index_put_hpu_lazy(self, indices, value, accumulate);

    return t;
  } else {
    return index_put_hpu(self, indices, value, accumulate);
  }
};
Tensor& index_put_hpu_wrap_(
    Tensor& self,
    TensorList indices,
    const Tensor& value,
    bool accumulate) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = index_put_hpu_lazy_(self, indices, value, accumulate);

    return t;
  } else {
    return index_put_hpu_(self, indices, value, accumulate);
  }
};
Tensor index_select_hpu_wrap(
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = index_select_hpu_lazy(self, dim, index);

    return t;
  } else {
    return index_select_hpu(self, dim, index);
  }
};
Tensor gather2d_hpu_wrap(
    const Tensor& input,
    const Tensor& indices,
    int64_t validCount) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = gather2d_hpu_lazy(input, indices, validCount);

    return t;
  } else {
    return gather2d_hpu(input, indices, validCount);
  }
};
Tensor slice_hpu_wrap(
    const Tensor& self,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t step) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = slice_hpu_lazy(self, dim, start, end, step);

    return t;
  } else {
    return slice_hpu(self, dim, start, end, step);
  }
};
Tensor select_hpu_wrap(const Tensor& self, int64_t dim, int64_t index) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = select_hpu_lazy(self, dim, index);

    return t;
  } else {
    return select_hpu(self, dim, index);
  }
};
Tensor& arange_hpu_wrap(Tensor& output, Scalar start, Scalar end, Scalar step) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = arange_hpu_lazy(output, start, end, step);

    return t;
  } else {
    return arange_hpu(output, start, end, step);
  }
};
Tensor mm_hpu_wrap(const at::Tensor& mat1, const at::Tensor& mat2) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return mm_hpu_lazy(mat1, mat2);
  } else {
    return mm_hpu(mat1, mat2);
  }
};
Tensor addmm_hpu_wrap(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return addmm_hpu_lazy(self, mat1, mat2, beta, alpha);
  } else {
    return addmm_hpu(self, mat1, mat2, beta, alpha);
  }
};
Tensor& batch_gemm_out_hpu_wrap(
    Tensor& out,
    const Tensor& self,
    const Tensor& mat2) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = batch_gemm_out_hpu_lazy(out, self, mat2);
    return t;
  } else {
    return batch_gemm_out_hpu(out, self, mat2);
  }
};
Tensor batch_gemm_hpu_wrap(const Tensor& self, const Tensor& mat2) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = batch_gemm_hpu_lazy(self, mat2);
    return t;
  } else {
    return batch_gemm_hpu(self, mat2);
  }
};
Tensor dot_hpu_wrap(const Tensor& self, const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = dot_hpu_lazy(self, other);
    return t;
  } else {
    return dot_hpu(self, other);
  }
};
Tensor mv_hpu_wrap(const Tensor& self, const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = mv_hpu_lazy(self, other);
    return t;
  } else {
    return mv_hpu(self, other);
  }
};
std::tuple<Tensor, Tensor> nll_loss_forward_hpu_wrap(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = nll_loss_forward_hpu_lazy(
        self, target, weight, reduction, ignore_index);
    return t;
  } else {
    return nll_loss_forward_hpu(self, target, weight, reduction, ignore_index);
  }
};
Tensor nll_loss_backward_hpu_wrap(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    UNUSED const Tensor& total_weight) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = nll_loss_backward_hpu_lazy(
        grad_output,
        self,
        target,
        weight,
        reduction,
        ignore_index,
        total_weight);
    return t;
  } else {
    return nll_loss_backward_hpu(
        grad_output,
        self,
        target,
        weight,
        reduction,
        ignore_index,
        total_weight);
  }
};
Tensor mse_loss_forward_hpu_wrap(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = mse_loss_forward_hpu_lazy(self, target, reduction);
    return t;
  } else {
    return mse_loss_forward_hpu(self, target, reduction);
  }
};
Tensor mse_loss_backward_hpu_wrap(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = mse_loss_backward_hpu_lazy(grad_output, self, target, reduction);
    return t;
  } else {
    return mse_loss_backward_hpu(grad_output, self, target, reduction);
  }
};
Tensor binary_cross_entropy_hpu_wrap(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = binary_cross_entropy_hpu_lazy(self, target, weight, reduction);
    return t;
  } else {
    return binary_cross_entropy_hpu(self, target, weight, reduction);
  }
};
Tensor binary_cross_entropy_backward_hpu_wrap(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = binary_cross_entropy_backward_hpu_lazy(
        grad_output, self, target, weight, reduction);
    return t;
  } else {
    return binary_cross_entropy_backward_hpu(
        grad_output, self, target, weight, reduction);
  }
};
std::tuple<Tensor, Tensor, Tensor> batch_norm_hpu_wrap(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool training,
    double momentum,
    double eps) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = batch_norm_hpu_lazy(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps);
    return t;
  } else {
    return batch_norm_hpu(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps);
  }
};
std::tuple<Tensor, Tensor, Tensor> batch_norm_bwd_hpu_wrap(
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
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = batch_norm_bwd_hpu_lazy(
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
    return t;
  } else {
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
  }
};
std::tuple<Tensor, Tensor, Tensor> layer_norm_hpu_wrap(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int64_t m,
    int64_t n,
    double eps) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = layer_norm_hpu_lazy(input, weight, bias, m, n, eps);
    return t;
  } else {
    return layer_norm_hpu(input, weight, bias, m, n, eps);
  }
};
std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_hpu_wrap(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    std::array<bool, 3> grad_input_mask) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = layer_norm_backward_hpu_lazy(
        dY, X, mean, rstd, gamma, M, N, grad_input_mask);
    return t;
  } else {
    return layer_norm_backward_hpu(
        dY, X, mean, rstd, gamma, M, N, grad_input_mask);
  }
};
Tensor norm_scalar_hpu_wrap(const Tensor& self, Scalar p) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = norm_scalar_hpu_lazy(self, p);
    return t;
  } else {
    return norm_scalar_hpu(self, p);
  }
};
std::tuple<Tensor, Tensor> max_pool2d_with_indices_hpu_wrap(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return max_pool2d_with_indices_hpu_lazy(
        input, kernel_size, stride, padding, dilation, ceil_mode);
  } else {
    return max_pool2d_with_indices_hpu(
        input, kernel_size, stride, padding, dilation, ceil_mode);
  }
};
Tensor& max_pool2d_with_indices_backward_out_hpu_wrap(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& indices,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return max_pool2d_with_indices_backward_out_hpu_lazy(
        grad_input,
        grad_output,
        input,
        indices,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode);
  } else {
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
  }
};
Tensor max_pool2d_with_indices_backward_hpu_wrap(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return max_pool2d_with_indices_backward_hpu_lazy(
        grad_output,
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        indices);
  } else {
    return max_pool2d_with_indices_backward_hpu(
        grad_output,
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        indices);
  }
};
Tensor avg_pool2d_hpu_wrap(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = avg_pool2d_hpu_lazy(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
    return t;
  } else {
    return avg_pool2d_hpu(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
  }
};
Tensor& avg_pool2d_backward_out_hpu_wrap(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = avg_pool2d_backward_out_hpu_lazy(
        grad_input,
        grad_output,
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
    return t;
  } else {
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
  }
};
Tensor avg_pool2d_backward_hpu_wrap(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = avg_pool2d_backward_hpu_lazy(
        grad_output,
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
    return t;
  } else {
    return avg_pool2d_backward_hpu(
        grad_output,
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
  }
};
void uniform_hpu_wrap(
    const Tensor& self,
    double from,
    double to,
    CPUGenerator* gen) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    uniform_hpu_lazy(self, from, to, gen);
  } else {
    uniform_hpu(self, from, to, gen);
  }
};
void normal_hpu_wrap(
    const Tensor& self,
    double mean,
    double std,
    CPUGenerator* gen) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    normal_hpu_lazy(self, mean, std, gen);
  } else {
    normal_hpu(self, mean, std, gen);
  }
};
Tensor bernoulli_hpu_wrap(const Tensor& self, CPUGenerator* gen) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = bernoulli_hpu_lazy(self, gen);
    return t;
  } else {
    return bernoulli_hpu(self, gen);
  }
};
Tensor& bernoulli_scalar_hpu_wrap(Tensor& self, double p, CPUGenerator* gen) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = bernoulli_scalar_hpu_lazy(self, p, gen);
    return t;
  } else {
    return bernoulli_scalar_hpu(self, p, gen);
  }
};
std::tuple<Tensor, Tensor> fused_dropout_hpu_wrap(
    Tensor self,
    double p,
    CPUGenerator* gen) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    // auto& t = fused_dropout_hpu_lazy(self, p, gen);
    // return t;
    return fused_dropout_hpu_lazy(self, p, gen);
    ;
  } else {
    return fused_dropout_hpu(self, p, gen);
  }
};
Tensor sum_dim_IntList_hpu_wrap(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = sum_dim_IntList_hpu_lazy(self, dim, keepdim, dtype);
    return t;
  } else {
    return sum_dim_IntList_hpu(self, dim, keepdim, dtype);
  }
};
Tensor& sum_IntList_out_hpu_wrap(
    Tensor& output,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = sum_IntList_out_hpu_lazy(output, self, dim, keepdim, dtype);
    return t;
  } else {
    return sum_IntList_out_hpu(output, self, dim, keepdim, dtype);
  }
};
Tensor mean_dim_hpu_wrap(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = mean_dim_hpu_lazy(self, dim, keepdim, dtype);
    return t;
  } else {
    return mean_dim_hpu(self, dim, keepdim, dtype);
  }
};
Tensor& mean_dim_out_hpu_wrap(
    Tensor& output,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = mean_dim_out_hpu_lazy(output, self, dim, keepdim, dtype);
    return t;
  } else {
    return mean_dim_out_hpu(output, self, dim, keepdim, dtype);
  }
};
Tensor sum_hpu_wrap(const Tensor& self, c10::optional<ScalarType> dtype) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = sum_hpu_lazy(self, dtype);
    return t;
  } else {
    return sum_hpu(self, dtype);
  }
};
Tensor mean_hpu_wrap(const Tensor& self, c10::optional<ScalarType> dtype) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = mean_hpu_lazy(self, dtype);

    return t;
  } else {
    return mean_hpu(self, dtype);
  }
};
Tensor& any_dim_out_hpu_wrap(
    Tensor& output,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = any_dim_out_hpu_lazy(output, self, dim, keepdim);
    return t;
  } else {
    return any_dim_out_hpu(output, self, dim, keepdim);
  }
};
Tensor any_dim_hpu_wrap(const Tensor& self, int64_t dim, bool keepdim) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = any_dim_hpu_lazy(self, dim, keepdim);
    return t;
  } else {
    return any_dim_hpu(self, dim, keepdim);
  }
};
Tensor any_hpu_wrap(const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = any_hpu_lazy(self);
    return t;
  } else {
    return any_hpu(self);
  }
};
namespace habana {
Tensor log_softmax_hpu_wrap(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = log_softmax_hpu_lazy(self, dim, half_to_float);
    return t;
  } else {
    return log_softmax_hpu(self, dim, half_to_float);
  }
};
Tensor log_softmax_backward_hpu_wrap(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    const Tensor& input) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = log_softmax_backward_hpu_lazy(grad, output, dim, input);
    return t;
  } else {
    return log_softmax_backward_hpu(grad, output, dim, input);
  }
};
Tensor softmax_hpu_wrap(
    const Tensor& self,
    int64_t dim,
    const bool half_to_float) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = softmax_hpu_lazy(self, dim, half_to_float);
    return t;
  } else {
    return softmax_hpu(self, dim, half_to_float);
  }
};
Tensor softmax_backward_hpu_wrap(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    const Tensor& input) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = softmax_backward_hpu_lazy(grad, output, dim, input);
    return t;
  } else {
    return softmax_backward_hpu(grad, output, dim, input);
  }
};
} // namespace habana
namespace at {
namespace native {
Tensor empty_hpu_wrap(
    IntArrayRef size,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return empty_hpu_lazy(size, options, optional_memory_format);
  }
  return empty_hpu(size, options, optional_memory_format);
};
Tensor empty_strided_hpu_wrap(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return empty_strided_hpu_lazy(size, stride, options);
  }
  return empty_strided_hpu(size, stride, options);
};
} // namespace native
} // namespace at
Tensor clone_hpu_wrap(
    const Tensor& self,
    c10::optional<MemoryFormat> memory_format) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = clone_hpu_lazy(self, memory_format);
    return t;
  } else {
    return clone_hpu(self, memory_format);
  }
};
Tensor& zero_hpu_wrap(Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = zero_hpu_lazy(self);
    return t;
  } else {
    return zero_hpu(self);
  }
};
Tensor cat_hpu_wrap(const TensorList tensors, int64_t dim_) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = cat_hpu_lazy(tensors, dim_);
    return t;
  } else {
    return cat_hpu(tensors, dim_);
  }
};
Tensor& cat_hpu_wrap_out(
    Tensor& result,
    const TensorList tensors,
    int64_t dim_) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = cat_hpu_lazy_out(result, tensors, dim_);
    return t;
  } else {
    return cat_hpu_out(result, tensors, dim_);
  }
};
Tensor transpose_hpu_wrap(const Tensor& self, int64_t dim0_, int64_t dim1_) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = transpose_hpu_lazy(self, dim0_, dim1_);
    return t;
  } else {
    return transpose_hpu(self, dim0_, dim1_);
  }
};
Tensor& transpose_hpu_wrap_(Tensor& self, int64_t dim0_, int64_t dim1_) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = transpose_hpu_lazy_(self, dim0_, dim1_);
    return t;
  } else {
    return transpose_hpu_(self, dim0_, dim1_);
  }
};
Tensor t_hpu_wrap(const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = t_hpu_lazy(self);
    return t;
  } else {
    return t_hpu(self);
  }
};
Tensor& t_hpu_wrap_(Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = t_hpu_lazy_(self);
    return t;
  } else {
    return t_hpu_(self);
  }
};
Tensor permute_hpu_wrap(const Tensor& self, IntArrayRef dims_) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = permute_hpu_lazy(self, dims_);
    return t;
  } else {
    return permute_hpu(self, dims_);
  }
};
Tensor expand_hpu_wrap(const Tensor& self, IntArrayRef size, bool implicit) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = expand_hpu_lazy(self, size, implicit);
    return t;
  } else {
    return expand_hpu(self, size, implicit);
  }
};
std::vector<Tensor> split_with_sizes_hpu_wrap(
    const Tensor& self,
    IntArrayRef split_sizes,
    int64_t dim) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = split_with_sizes_hpu_lazy(self, split_sizes, dim);
    return t;
  } else {
    return split_with_sizes_hpu(self, split_sizes, dim);
  }
};
Tensor threshold_backward_hpu_wrap(
    const Tensor& grad_output,
    const Tensor& self,
    Scalar threshold) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = threshold_backward_hpu_lazy(grad_output, self, threshold);
    return t;
  } else {
    return threshold_backward_hpu(grad_output, self, threshold);
  }
};
std::tuple<Tensor&, Tensor&> topk_out_hpu_wrap(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool largest,
    bool sorted) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = topk_out_hpu_lazy(values, indices, self, k, dim_, largest, sorted);
    return t;
  } else {
    return topk_out_hpu(values, indices, self, k, dim_, largest, sorted);
  }
};
std::tuple<Tensor, Tensor> topk_hpu_wrap(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = topk_hpu_lazy(self, k, dim, largest, sorted);
    return t;
  } else {
    return topk_hpu(self, k, dim, largest, sorted);
  }
};
std::tuple<Tensor, Tensor> sort_hpu_wrap(
    const Tensor& self,
    int64_t dim,
    bool descending) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = sort_hpu_lazy(self, dim, descending);
    return t;
  } else {
    return sort_hpu(self, dim, descending);
  }
};
Tensor unary_op_hpu_wrap(
    const Tensor& input,
    std::string& node_type,
    UnaryOperator* Op) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = unary_op_hpu_lazy(input, node_type, Op);
    return t;
  } else {
    return unary_op_hpu(input, node_type, Op);
  }
};
Tensor unary_backward_op_hpu_wrap(
    const Tensor& grad_in,
    const Tensor& input,
    std::string& node_type,
    UnaryBackwardOperator* Op) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = unary_backward_op_hpu_lazy(grad_in, input, node_type, Op);
    return t;
  } else {
    return unary_backward_op_hpu(grad_in, input, node_type, Op);
  }
};
Tensor relu_hpu_wrap(const Tensor& input) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return relu_hpu_lazy(input);
  } else {
    return relu_hpu(input);
  }
};
Tensor& relu_hpu_wrap_(Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return relu_hpu_lazy_(self);
  } else {
    return relu_hpu_(self);
  }
};
Tensor sigmoid_hpu_wrap(const Tensor& input) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = sigmoid_hpu_lazy(input);
    return t;
  } else {
    return sigmoid_hpu(input);
  }
};
Tensor sigmoid_backward_hpu_wrap(const Tensor& grad_in, const Tensor& input) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = sigmoid_backward_hpu_lazy(grad_in, input);
    return t;
  } else {
    return sigmoid_backward_hpu(grad_in, input);
  }
};
Tensor sqrt_hpu_wrap(Tensor& input) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = sqrt_hpu_lazy(input);
    return t;
  } else {
    return sqrt_hpu(input);
  }
};
Tensor tanh_hpu_wrap(const Tensor& input) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = tanh_hpu_lazy(input);
    return t;
  } else {
    return tanh_hpu(input);
  }
};
Tensor& tanh_hpu_wrap_(Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = tanh_hpu_lazy_(self);
    return t;
  } else {
    return tanh_hpu_(self);
  }
};
Tensor& tanh_out_hpu_wrap(Tensor& out, Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = tanh_out_hpu_lazy(out, self);
    return t;
  } else {
    return tanh_out_hpu(out, self);
  }
};
Tensor tanh_backward_hpu_wrap(const Tensor& grad_in, const Tensor& input) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = tanh_backward_hpu_lazy(grad_in, input);
    return t;
  } else {
    return tanh_backward_hpu(grad_in, input);
  }
};
Tensor gelu_hpu_wrap(const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = gelu_hpu_lazy(self);
    return t;
  } else {
    return gelu_hpu(self);
  }
};
Tensor gelu_backward_hpu_wrap(const Tensor& grad, const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = gelu_backward_hpu_lazy(grad, self);
    return t;
  } else {
    return gelu_backward_hpu(grad, self);
  }
};
Tensor& erf_hpu_wrap_(Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = erf_hpu_lazy_(self);
    return t;
  } else {
    return erf_hpu_(self);
  }
};
Tensor erf_hpu_wrap(const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = erf_hpu_lazy(self);
    return t;
  } else {
    return erf_hpu(self);
  }
};
Tensor& exp_hpu_wrap_(Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = exp_hpu_lazy_(self);
    return t;
  } else {
    return exp_hpu_(self);
  }
};
Tensor exp_hpu_wrap(const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = exp_hpu_lazy(self);
    return t;
  } else {
    return exp_hpu(self);
  }
};
Tensor& neg_out_hpu_wrap(Tensor& result, const Tensor& input) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = neg_out_hpu_lazy(result, input);
    return t;
  } else {
    return neg_out_hpu(result, input);
  }
};
Tensor& reciprocal_hpu_wrap_(Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = reciprocal_hpu_lazy_(self);
    return t;
  } else {
    return reciprocal_hpu_(self);
  }
};
Tensor reciprocal_hpu_wrap(const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = reciprocal_hpu_lazy(self);
    return t;
  } else {
    return reciprocal_hpu(self);
  }
};
Tensor& reciprocal_out_hpu_wrap(Tensor& result, const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = reciprocal_out_hpu_lazy(result, self);
    return t;
  } else {
    return reciprocal_out_hpu(result, self);
  }
};
Tensor clamp_min_hpu_wrap(const Tensor& self, Scalar min) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = clamp_min_hpu_lazy(self, min);
    return t;
  } else {
    return clamp_min_hpu(self, min);
  }
};
Tensor& clamp_hpu_wrap_(
    Tensor& self,
    c10::optional<Scalar> min,
    c10::optional<Scalar> max) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = clamp_hpu_lazy_(self, min, max);
    return t;
  } else {
    return clamp_hpu_(self, min, max);
  }
};
Tensor clamp_hpu_wrap(
    const Tensor& self,
    c10::optional<Scalar> min,
    c10::optional<Scalar> max) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = clamp_hpu_lazy(self, min, max);
    return t;
  } else {
    return clamp_hpu(self, min, max);
  }
};
Tensor abs_hpu_wrap(const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = abs_hpu_lazy(self);
    return t;
  } else {
    return abs_hpu(self);
  }
};
Tensor neg_hpu_wrap(const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = neg_hpu_lazy(self);
    return t;
  } else {
    return neg_hpu(self);
  }
};
Tensor floor_hpu_wrap(const Tensor& input) {
  if (!habana_lazy::isDeviceInLoweringMode(input.device().index()) &&
      std::getenv("PT_HPU_LAZY_MODE")) {
    return floor_hpu_lazy(input);
  } else {
    return floor_hpu(input);
  }
};
Tensor& floor_hpu_wrap_(Tensor& self) {
  if (!habana_lazy::isDeviceInLoweringMode(self.device().index()) &&
      std::getenv("PT_HPU_LAZY_MODE")) {
    return floor_hpu_lazy_(self);
  } else {
    return floor_hpu_(self);
  }
};
namespace at {
namespace native {
Scalar _local_scalar_dense_hpu_wrap(const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = _local_scalar_dense_hpu_lazy(self);
    return t;
  } else {
    return _local_scalar_dense_hpu(self);
  }
}
} // namespace native
} // namespace at
std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_sgd_with_valid_count_hpu_wrap(
    const Tensor& gradients,
    Tensor& weights_in,
    Tensor& moments_in,
    const Tensor& indices,
    const Tensor& learning_rate,
    const Tensor& valid_count_tensor,
    float mom,
    bool nesterov) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return optimizer_sparse_sgd_with_valid_count_hpu_lazy(
        gradients,
        weights_in,
        moments_in,
        indices,
        learning_rate,
        valid_count_tensor,
        mom,
        nesterov);
  } else {
    return optimizer_sparse_sgd_with_valid_count_hpu(
        gradients,
        weights_in,
        moments_in,
        indices,
        learning_rate,
        valid_count_tensor,
        mom,
        nesterov);
  }
}
std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_adagrad_with_valid_count_hpu_wrap(
    const Tensor& gradients,
    Tensor& weights_in,
    Tensor& moments_in,
    const Tensor& indices,
    const Tensor& learning_rate,
    const Tensor& valid_count_tensor) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return optimizer_sparse_adagrad_with_valid_count_hpu_lazy(
        gradients,
        weights_in,
        moments_in,
        indices,
        learning_rate,
        valid_count_tensor);
  } else {
    return optimizer_sparse_adagrad_with_valid_count_hpu(
        gradients,
        weights_in,
        moments_in,
        indices,
        learning_rate,
        valid_count_tensor);
  }
}
void optimizer_adamw_hpu_wrap(
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
    const float weight_decay) {
  TORCH_CHECK((weight_vec.size() > 0), "Can not process empty weight vector");
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    optimizer_adamw_hpu_lazy(
        gradient_vec,
        weight_vec,
        exp_avg_vec,
        exp_avg_sq_vec,
        lr,
        beta1,
        beta2,
        epsilon,
        step,
        bias_correction,
        weight_decay);
  } else {
    optimizer_adamw_hpu(
        gradient_vec,
        weight_vec,
        exp_avg_vec,
        exp_avg_sq_vec,
        lr,
        beta1,
        beta2,
        epsilon,
        step,
        bias_correction,
        weight_decay);
  }
}
Tensor fused_norm_hpu_wrap(
    std::vector<at::Tensor>& grad,
    const Tensor& max_norm,
    float norm_type) {
  TORCH_CHECK((grad.size() > 0), "Can not process empty grad vector");
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return fused_norm_hpu_lazy(grad, max_norm, norm_type);
  } else {
    return fused_norm_hpu(grad, max_norm, norm_type);
  }
}
Tensor& optimizer_adagrad_hpu_wrap(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& variances,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float lrd,
    const float epsilon) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    optimizer_adagrad_hpu_lazy(
        gradients, weights, variances, epoch_num, lr, wd, lrd, epsilon);
  } else {
    optimizer_adagrad_hpu(
        gradients, weights, variances, epoch_num, lr, wd, lrd, epsilon);
  }

  return lr;
}

Tensor ones_like_hpu_wrap(
    const Tensor& self,
    const TensorOptions& options,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return ones_like_hpu_lazy(self, options, optional_memory_format);
  } else {
    return ones_like_hpu(self, options, optional_memory_format);
  }
}

std::tuple<Tensor, Tensor> matmul_backward_hpu_wrap(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return matmul_backward_hpu(grad_output, self, other);
  } else {
    return matmul_backward_hpu(grad_output, self, other);
  }
}

Tensor masked_scale_hpu_wrap(
    const Tensor& self,
    const Tensor& mask,
    double scale) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    HABANA_ASSERT(0); // Not supported for Lazy mode
    return masked_scale_hpu(self, mask, scale);
  } else {
    return masked_scale_hpu(self, mask, scale);
  }
}

Tensor habana_d2d_memcpy(const Tensor& self) {
  HABANA_ASSERT(0);
  return self;
}

Tensor habana_d2d_memcpy_other(const Tensor& self, Tensor& other) {
  HABANA_ASSERT(0);
  return self;
}

Tensor& cast_hpu_for_registration_only(Tensor& out, const Tensor& self) {
  // we should never reach here. This function is a dummy written
  // only to satisfy registration requirements.
  // Registration of custom cast OP (hpu::cast) is done so that
  // JIT optimization passes recognize this OP as a valid OP.
  HABANA_ASSERT(0);
}

Tensor& permute_cl_registration_only(Tensor& out) {
  // we should never reach here. This function is a dummy written
  // only to satisfy registration requirements.
  // Registration of custom permute_cl OP (hpu::permute_cl) is done so that
  HABANA_ASSERT(0);
}

Tensor& graph_connect_for_registration_only(Tensor& out, const Tensor& self) {
  // we should never reach here. This function is a dummy written
  // only to satisfy registration requirements.
  // Registration of custom cast OP (hpu::control_edge_) is done so that
  // JIT optimization passes recognize this OP as a valid OP.
  HABANA_ASSERT(0);
}

static auto
    registry =
        torch::
            RegisterOperators()
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(copy_hpu_wrap_),
                            &copy_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a)")
                        .impl_unboxedOnlyKernel<
                            decltype(as_strided_hpu_wrap),
                            &as_strided_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::set_.source_Storage_storage_offset( Tensor(a !) self, Storage source, int storage_offset, int[] size, int[] stride = []) ->Tensor(a !)")
                        .impl_unboxedOnlyKernel<
                            decltype(set_hpu_wrap_),
                            &set_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::view(Tensor(a) self, int[] size) -> Tensor(a)")
                        .impl_unboxedOnlyKernel<
                            decltype(view_hpu_wrap),
                            &view_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(addcmul_hpu_wrap),
                            &addcmul_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::addcmul_(Tensor(a !) self, Tensor tensor1, Tensor tensor2, *, Scalar value = 1) -> Tensor(a !)")
                        .impl_unboxedOnlyKernel<
                            decltype(addcmul_hpu_wrap_),
                            &addcmul_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(addcdiv_hpu_wrap),
                            &addcdiv_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(addcdiv_hpu_wrap_),
                            &addcdiv_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(add_tensor_hpu_wrap_),
                            &add_tensor_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(add_tensor_hpu_wrap),
                            &add_tensor_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::add.Scalar(Tensor self, Scalar other, Scalar alpha = 1) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(add_scalar_hpu_wrap),
                            &add_scalar_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(add_scalar_hpu_wrap_),
                                    &add_scalar_hpu_wrap_>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(sub_tensor_hpu_wrap),
                                    &sub_tensor_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(sub_tensor_hpu_wrap_),
                                    &sub_tensor_hpu_wrap_>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(sub_scalar_hpu_wrap),
                                    &sub_scalar_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(sub_scalar_hpu_wrap_),
                                    &sub_scalar_hpu_wrap_>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(mul_tensor_hpu_wrap_),
                                    &mul_tensor_hpu_wrap_>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(mul_tensor_hpu_wrap),
                                    &mul_tensor_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::mul.Scalar(Tensor self, Scalar other) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(mul_scalar_hpu_wrap),
                                    &mul_scalar_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(mul_scalar_hpu_wrap_),
                                    &mul_scalar_hpu_wrap_>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::_masked_scale(Tensor self, Tensor mask, float scale) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(masked_scale_hpu_wrap),
                                    &masked_scale_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::div.Tensor(Tensor self, Tensor other) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(div_tensor_hpu_wrap),
                                    &div_tensor_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(div_tensor_hpu_wrap_out),
                                    &div_tensor_hpu_wrap_out>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::div_.Tensor(Tensor(a!) self, Tensor other) -> (Tensor(a!))")
                                .impl_unboxedOnlyKernel<
                                    decltype(div_tensor_hpu_wrap_),
                                    &div_tensor_hpu_wrap_>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::div.Scalar(Tensor self, Scalar other) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(div_scalar_hpu_wrap),
                                    &div_scalar_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::div_.Scalar(Tensor(a!) self, Scalar other) -> (Tensor(a!))")
                                .impl_unboxedOnlyKernel<
                                    decltype(div_scalar_hpu_wrap_),
                                    &div_scalar_hpu_wrap_>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(pow_tensor_tensor_hpu_wrap),
                                    &pow_tensor_tensor_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::pow_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(pow_tensor_tensor_hpu_wrap_),
                                    &pow_tensor_tensor_hpu_wrap_>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(pow_tensor_scalar_hpu_wrap),
                                    &pow_tensor_scalar_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::pow_.Scalar(Tensor(a !) self, Scalar exponent) -> Tensor(a !)")
                                .impl_unboxedOnlyKernel<
                                    decltype(pow_tensor_scalar_hpu_wrap_),
                                    &pow_tensor_scalar_hpu_wrap_>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::pow.Scalar(Scalar self, Tensor exponent)->Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(pow_scalar_tensor_hpu_wrap),
                                    &pow_scalar_tensor_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(rsub_scalar_hpu_wrap),
                                    &rsub_scalar_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::gt.Tensor(Tensor self, Tensor other) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(gt_tensor_hpu_wrap),
                                    &gt_tensor_hpu_wrap>(DispatchKey::
                                                             HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::gt.Scalar(Tensor self, Scalar other) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(gt_scalar_hpu_wrap),
                                    &gt_scalar_hpu_wrap>(DispatchKey::
                                                             HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::eq.Tensor(Tensor self, Tensor other) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(eq_tensor_hpu_wrap),
                                    &eq_tensor_hpu_wrap>(DispatchKey::
                                                             HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(eq_tensor_out_hpu_wrap),
                                    &eq_tensor_out_hpu_wrap>(DispatchKey::
                                                                 HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::eq.Scalar(Tensor self, Scalar other) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(eq_tensor_scalar_hpu_wrap),
                                    &eq_tensor_scalar_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::lt.Scalar(Tensor self, Scalar other) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(lt_scalar_hpu_wrap),
                                    &lt_scalar_hpu_wrap>(DispatchKey::
                                                             HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::lt.Tensor(Tensor self, Tensor other) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(lt_tensor_hpu_wrap),
                                    &lt_tensor_hpu_wrap>(DispatchKey::
                                                             HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::ge.Scalar(Tensor self, Scalar other) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(ge_scalar_hpu_wrap),
                                    &ge_scalar_hpu_wrap>(DispatchKey::
                                                             HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::ge.Tensor(Tensor self, Tensor other) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(ge_tensor_hpu_wrap),
                                    &ge_tensor_hpu_wrap>(DispatchKey::
                                                             HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::ne.Scalar(Tensor self, Scalar other) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(ne_scalar_hpu_wrap),
                                    &ne_scalar_hpu_wrap>(DispatchKey::
                                                             HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::ne.Tensor(Tensor self, Tensor other) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(ne_tensor_hpu_wrap),
                                    &ne_tensor_hpu_wrap>(DispatchKey::
                                                             HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::convolution_overrideable(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(convolution_hpu_wrap),
                                    &convolution_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::convolution_backward_overrideable(Tensor grad_output, Tensor input, Tensor weight, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)")
                                .impl_unboxedOnlyKernel<
                                    decltype(convolution_backward_hpu_wrap),
                                    &convolution_backward_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False) -> (Tensor, Tensor, Tensor, Tensor)")
                                .impl_unboxedOnlyKernel<
                                    decltype(embedding_bag_hpu_wrap),
                                    &embedding_bag_hpu_wrap>(DispatchKey::
                                                                 HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::_embedding_bag_dense_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, int num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(embedding_bag_bwd_hpu_wrap),
                                    &embedding_bag_bwd_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(constant_pad_hpu_wrap),
                                    &constant_pad_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(embedding_hpu_wrap),
                                    &embedding_hpu_wrap>(DispatchKey::
                                                             HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::embedding_dense_backward(Tensor grad_output, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(embedding_dense_backward_hpu_wrap),
                                    &embedding_dense_backward_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::embedding_bag_sum_fwd(Tensor input, Tensor indices_fwd, Tensor offsets_fwd, Tensor valid_count_fwd, Tensor indices_bwd, Tensor offsets_bwd, Tensor valid_count_bwd, Tensor grad_weight) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(embedding_bag_sum_fwd_hpu_wrap),
                                    &embedding_bag_sum_fwd_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::embedding_bag_sum_bwd.out(Tensor input, Tensor indices_bwd, Tensor offsets_bwd, Tensor valid_count_bwd, *, Tensor(a!) out) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(
                                        embedding_bag_sum_bwd_out_hpu_wrap),
                                    &embedding_bag_sum_bwd_out_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(fill_hpu_wrap_),
                                    &fill_hpu_wrap_>(DispatchKey::
                                                         HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::masked_fill_.Tensor(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(masked_fill_hpu_wrap_),
                                    &masked_fill_hpu_wrap_>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)")
                            .impl_unboxedOnlyKernel<
                                decltype(masked_fill_scalar_hpu_wrap_),
                                &masked_fill_scalar_hpu_wrap_>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::index_select(Tensor self, int dim, Tensor index) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(index_select_hpu_wrap),
                                &index_select_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)")
                            .impl_unboxedOnlyKernel<
                                decltype(index_put_hpu_wrap_),
                                &index_put_hpu_wrap_>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(index_put_hpu_wrap),
                                &index_put_hpu_wrap>(DispatchKey::
                                                         HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)")
                            .impl_unboxedOnlyKernel<
                                decltype(index_add_hpu_wrap_),
                                &index_add_hpu_wrap_>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::scatter_.src(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)")
                            .impl_unboxedOnlyKernel<
                                decltype(scatter_inplace_src_hpu_wrap),
                                &scatter_inplace_src_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(scatter_src_hpu_wrap),
                                &scatter_src_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(gather_src_hpu_wrap),
                                &gather_src_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(scatter_add_src_hpu_wrap),
                                &scatter_add_src_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)")
                            .impl_unboxedOnlyKernel<
                                decltype(scatter_add_inplace_src_hpu_wrap),
                                &scatter_add_inplace_src_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::slice.Tensor(Tensor(a) self, int dim=0, int start=0, int end=9223372036854775807, int step=1) -> Tensor(a)")
                            .impl_unboxedOnlyKernel<
                                decltype(slice_hpu_wrap),
                                &slice_hpu_wrap>(DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)")
                            .impl_unboxedOnlyKernel<
                                decltype(select_hpu_wrap),
                                &select_hpu_wrap>(DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::arange.start_out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)")
                            .impl_unboxedOnlyKernel<
                                decltype(arange_hpu_wrap),
                                &arange_hpu_wrap>(DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::mm(Tensor self, Tensor mat2) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(mm_hpu_wrap),
                                &mm_hpu_wrap>(DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "hpu::mm_t(Tensor self, Tensor mat2, bool self_transposed, bool mat2_transposed) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(mm_hpu_wrap),
                                &mm_hpu_wrap>(DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta = 1, Scalar alpha = 1) ->Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(addmm_hpu_wrap),
                                &addmm_hpu_wrap>(DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)")
                            .impl_unboxedOnlyKernel<
                                decltype(batch_gemm_out_hpu_wrap),
                                &batch_gemm_out_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::bmm(Tensor self, Tensor mat2) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(batch_gemm_hpu_wrap),
                                &batch_gemm_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::dot(Tensor self, Tensor tensor) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(dot_hpu_wrap),
                                &dot_hpu_wrap>(DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema("aten::mv(Tensor self, Tensor vec)->Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(mv_hpu_wrap),
                                &mv_hpu_wrap>(DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) ->(Tensor output, Tensor total_weight)")
                            .impl_unboxedOnlyKernel<
                                decltype(nll_loss_forward_hpu_wrap),
                                &nll_loss_forward_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(nll_loss_backward_hpu_wrap),
                            &nll_loss_backward_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(mse_loss_forward_hpu_wrap),
                            &mse_loss_forward_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(mse_loss_backward_hpu_wrap),
                            &mse_loss_backward_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(binary_cross_entropy_hpu_wrap),
                            &binary_cross_entropy_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(binary_cross_entropy_backward_hpu_wrap),
                            &binary_cross_entropy_backward_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)")
                        .impl_unboxedOnlyKernel<
                            decltype(batch_norm_hpu_wrap),
                            &batch_norm_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")
                        .impl_unboxedOnlyKernel<
                            decltype(batch_norm_bwd_hpu_wrap),
                            &batch_norm_bwd_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::native_layer_norm(Tensor input, Tensor? weight, Tensor? bias, int M, int N, float eps) -> (Tensor, Tensor, Tensor)")
                        .impl_unboxedOnlyKernel<
                            decltype(layer_norm_hpu_wrap),
                            &layer_norm_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::native_layer_norm_backward(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, int M, int N, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")
                        .impl_unboxedOnlyKernel<
                            decltype(layer_norm_backward_hpu_wrap),
                            &layer_norm_backward_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::norm.Scalar(Tensor self, Scalar p=2) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(norm_scalar_hpu_wrap),
                            &norm_scalar_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride = [], int[2] padding = 0, int[2] dilation = 1, bool ceil_mode = False) ->(Tensor, Tensor)")
                        .impl_unboxedOnlyKernel<
                            decltype(max_pool2d_with_indices_hpu_wrap),
                            &max_pool2d_with_indices_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(max_pool2d_with_indices_backward_hpu_wrap),
                            &max_pool2d_with_indices_backward_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema("aten::max_pool2d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)")
                            .impl_unboxedOnlyKernel<
                                decltype(
                                    max_pool2d_with_indices_backward_out_hpu_wrap),
                                &max_pool2d_with_indices_backward_out_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(avg_pool2d_hpu_wrap),
                            &avg_pool2d_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(avg_pool2d_backward_hpu_wrap),
                            &avg_pool2d_backward_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(uniform_hpu_wrap),
                            &uniform_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(normal_hpu_wrap),
                            &normal_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::bernoulli(Tensor self, *, Generator? generator=None) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(bernoulli_hpu_wrap),
                            &bernoulli_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(bernoulli_scalar_hpu_wrap),
                            &bernoulli_scalar_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::_fused_dropout(Tensor self, float p, Generator? generator=None) -> (Tensor, Tensor)")
                        .impl_unboxedOnlyKernel<
                            decltype(fused_dropout_hpu_wrap),
                            &fused_dropout_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(sum_dim_IntList_hpu_wrap),
                            &sum_dim_IntList_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::sum_dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(sum_dim_IntList_hpu_wrap),
                            &sum_dim_IntList_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::sum.IntList_out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(sum_IntList_out_hpu_wrap),
                            &sum_IntList_out_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(mean_dim_hpu_wrap),
                            &mean_dim_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::mean.out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(mean_dim_out_hpu_wrap),
                            &mean_dim_out_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(sum_hpu_wrap),
                            &sum_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(mean_hpu_wrap),
                            &mean_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(any_dim_hpu_wrap),
                            &any_dim_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::any(Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(any_hpu_wrap),
                            &any_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::any.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(any_dim_out_hpu_wrap),
                            &any_dim_out_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::_local_scalar_dense(Tensor self) -> Scalar")
                        .impl_unboxedOnlyKernel<
                            decltype(at::native::_local_scalar_dense_hpu_wrap),
                            &at::native::_local_scalar_dense_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(log_softmax_hpu_wrap),
                            &log_softmax_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(habana::log_softmax_backward_hpu_wrap),
                            &habana::log_softmax_backward_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(habana::softmax_hpu_wrap),
                            &habana::softmax_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(habana::softmax_backward_hpu_wrap),
                            &habana::softmax_backward_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(clone_hpu_wrap),
                            &clone_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(at::native::empty_hpu_wrap),
                            &at::native::empty_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::empty_strided(int[] size, int[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(at::native::empty_strided_hpu_wrap),
                            &at::native::empty_strided_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::zero_(Tensor(a!) self) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(zero_hpu_wrap),
                            &zero_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)")
                        .impl_unboxedOnlyKernel<
                            decltype(permute_hpu_wrap),
                            &permute_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)")
                        .impl_unboxedOnlyKernel<
                            decltype(expand_hpu_wrap),
                            &expand_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::cat(Tensor[] tensors, int dim=0) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(cat_hpu_wrap),
                            &cat_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(cat_hpu_wrap_out),
                            &cat_hpu_wrap_out>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::_cat(Tensor[] tensors, int dim=0) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(cat_hpu_wrap),
                            &cat_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::_cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(cat_hpu_wrap_out),
                            &cat_hpu_wrap_out>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::split_with_sizes(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]")
                        .impl_unboxedOnlyKernel<
                            decltype(split_with_sizes_hpu_wrap),
                            &split_with_sizes_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)")
                        .impl_unboxedOnlyKernel<
                            decltype(transpose_hpu_wrap),
                            &transpose_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(transpose_hpu_wrap_),
                            &transpose_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::t(Tensor(a) self) -> Tensor(a)")
                        .impl_unboxedOnlyKernel<
                            decltype(t_hpu_wrap),
                            &t_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::t_(Tensor(a!) self) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(t_hpu_wrap_),
                            &t_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(threshold_backward_hpu_wrap),
                            &threshold_backward_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)")
                        .impl_unboxedOnlyKernel<
                            decltype(topk_hpu_wrap),
                            &topk_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::topk.values(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True, *, Tensor(a!) values, Tensor(b!) indices) ->(Tensor(a!) values, Tensor(b!) indices)")
                        .impl_unboxedOnlyKernel<
                            decltype(topk_out_hpu_wrap),
                            &topk_out_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)")
                        .impl_unboxedOnlyKernel<
                            decltype(sort_hpu_wrap),
                            &sort_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::relu_(Tensor(a!) self) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(relu_hpu_wrap_),
                            &relu_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::relu(Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(relu_hpu_wrap),
                            &relu_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::sigmoid(Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(sigmoid_hpu_wrap),
                            &sigmoid_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(sigmoid_backward_hpu_wrap),
                            &sigmoid_backward_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::sqrt(Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(sqrt_hpu_wrap),
                            &sqrt_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::tanh(Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(tanh_hpu_wrap),
                            &tanh_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::tanh_backward(Tensor grad_output, Tensor output)->Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(tanh_backward_hpu_wrap),
                            &tanh_backward_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::tanh_(Tensor(a!) self) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(tanh_hpu_wrap_),
                            &tanh_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(tanh_out_hpu_wrap),
                            &tanh_out_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::gelu(Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(gelu_hpu_wrap),
                            &gelu_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::gelu_backward(Tensor grad, Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(gelu_backward_hpu_wrap),
                            &gelu_backward_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::erf_(Tensor(a!) self) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(erf_hpu_wrap_),
                            &erf_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::erf(Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(erf_hpu_wrap),
                            &erf_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::exp_(Tensor(a!) self) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(exp_hpu_wrap_),
                            &exp_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::exp(Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(exp_hpu_wrap),
                            &exp_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(neg_out_hpu_wrap),
                            &neg_out_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(reciprocal_hpu_wrap_),
                            &reciprocal_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::reciprocal(Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(reciprocal_hpu_wrap),
                            &reciprocal_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::reciprocal.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(reciprocal_out_hpu_wrap),
                            &reciprocal_out_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::clamp_min(Tensor self, Scalar min) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(clamp_min_hpu_wrap),
                            &clamp_min_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(clamp_hpu_wrap_),
                            &clamp_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(clamp_hpu_wrap),
                            &clamp_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::abs(Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(abs_hpu_wrap),
                            &abs_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::neg(Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(neg_hpu_wrap),
                            &neg_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::ones_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(ones_like_hpu_wrap),
                            &ones_like_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::matmul_backward(Tensor grad_out, Tensor self, Tensor other) -> (Tensor, Tensor)")
                        .impl_unboxedOnlyKernel<
                            decltype(matmul_backward_hpu_wrap),
                            &matmul_backward_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "hpu::embedding_bag_sum(Tensor input, Tensor indices, Tensor offsets, Tensor valid_count, int64_t kernel_mode) -> (Tensor)")
                        .impl_unboxedOnlyKernel<
                            decltype(embedding_bag_sum_hpu_wrap),
                            &embedding_bag_sum_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "hpu::embedding_bag_sum_bwd_out(Tensor out, Tensor input, Tensor indices_bwd, Tensor offsets_bwd, Tensor valid_count_bwd, int64_t kernel_mode) -> (Tensor)")
                        .impl_unboxedOnlyKernel<
                            decltype(
                                embedding_bag_sum_bwd_out_kernel_mode_hpu_wrap),
                            &embedding_bag_sum_bwd_out_kernel_mode_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema("hpu::habanaOptimizerSparseSgd(Tensor gradients, Tensor weights_in, Tensor moments_in, Tensor indices, Tensor learning_rate, Tensor valid_count_tensor, float mom, bool nesterov) -> (Tensor, Tensor)")
                                .impl_unboxedOnlyKernel<
                                    decltype(
                                        optimizer_sparse_sgd_with_valid_count_hpu_wrap),
                                    &optimizer_sparse_sgd_with_valid_count_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "hpu::habana_d2d_memcpy(Tensor self) -> (Tensor)")
                        .impl_unboxedOnlyKernel<
                            decltype(habana_d2d_memcpy),
                            &habana_d2d_memcpy>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "hpu::habana_d2d_memcpy_other(Tensor self, Tensor other) -> (Tensor)")
                        .impl_unboxedOnlyKernel<
                            decltype(habana_d2d_memcpy_other),
                            &habana_d2d_memcpy_other>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "hpu::habanaOptimizerSparseAdagrad(Tensor gradients, Tensor weights_in, Tensor moments_in, Tensor indices, Tensor learning_rate, Tensor valid_count_tensor) -> (Tensor, Tensor)")
                        .impl_unboxedOnlyKernel<
                            decltype(
                                optimizer_sparse_adagrad_with_valid_count_hpu_wrap),
                            &optimizer_sparse_adagrad_with_valid_count_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "hpu::habanaOptimizerFusedAdagrad(Tensor[] gradients, Tensor[] weights_in, Tensor[] variances_in, Tensor epoch_num, Tensor learning_rate, float wd, float lrd, float eps) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(optimizer_adagrad_hpu_wrap),
                            &optimizer_adagrad_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "hpu::cast(Tensor self, Scalar type) -> Tensor(a)")
                        .impl_unboxedOnlyKernel<
                            decltype(cast_hpu_for_registration_only),
                            &cast_hpu_for_registration_only>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::floor_(Tensor(a!) self) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(floor_hpu_wrap_),
                            &floor_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::floor(Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(floor_hpu_wrap),
                            &floor_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "hpu::permute_cl(Tensor(a) self, int[] dims) -> Tensor(a)")
                        .impl_unboxedOnlyKernel<
                            decltype(permute_cl_registration_only),
                            &permute_cl_registration_only>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "hpu::control_edge_other_(Tensor self, Tensor other) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(graph_connect_for_registration_only),
                            &graph_connect_for_registration_only>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("hpu::control_edge_(Tensor self) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(graph_connect_for_registration_only),
                            &graph_connect_for_registration_only>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
