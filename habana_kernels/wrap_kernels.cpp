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

Tensor& copy_hpu_wrap_(Tensor& self, const Tensor& src, bool non_blocking) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return copy_hpu_lazy_(self, src, non_blocking);
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
    return as_strided_hpu_lazy(self, size, stride, storage_offset);
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
    return set_hpu_lazy_(self, source, storage_offset, size, stride);
  } else {
    return set_hpu_(self, source, storage_offset, size, stride);
  }
};
Tensor view_hpu_wrap(const Tensor& self, IntArrayRef size) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return view_hpu_lazy(self, size);
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
    return addcmul_hpu_lazy(self, tensor1, tensor2, alpha);
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
    return addcmul_hpu_lazy_(self, tensor1, tensor2, alpha);
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
    return addcdiv_hpu_lazy(self, tensor1, tensor2, alpha);
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
    return addcdiv_hpu_lazy_(self, tensor1, tensor2, alpha);
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
    return add_scalar_hpu_lazy(self, other, alpha);
  } else {
    return add_scalar_hpu(self, other, alpha);
  }
};
Tensor& add_scalar_hpu_wrap_(Tensor& self, Scalar other, Scalar alpha) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return add_scalar_hpu_lazy_(self, other, alpha);
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
    return sub_tensor_hpu_lazy(self, other, alpha);
  } else {
    return sub_tensor_hpu(self, other, alpha);
  }
};
Tensor& sub_tensor_hpu_wrap_(Tensor& self, const Tensor& other, Scalar alpha) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return sub_tensor_hpu_lazy_(self, other, alpha);
  } else {
    return sub_tensor_hpu_(self, other, alpha);
  }
};
Tensor sub_scalar_hpu_wrap(const Tensor& self, Scalar other, Scalar alpha) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return sub_scalar_hpu_lazy(self, other, alpha);
  } else {
    return sub_scalar_hpu(self, other, alpha);
  }
};
Tensor& sub_scalar_hpu_wrap_(Tensor& self, Scalar other, Scalar alpha) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return sub_scalar_hpu_lazy_(self, other, alpha);
  } else {
    return sub_scalar_hpu_(self, other, alpha);
  }
};
Tensor rsub_scalar_hpu_wrap(const Tensor& self, Scalar other, Scalar alpha) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return rsub_scalar_hpu_lazy(self, other, alpha);
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
    return mul_scalar_hpu_lazy(self, other);
  } else {
    return mul_scalar_hpu(self, other);
  }
};
Tensor& mul_scalar_hpu_wrap_(Tensor& self, Scalar other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return mul_scalar_hpu_lazy_(self, other);
  } else {
    return mul_scalar_hpu_(self, other);
  }
};
Tensor div_tensor_hpu_wrap(const Tensor& self, const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return div_tensor_hpu_lazy(self, other);
  } else {
    return div_tensor_hpu(self, other);
  }
};
Tensor& div_tensor_hpu_wrap_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return div_tensor_hpu_lazy_out(result, self, other);
  } else {
    return div_tensor_hpu_out(result, self, other);
  }
};
Tensor& div_tensor_hpu_wrap_(Tensor& self, const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return div_tensor_hpu_lazy_(self, other);
  } else {
    return div_tensor_hpu_(self, other);
  }
};
Tensor div_scalar_hpu_wrap(const Tensor& self, Scalar other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return div_scalar_hpu_lazy(self, other);
  } else {
    return div_scalar_hpu(self, other);
  }
};
Tensor& div_scalar_hpu_wrap_(Tensor& self, Scalar other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return div_scalar_hpu_lazy_(self, other);
  } else {
    return div_scalar_hpu_(self, other);
  }
};
Tensor pow_tensor_tensor_hpu_wrap(const Tensor& self, const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return pow_tensor_tensor_hpu_lazy(self, other);
  } else {
    return pow_tensor_tensor_hpu(self, other);
  }
};
Tensor& pow_tensor_tensor_hpu_wrap_(Tensor& self, const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return pow_tensor_tensor_hpu_lazy_(self, other);
  } else {
    return pow_tensor_tensor_hpu_(self, other);
  }
};
Tensor pow_tensor_scalar_hpu_wrap(const Tensor& self, Scalar other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return pow_tensor_scalar_hpu_lazy(self, other);
  } else {
    return pow_tensor_scalar_hpu(self, other);
  }
};
Tensor& pow_tensor_scalar_hpu_wrap_(Tensor& self, Scalar other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return pow_tensor_scalar_hpu_lazy_(self, other);
  } else {
    return pow_tensor_scalar_hpu_(self, other);
  }
};
Tensor pow_scalar_tensor_hpu_wrap(Scalar other, const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return pow_scalar_tensor_hpu_lazy(other, self);
  } else {
    return pow_scalar_tensor_hpu(other, self);
  }
};
Tensor gt_hpu_wrap(Tensor& self, Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return gt_hpu_lazy(self, other);
  } else {
    return gt_hpu(self, other);
  }
};
void eq_tensor_out_hpu_wrap(
    Tensor& output,
    const Tensor& self,
    const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return eq_tensor_out_hpu_lazy(output, self, other);
  } else {
    return eq_tensor_out_hpu(output, self, other);
  }
};
Tensor eq_tensor_hpu_wrap(Tensor& self, Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return eq_tensor_hpu_lazy(self, other);
  } else {
    return eq_tensor_hpu(self, other);
  }
};
Tensor eq_tensor_scalar_hpu_wrap(Tensor& self, Scalar other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return eq_tensor_scalar_hpu_lazy(self, other);
  } else {
    return eq_tensor_scalar_hpu(self, other);
  }
};
Tensor lt_scalar_hpu_wrap(Tensor& self, Scalar other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return lt_scalar_hpu_lazy(self, other);
  } else {
    return lt_scalar_hpu(self, other);
  }
};
Tensor lt_tensor_hpu_wrap(Tensor& self, Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return lt_tensor_hpu_lazy(self, other);
  } else {
    return lt_tensor_hpu(self, other);
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
    return convolution_backward_hpu_lazy(
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
    return embedding_bag_hpu_lazy(
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset);
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
    return embedding_bag_bwd_hpu_lazy(
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
    return constant_pad_hpu_lazy(self, pad, value);
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
    return embedding_hpu_lazy(
        weight, indices, padding_idx, scale_grad_by_freq, sparse);
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
    return embedding_dense_backward_hpu_lazy(
        grad, indices, num_weights, padding_idx, scale_grad_by_freq);
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
    return embedding_bag_sum_hpu_lazy(
        input, indices, offsets, valid_count, kernel_mode);
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
    return embedding_bag_sum_fwd_hpu_lazy(
        input,
        indices_fwd,
        offsets_fwd,
        valid_count,
        indices_bwd,
        offsets_bwd,
        valid_count_bwd,
        grad_weight);
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
    return embedding_bag_sum_bwd_out_hpu_lazy(
        out, input, indices_bwd, offsets_bwd, valid_count_bwd);
  } else {
    return embedding_bag_sum_bwd_out_hpu(
        out, input, indices_bwd, offsets_bwd, valid_count_bwd);
  }
};
Tensor& fill_hpu_wrap_(Tensor& self, Scalar value) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return fill_hpu_lazy_(self, value);
  } else {
    return fill_hpu_(self, value);
  }
};
Tensor& masked_fill_hpu_wrap_(
    Tensor& self,
    const Tensor& mask,
    const Tensor& value) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return masked_fill_hpu_lazy_(self, mask, value);
  } else {
    return masked_fill_hpu_(self, mask, value);
  }
};
Tensor& masked_fill_scalar_hpu_wrap_(
    Tensor& self,
    const Tensor& mask,
    Scalar value) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return masked_fill_scalar_hpu_lazy_(self, mask, value);
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
    return gather_src_hpu_lazy(self, dim_, index, sparse_grad);
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
    return scatter_inplace_src_hpu_lazy(self, dim_, index, src);
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
    return scatter_src_hpu_lazy(self, dim_, index, src);
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
    return scatter_add_src_hpu_lazy(self, dim_, index, src);
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
    return scatter_add_inplace_src_hpu_lazy(self, dim_, index, src);
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
    return index_add_hpu_lazy_(self, dim_, indices, source);
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
    return index_put_hpu_lazy(self, indices, value, accumulate);
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
    return index_put_hpu_lazy_(self, indices, value, accumulate);
  } else {
    return index_put_hpu_(self, indices, value, accumulate);
  }
};
Tensor index_select_hpu_wrap(
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return index_select_hpu_lazy(self, dim, index);
  } else {
    return index_select_hpu(self, dim, index);
  }
};
Tensor gather2d_hpu_wrap(
    const Tensor& input,
    const Tensor& indices,
    int64_t validCount) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return gather2d_hpu_lazy(input, indices, validCount);
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
    return slice_hpu_lazy(self, dim, start, end, step);
  } else {
    return slice_hpu(self, dim, start, end, step);
  }
};
Tensor select_hpu_wrap(const Tensor& self, int64_t dim, int64_t index) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return select_hpu_lazy(self, dim, index);
  } else {
    return select_hpu(self, dim, index);
  }
};
Tensor& arange_hpu_wrap(Tensor& output, Scalar start, Scalar end, Scalar step) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return arange_hpu_lazy(output, start, end, step);
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
    return batch_gemm_out_hpu_lazy(out, self, mat2);
  } else {
    return batch_gemm_out_hpu(out, self, mat2);
  }
};
Tensor batch_gemm_hpu_wrap(const Tensor& self, const Tensor& mat2) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return batch_gemm_hpu_lazy(self, mat2);
  } else {
    return batch_gemm_hpu(self, mat2);
  }
};
Tensor dot_hpu_wrap(const Tensor& self, const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return dot_hpu_lazy(self, other);
  } else {
    return dot_hpu(self, other);
  }
};
Tensor mv_hpu_wrap(const Tensor& self, const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return mv_hpu_lazy(self, other);
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
    return nll_loss_forward_hpu_lazy(
        self, target, weight, reduction, ignore_index);
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
    return nll_loss_backward_hpu_lazy(
        grad_output,
        self,
        target,
        weight,
        reduction,
        ignore_index,
        total_weight);
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
    return mse_loss_forward_hpu_lazy(self, target, reduction);
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
    return mse_loss_backward_hpu_lazy(grad_output, self, target, reduction);
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
    return binary_cross_entropy_hpu_lazy(self, target, weight, reduction);
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
    return binary_cross_entropy_backward_hpu_lazy(
        grad_output, self, target, weight, reduction);
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
    return batch_norm_hpu_lazy(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps);
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
    return batch_norm_bwd_hpu_lazy(
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
    return layer_norm_hpu_lazy(input, weight, bias, m, n, eps);
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
    return layer_norm_backward_hpu_lazy(
        dY, X, mean, rstd, gamma, M, N, grad_input_mask);
  } else {
    return layer_norm_backward_hpu(
        dY, X, mean, rstd, gamma, M, N, grad_input_mask);
  }
};
Tensor norm_scalar_hpu_wrap(const Tensor& self, Scalar p) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return norm_scalar_hpu_lazy(self, p);
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
    return avg_pool2d_hpu_lazy(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
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
    return avg_pool2d_backward_out_hpu_lazy(
        grad_input,
        grad_output,
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
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
    return avg_pool2d_backward_hpu_lazy(
        grad_output,
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
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
    return uniform_hpu_lazy(self, from, to, gen);
  } else {
    return uniform_hpu(self, from, to, gen);
  }
};
void normal_hpu_wrap(
    const Tensor& self,
    double mean,
    double std,
    CPUGenerator* gen) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return normal_hpu_lazy(self, mean, std, gen);
  } else {
    return normal_hpu(self, mean, std, gen);
  }
};
Tensor bernoulli_hpu_wrap(const Tensor& self, CPUGenerator* gen) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return bernoulli_hpu_lazy(self, gen);
  } else {
    return bernoulli_hpu(self, gen);
  }
};
Tensor& bernoulli_scalar_hpu_wrap(Tensor& self, double p, CPUGenerator* gen) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return bernoulli_scalar_hpu_lazy(self, p, gen);
  } else {
    return bernoulli_scalar_hpu(self, p, gen);
  }
};
Tensor sum_dim_IntList_hpu_wrap(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return sum_dim_IntList_hpu_lazy(self, dim, keepdim, dtype);
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
    return sum_IntList_out_hpu_lazy(output, self, dim, keepdim, dtype);
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
    return mean_dim_hpu_lazy(self, dim, keepdim, dtype);
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
    return mean_dim_out_hpu_lazy(output, self, dim, keepdim, dtype);
  } else {
    return mean_dim_out_hpu(output, self, dim, keepdim, dtype);
  }
};
Tensor sum_hpu_wrap(const Tensor& self, c10::optional<ScalarType> dtype) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return sum_hpu_lazy(self, dtype);
  } else {
    return sum_hpu(self, dtype);
  }
};
Tensor mean_hpu_wrap(const Tensor& self, c10::optional<ScalarType> dtype) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return mean_hpu_lazy(self, dtype);
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
    return any_dim_out_hpu_lazy(output, self, dim, keepdim);
  } else {
    return any_dim_out_hpu(output, self, dim, keepdim);
  }
};
Tensor any_dim_hpu_wrap(const Tensor& self, int64_t dim, bool keepdim) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return any_dim_hpu_lazy(self, dim, keepdim);
  } else {
    return any_dim_hpu(self, dim, keepdim);
  }
};
Tensor any_hpu_wrap(const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return any_hpu_lazy(self);
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
    return log_softmax_hpu_lazy(self, dim, half_to_float);
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
    return log_softmax_backward_hpu_lazy(grad, output, dim, input);
  } else {
    return log_softmax_backward_hpu(grad, output, dim, input);
  }
};
Tensor softmax_hpu_wrap(
    const Tensor& self,
    int64_t dim,
    const bool half_to_float) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return softmax_hpu_lazy(self, dim, half_to_float);
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
    return softmax_backward_hpu_lazy(grad, output, dim, input);
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
  } else {
    return empty_hpu(size, options, optional_memory_format);
  }
};
Tensor empty_strided_hpu_wrap(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return empty_strided_hpu_lazy(size, stride, options);
  } else {
    return empty_strided_hpu(size, stride, options);
  }
};
} // namespace native
} // namespace at
Tensor clone_hpu_wrap(
    const Tensor& self,
    c10::optional<MemoryFormat> memory_format) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return clone_hpu_lazy(self, memory_format);
  } else {
    return clone_hpu(self, memory_format);
  }
};
Tensor& zero_hpu_wrap(Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return zero_hpu_lazy(self);
  } else {
    return zero_hpu(self);
  }
};
Tensor cat_hpu_wrap(const TensorList tensors, int64_t dim_) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return cat_hpu_lazy(tensors, dim_);
  } else {
    return cat_hpu(tensors, dim_);
  }
};
Tensor& cat_hpu_wrap_out(
    Tensor& result,
    const TensorList tensors,
    int64_t dim_) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return cat_hpu_lazy_out(result, tensors, dim_);
  } else {
    return cat_hpu_out(result, tensors, dim_);
  }
};
Tensor transpose_hpu_wrap(const Tensor& self, int64_t dim0_, int64_t dim1_) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return transpose_hpu_lazy(self, dim0_, dim1_);
  } else {
    return transpose_hpu(self, dim0_, dim1_);
  }
};
Tensor& transpose_hpu_wrap_(Tensor& self, int64_t dim0_, int64_t dim1_) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return transpose_hpu_lazy_(self, dim0_, dim1_);
  } else {
    return transpose_hpu_(self, dim0_, dim1_);
  }
};
Tensor t_hpu_wrap(const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return t_hpu_lazy(self);
  } else {
    return t_hpu(self);
  }
};
Tensor& t_hpu_wrap_(Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return t_hpu_lazy_(self);
  } else {
    return t_hpu_(self);
  }
};
Tensor permute_hpu_wrap(const Tensor& self, IntArrayRef dims_) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return permute_hpu_lazy(self, dims_);
  } else {
    return permute_hpu(self, dims_);
  }
};
Tensor expand_hpu_wrap(const Tensor& self, IntArrayRef size, bool implicit) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return expand_hpu_lazy(self, size, implicit);
  } else {
    return expand_hpu(self, size, implicit);
  }
};
std::vector<Tensor> split_with_sizes_hpu_wrap(
    const Tensor& self,
    IntArrayRef split_sizes,
    int64_t dim) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return split_with_sizes_hpu_lazy(self, split_sizes, dim);
  } else {
    return split_with_sizes_hpu(self, split_sizes, dim);
  }
};
Tensor threshold_backward_hpu_wrap(
    const Tensor& grad_output,
    const Tensor& self,
    Scalar threshold) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return threshold_backward_hpu_lazy(grad_output, self, threshold);
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
    return topk_out_hpu_lazy(values, indices, self, k, dim_, largest, sorted);
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
    return topk_hpu_lazy(self, k, dim, largest, sorted);
  } else {
    return topk_hpu(self, k, dim, largest, sorted);
  }
};
std::tuple<Tensor, Tensor> sort_hpu_wrap(
    const Tensor& self,
    int64_t dim,
    bool descending) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return sort_hpu_lazy(self, dim, descending);
  } else {
    return sort_hpu(self, dim, descending);
  }
};
Tensor unary_op_hpu_wrap(
    const Tensor& input,
    std::string& node_type,
    UnaryOperator* Op) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return unary_op_hpu_lazy(input, node_type, Op);
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
    return unary_backward_op_hpu_lazy(grad_in, input, node_type, Op);
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
    return sigmoid_hpu_lazy(input);
  } else {
    return sigmoid_hpu(input);
  }
};
Tensor sigmoid_backward_hpu_wrap(const Tensor& grad_in, const Tensor& input) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return sigmoid_backward_hpu_lazy(grad_in, input);
  } else {
    return sigmoid_backward_hpu(grad_in, input);
  }
};
Tensor sqrt_hpu_wrap(const Tensor& input) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return sqrt_hpu_lazy(input);
  } else {
    return sqrt_hpu(input);
  }
};
Tensor tanh_hpu_wrap(const Tensor& input) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return tanh_hpu_lazy(input);
  } else {
    return tanh_hpu(input);
  }
};
Tensor& tanh_hpu_wrap_(Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return tanh_hpu_lazy_(self);
  } else {
    return tanh_hpu_(self);
  }
};
Tensor& tanh_out_hpu_wrap(Tensor& out, Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return tanh_out_hpu_lazy(out, self);
  } else {
    return tanh_out_hpu(out, self);
  }
};
Tensor tanh_backward_hpu_wrap(const Tensor& grad_in, const Tensor& input) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return tanh_backward_hpu_lazy(grad_in, input);
  } else {
    return tanh_backward_hpu(grad_in, input);
  }
};
Tensor gelu_hpu_wrap(const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return gelu_hpu_lazy(self);
  } else {
    return gelu_hpu(self);
  }
};
Tensor gelu_backward_hpu_wrap(const Tensor& grad, const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return gelu_backward_hpu_lazy(grad, self);
  } else {
    return gelu_backward_hpu(grad, self);
  }
};
Tensor& erf_hpu_wrap_(Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return erf_hpu_lazy_(self);
  } else {
    return erf_hpu_(self);
  }
};
Tensor erf_hpu_wrap(const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return erf_hpu_lazy(self);
  } else {
    return erf_hpu(self);
  }
};
Tensor& exp_hpu_wrap_(Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return exp_hpu_lazy_(self);
  } else {
    return exp_hpu_(self);
  }
};
Tensor exp_hpu_wrap(const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return exp_hpu_lazy(self);
  } else {
    return exp_hpu(self);
  }
};
Tensor& neg_out_hpu_wrap(Tensor& result, const Tensor& input) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return neg_out_hpu_lazy(result, input);
  } else {
    return neg_out_hpu(result, input);
  }
};
Tensor& reciprocal_hpu_wrap_(Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return reciprocal_hpu_lazy_(self);
  } else {
    return reciprocal_hpu_(self);
  }
};
Tensor reciprocal_hpu_wrap(const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return reciprocal_hpu_lazy(self);
  } else {
    return reciprocal_hpu(self);
  }
};
Tensor& reciprocal_out_hpu_wrap(Tensor& result, const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return reciprocal_out_hpu_lazy(result, self);
  } else {
    return reciprocal_out_hpu(result, self);
  }
};
Tensor clamp_min_hpu_wrap(const Tensor& self, Scalar min) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return clamp_min_hpu_lazy(self, min);
  } else {
    return clamp_min_hpu(self, min);
  }
};
Tensor& clamp_hpu_wrap_(Tensor& self, c10::optional<Scalar> min, c10::optional<Scalar> max) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return clamp_hpu_lazy_(self, min, max);
  } else {
    return clamp_hpu_(self, min, max);
  }
};
Tensor clamp_hpu_wrap(const Tensor& self, c10::optional<Scalar> min, c10::optional<Scalar> max) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return clamp_hpu_lazy(self, min, max);
  } else {
    return clamp_hpu(self, min, max);
  }
};
Tensor abs_hpu_wrap(const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return abs_hpu_lazy(self);
  } else {
    return abs_hpu(self);
  }
};
Tensor neg_hpu_wrap(const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return neg_hpu_lazy(self);
  } else {
    return neg_hpu(self);
  }
};
namespace at {
namespace native {
Scalar _local_scalar_dense_hpu_wrap(const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    return _local_scalar_dense_hpu_lazy(self);
  } else {
    return _local_scalar_dense_hpu(self);
  }
}
} // namespace native
} // namespace at
