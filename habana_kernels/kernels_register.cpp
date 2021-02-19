/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <ATen/core/op_registration/hacky_wrapper_for_legacy_signatures.h>
#include <torch/library.h>

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
    const Tensor& self,
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
    const Tensor& self,
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

Tensor& mul_out_hpu_wrap(Tensor& out, const Tensor& self, const Tensor& other) {
  if (!habana_lazy::isDeviceInLoweringMode(self.device().index()) &&
      std::getenv("PT_HPU_LAZY_MODE")) {
    return mul_out_hpu_lazy(out, self, other);
  } else {
    return mul_out_hpu(out, self, other);
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

Tensor gt_tensor_hpu_wrap(const Tensor& self, const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = gt_tensor_hpu_lazy(self, other);

    return t;
  } else {
    return gt_tensor_hpu(self, other);
  }
};

Tensor gt_scalar_hpu_wrap(const Tensor& self, Scalar other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = gt_scalar_hpu_lazy(self, other);

    return t;
  } else {
    return gt_scalar_hpu(self, other);
  }
};
Tensor& eq_tensor_out_hpu_wrap(
    Tensor& output,
    const Tensor& self,
    const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    eq_tensor_out_hpu_lazy(output, self, other);
  } else {
    eq_tensor_out_hpu(output, self, other);
  }
};
Tensor eq_tensor_hpu_wrap(const Tensor& self, const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = eq_tensor_hpu_lazy(self, other);

    return t;
  } else {
    return eq_tensor_hpu(self, other);
  }
};
Tensor eq_tensor_scalar_hpu_wrap(const Tensor& self, Scalar other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = eq_tensor_scalar_hpu_lazy(self, other);

    return t;
  } else {
    return eq_tensor_scalar_hpu(self, other);
  }
};
Tensor lt_scalar_hpu_wrap(const Tensor& self, Scalar other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = lt_scalar_hpu_lazy(self, other);

    return t;
  } else {
    return lt_scalar_hpu(self, other);
  }
};

Tensor lt_tensor_hpu_wrap(const Tensor& self, const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = lt_tensor_hpu_lazy(self, other);

    return t;
  } else {
    return lt_tensor_hpu(self, other);
  }
};
Tensor ge_scalar_hpu_wrap(const Tensor& self, Scalar other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = ge_scalar_hpu_lazy(self, other);
    return t;
  } else {
    return ge_scalar_hpu(self, other);
  }
};
Tensor ge_tensor_hpu_wrap(const Tensor& self, const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = ge_tensor_hpu_lazy(self, other);
    return t;
  } else {
    return ge_tensor_hpu(self, other);
  }
};
Tensor ne_scalar_hpu_wrap(const Tensor& self, Scalar other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = ne_scalar_hpu_lazy(self, other);

    return t;
  } else {
    return ne_scalar_hpu(self, other);
  }
};
Tensor ne_tensor_hpu_wrap(const Tensor& self, const Tensor& other) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
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
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& save_mean,
    const Tensor& save_invstd,
    bool train,
    double eps,
    std::array<bool, 3> output_mask) {
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

// PTv_1.7 uses new kernel
Tensor norm_ScalarOpt_dim_hpu(
    const Tensor& self,
    c10::optional<c10::Scalar> p,
    IntArrayRef dim,
    bool keepdim) {
  return norm_scalar_hpu_wrap(self, p.value_or(2));
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
Tensor& uniform_hpu_wrap(
    Tensor& self,
    double from,
    double to,
    c10::optional<Generator> gen) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    uniform_hpu_lazy(self, from, to, gen);
  } else {
    uniform_hpu(self, from, to, gen);
  }
};
Tensor& normal_hpu_wrap(
    Tensor& self,
    double mean,
    double std,
    c10::optional<Generator> gen) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    normal_hpu_lazy(self, mean, std, gen);
  } else {
    normal_hpu(self, mean, std, gen);
  }
};

Tensor bernoulli_hpu_wrap(const Tensor& self, c10::optional<Generator> gen) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = bernoulli_hpu_lazy(self, gen);
    return t;
  } else {
    return bernoulli_hpu(self, gen);
  }
};

Tensor& bernoulli_scalar_hpu_wrap(
    Tensor& self,
    double p,
    c10::optional<Generator> gen) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto& t = bernoulli_scalar_hpu_lazy(self, p, gen);
    return t;
  } else {
    return bernoulli_scalar_hpu(self, p, gen);
  }
};
std::tuple<Tensor, Tensor> fused_dropout_hpu_wrap(
    const Tensor& self,
    double p,
    c10::optional<Generator> gen) {
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

// wrapper for empty_strided in PT-1.7
Tensor empty_strided_hpu_new(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  at::TensorOptions options = at::TensorOptions()
                                  .dtype(dtype)
                                  .layout(layout)
                                  .pinned_memory(pin_memory)
                                  .device(device);
  return empty_strided_hpu_wrap(size, stride, options);
}
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
Tensor sqrt_hpu_wrap(const Tensor& input) {
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

Tensor& tanh_out_hpu_wrap(Tensor& out, const Tensor& self) {
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
Tensor round_hpu_wrap(const Tensor& self) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    auto t = round_hpu_lazy(self);
    return t;
  } else {
    return round_hpu(self);
  }
};
Tensor& round_hpu_wrap_(Tensor& self) {
  if (!habana_lazy::isDeviceInLoweringMode(self.device().index()) &&
      std::getenv("PT_HPU_LAZY_MODE")) {
    return round_hpu_lazy_(self);
  } else {
    return round_hpu_(self);
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
// compatible with PT1.7
Tensor ones_like_hpu_new(
    const Tensor& self,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> memory_format) {
  at::TensorOptions options = at::TensorOptions()
                                  .dtype(dtype)
                                  .layout(layout)
                                  .pinned_memory(pin_memory)
                                  .device(device);
  return ones_like_hpu_wrap(self, options, memory_format);
}

Tensor& optimizer_sgd_hpu_wrap(
    const TensorList& gradients,
    TensorList& weights,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov) {
  if (!habana_lazy::isDeviceInLoweringMode(weights[0].device().index()) &&
      std::getenv("PT_HPU_LAZY_MODE")) {
    optimizer_sgd_hpu_lazy(gradients, weights, lr, wd, mom, damp, nesterov);
  } else {
    optimizer_sgd_hpu(gradients, weights, lr, wd, mom, damp, nesterov);
  }

  return lr;
}

Tensor& optimizer_sgd_momentum_hpu_wrap(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& momentum,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov) {
  if (!habana_lazy::isDeviceInLoweringMode(weights[0].device().index()) &&
      std::getenv("PT_HPU_LAZY_MODE")) {
    optimizer_sgd_momentum_hpu_lazy(
        gradients, weights, momentum, epoch_num, lr, wd, mom, damp, nesterov);
  } else {
    optimizer_sgd_momentum_hpu(
        gradients, weights, momentum, epoch_num, lr, wd, mom, damp, nesterov);
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
    HABANA_ASSERT(0 && "matmul_backward not implemented for lazy mode");
    return matmul_backward_hpu(grad_output, self, other);
  } else {
    return matmul_backward_hpu(grad_output, self, other);
  }
}

Tensor matmul_hpu_wrap(const at::Tensor& tensor1, const at::Tensor& tensor2) {
  if (std::getenv("PT_HPU_LAZY_MODE")) {
    HABANA_ASSERT(0 && "matmul not implemented for lazy mode");
    return matmul_hpu(tensor1, tensor2);
  } else {
    return matmul_hpu(tensor1, tensor2);
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

Tensor cast_hpu_for_registration_only(const Tensor& self, Scalar type) {
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

TORCH_LIBRARY_IMPL(aten, HABANATensorId, m) {
  m.impl_UNBOXED("add.Tensor", add_tensor_hpu_wrap);
  m.impl_UNBOXED("add.Scalar", add_scalar_hpu_wrap);
  m.impl_UNBOXED("add_.Tensor", add_tensor_hpu_wrap_);
  m.impl_UNBOXED("add_.Scalar", add_scalar_hpu_wrap_);
  // m.impl_UNBOXED("empty_strided", at::native::empty_strided_hpu_new);
  // this is workaround, we should remove this hacky wrapper
  m.impl(
      "empty_strided",
      c10::impl::hacky_wrapper_for_legacy_signatures<Tensor(
          IntArrayRef,
          IntArrayRef,
          c10::optional<ScalarType>,
          c10::optional<Layout>,
          c10::optional<Device>,
          c10::optional<bool>)>(TORCH_FN(at::native::empty_strided_hpu_wrap)));
  m.impl_UNBOXED("empty.memory_format", at::native::empty_hpu_wrap);
  m.impl_UNBOXED("copy_", copy_hpu_wrap_);
  m.impl_UNBOXED("as_strided", as_strided_hpu_wrap);
  m.impl_UNBOXED("set_.source_Storage_storage_offset", set_hpu_wrap_);
  m.impl_UNBOXED("view", view_hpu_wrap);
  m.impl_UNBOXED("addcmul", addcmul_hpu_wrap);
  m.impl_UNBOXED("addcmul_", addcmul_hpu_wrap_);
  m.impl_UNBOXED("addcdiv", addcdiv_hpu_wrap);
  m.impl_UNBOXED("addcdiv_", addcdiv_hpu_wrap_);
  m.impl_UNBOXED("sub.Tensor", sub_tensor_hpu_wrap);
  m.impl_UNBOXED("sub.Scalar", sub_scalar_hpu_wrap);
  m.impl_UNBOXED("sub_.Tensor", sub_tensor_hpu_wrap_);
  m.impl_UNBOXED("sub_.Scalar", sub_scalar_hpu_wrap_);
  m.impl_UNBOXED("mul.Tensor", mul_tensor_hpu_wrap);
  m.impl_UNBOXED("mul_.Tensor", mul_tensor_hpu_wrap_);
  m.impl_UNBOXED("mul.Scalar", mul_scalar_hpu_wrap);
  m.impl_UNBOXED("mul_.Scalar", mul_scalar_hpu_wrap_);
  m.impl_UNBOXED("mul.out", mul_out_hpu_wrap);
  m.impl_UNBOXED("div.Tensor", div_tensor_hpu_wrap);
  m.impl_UNBOXED("div.out", div_tensor_hpu_wrap_out);
  m.impl_UNBOXED("div_.Tensor", div_tensor_hpu_wrap_);
  m.impl_UNBOXED("div.Scalar", div_scalar_hpu_wrap);
  m.impl_UNBOXED("div_.Scalar", div_scalar_hpu_wrap_);
  m.impl_UNBOXED("pow.Tensor_Tensor", pow_tensor_tensor_hpu_wrap);
  m.impl_UNBOXED("pow_.Tensor", pow_tensor_tensor_hpu_wrap_);
  m.impl_UNBOXED("pow.Tensor_Scalar", pow_tensor_scalar_hpu_wrap);
  m.impl_UNBOXED("pow.Scalar", pow_scalar_tensor_hpu_wrap);
  m.impl_UNBOXED("pow_.Scalar", pow_tensor_scalar_hpu_wrap_);
  m.impl_UNBOXED("rsub.Scalar", rsub_scalar_hpu_wrap);
  m.impl_UNBOXED("eq.Tensor", eq_tensor_hpu_wrap);
  m.impl_UNBOXED("eq.Tensor_out", eq_tensor_out_hpu_wrap);
  m.impl_UNBOXED("eq.Scalar", eq_tensor_scalar_hpu_wrap);
  m.impl_UNBOXED("lt.Scalar", lt_scalar_hpu_wrap);
  m.impl_UNBOXED("lt.Tensor", lt_tensor_hpu_wrap);
  m.impl_UNBOXED("gt.Scalar", gt_scalar_hpu_wrap);
  m.impl_UNBOXED("gt.Tensor", gt_tensor_hpu_wrap);
  m.impl_UNBOXED("ne.Tensor", ne_tensor_hpu_wrap);
  m.impl_UNBOXED("ne.Scalar", ne_scalar_hpu_wrap);
  m.impl_UNBOXED("ge.Tensor", ge_tensor_hpu_wrap);
  m.impl_UNBOXED("ge.Scalar", ge_scalar_hpu_wrap);
  m.impl_UNBOXED("fill_.Scalar", fill_hpu_wrap_);
  m.impl_UNBOXED("masked_fill_.Tensor", masked_fill_hpu_wrap_);
  m.impl_UNBOXED("masked_fill_.Scalar", masked_fill_scalar_hpu_wrap_);
  m.impl_UNBOXED("index_select", index_select_hpu_wrap);
  m.impl_UNBOXED(
      "convolution_backward_overrideable", convolution_backward_hpu_wrap);
  m.impl(
      "convolution_overrideable",
      c10::impl::hacky_wrapper_for_legacy_signatures<at::Tensor(
          const at::Tensor&,
          const at::Tensor&,
          const c10::optional<at::Tensor>&,
          at::IntArrayRef,
          at::IntArrayRef,
          at::IntArrayRef,
          bool,
          at::IntArrayRef,
          int64_t)>(TORCH_FN(convolution_hpu_wrap)));
  m.impl_UNBOXED("constant_pad_nd", constant_pad_hpu_wrap);
  m.impl_UNBOXED("embedding", embedding_hpu_wrap);
  m.impl_UNBOXED("embedding_dense_backward", embedding_dense_backward_hpu_wrap);
  m.impl_UNBOXED("embedding_bag_sum_fwd", embedding_bag_sum_fwd_hpu_wrap);
  m.impl_UNBOXED(
      "embedding_bag_sum_bwd.out", embedding_bag_sum_bwd_out_hpu_wrap);
  m.impl_UNBOXED("index_put_", index_put_hpu_wrap_);
  m.impl_UNBOXED("index_put", index_put_hpu_wrap);
  m.impl_UNBOXED("index_add_", index_add_hpu_wrap_);
  m.impl_UNBOXED("scatter_.src", scatter_inplace_src_hpu_wrap);
  m.impl_UNBOXED("scatter.src", scatter_src_hpu_wrap);
  m.impl_UNBOXED("gather", gather_src_hpu_wrap);
  m.impl_UNBOXED("scatter_add", scatter_add_src_hpu_wrap);
  m.impl_UNBOXED("scatter_add_", scatter_add_inplace_src_hpu_wrap);
  m.impl_UNBOXED("slice.Tensor", slice_hpu_wrap);
  m.impl_UNBOXED("select.int", select_hpu_wrap);
  m.impl_UNBOXED("arange.start_out", arange_hpu_wrap);
  m.impl_UNBOXED("mm", mm_hpu_wrap);
  m.impl_UNBOXED("addmm", addmm_hpu_wrap);
  m.impl_UNBOXED("bmm.out", batch_gemm_out_hpu_wrap);
  m.impl_UNBOXED("bmm", batch_gemm_hpu_wrap);
  m.impl_UNBOXED("dot", dot_hpu_wrap);
  m.impl_UNBOXED("mv", mv_hpu_wrap);
  m.impl(
      "nll_loss_forward",
      c10::impl::hacky_wrapper_for_legacy_signatures<std::tuple<Tensor, Tensor>(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          int64_t,
          int64_t)>(TORCH_FN(nll_loss_forward_hpu_wrap)));
  m.impl(
      "nll_loss_backward",
      c10::impl::hacky_wrapper_for_legacy_signatures<Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          int64_t,
          int64_t,
          const Tensor&)>(TORCH_FN(nll_loss_backward_hpu_wrap)));
  m.impl_UNBOXED("mse_loss", mse_loss_forward_hpu_wrap);
  m.impl_UNBOXED("mse_loss_backward", mse_loss_backward_hpu_wrap);
  m.impl(
      "binary_cross_entropy",
      c10::impl::hacky_wrapper_for_legacy_signatures<Tensor(
          const Tensor&, const Tensor&, const c10::optional<Tensor>&, int64_t)>(
          TORCH_FN(binary_cross_entropy_hpu_wrap)));
  m.impl(
      "binary_cross_entropy_backward",
      c10::impl::hacky_wrapper_for_legacy_signatures<Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          int64_t)>(TORCH_FN(binary_cross_entropy_backward_hpu_wrap)));

  m.impl(
      "native_batch_norm",
      c10::impl::hacky_wrapper_for_legacy_signatures<
          std::tuple<Tensor, Tensor, Tensor>(
              const Tensor&,
              const c10::optional<Tensor>&,
              const c10::optional<Tensor>&,
              const c10::optional<Tensor>&,
              const c10::optional<Tensor>&,
              bool,
              double,
              double)>(TORCH_FN(batch_norm_hpu_wrap)));

  m.impl(
      "native_batch_norm_backward",
      c10::impl::hacky_wrapper_for_legacy_signatures<
          std::tuple<Tensor, Tensor, Tensor>(
              const Tensor&,
              const Tensor&,
              const c10::optional<Tensor>&,
              const c10::optional<Tensor>&,
              const c10::optional<Tensor>&,
              const c10::optional<Tensor>&,
              const c10::optional<Tensor>&,
              bool,
              double,
              std::array<bool, 3>)>(TORCH_FN(batch_norm_bwd_hpu_wrap)));

  m.impl(
      "native_layer_norm",
      c10::impl::hacky_wrapper_for_legacy_signatures<
          std::tuple<Tensor, Tensor, Tensor>(
              const Tensor&,
              const c10::optional<Tensor>&,
              const c10::optional<Tensor>&,
              int64_t,
              int64_t,
              double)>(TORCH_FN(layer_norm_hpu_wrap)));

  m.impl(
      "native_layer_norm_backward",
      c10::impl::hacky_wrapper_for_legacy_signatures<
          std::tuple<Tensor, Tensor, Tensor>(
              const Tensor&,
              const Tensor&,
              const Tensor&,
              const Tensor&,
              const c10::optional<Tensor>&,
              int64_t,
              int64_t,
              std::array<bool, 3>)>(TORCH_FN(layer_norm_backward_hpu_wrap)));
  m.impl_UNBOXED("norm.ScalarOpt_dim", norm_ScalarOpt_dim_hpu);
  m.impl_UNBOXED("norm.Scalar", norm_scalar_hpu_wrap);
  m.impl_UNBOXED("max_pool2d_with_indices", max_pool2d_with_indices_hpu_wrap);
  m.impl_UNBOXED(
      "max_pool2d_with_indices_backward",
      max_pool2d_with_indices_backward_hpu_wrap);
  // m.impl_UNBOXED("max_pool2d_with_indices_backward.grad_input",
  // max_pool2d_with_indices_backward_out_hpu_wrap);
  m.impl_UNBOXED("avg_pool2d", avg_pool2d_hpu_wrap);
  m.impl_UNBOXED("avg_pool2d_backward", avg_pool2d_backward_hpu_wrap);
  m.impl_UNBOXED("uniform_", uniform_hpu_wrap);
  m.impl_UNBOXED("normal_", normal_hpu_wrap);
  m.impl_UNBOXED("bernoulli", bernoulli_hpu_wrap);
  m.impl_UNBOXED("bernoulli_.float", bernoulli_scalar_hpu_wrap);
  m.impl_UNBOXED("_fused_dropout", fused_dropout_hpu_wrap);
  m.impl_UNBOXED("sum.dim_IntList", sum_dim_IntList_hpu_wrap);
  m.impl_UNBOXED("sum.IntList_out", sum_IntList_out_hpu_wrap);
  m.impl_UNBOXED("mean.dim", mean_dim_hpu_wrap);
  m.impl_UNBOXED("mean.out", mean_dim_out_hpu_wrap);
  m.impl_UNBOXED("sum", sum_hpu_wrap);
  m.impl_UNBOXED("mean", mean_hpu_wrap);
  m.impl_UNBOXED("any.dim", any_dim_hpu_wrap);
  m.impl_UNBOXED("any", any_hpu_wrap);
  m.impl_UNBOXED("any.out", any_dim_out_hpu_wrap);
  m.impl_UNBOXED(
      "_local_scalar_dense", at::native::_local_scalar_dense_hpu_wrap);
  m.impl_UNBOXED("_log_softmax", log_softmax_hpu_wrap);
  m.impl_UNBOXED("_log_softmax_backward_data", log_softmax_backward_hpu_wrap);
  m.impl_UNBOXED("_softmax", habana::softmax_hpu_wrap);
  m.impl_UNBOXED("_softmax_backward_data", softmax_backward_hpu_wrap);
  m.impl_UNBOXED("clone", clone_hpu_wrap);
  m.impl_UNBOXED("zero_", zero_hpu_wrap);
  m.impl_UNBOXED("permute", permute_hpu_wrap);
  m.impl_UNBOXED("expand", expand_hpu_wrap);
  m.impl_UNBOXED("cat", cat_hpu_wrap);
  m.impl_UNBOXED("cat.out", cat_hpu_wrap_out);
  m.impl_UNBOXED("split_with_sizes", split_with_sizes_hpu_wrap);
  m.impl_UNBOXED("transpose.int", transpose_hpu_wrap);
  m.impl_UNBOXED("transpose_", transpose_hpu_wrap_);
  m.impl_UNBOXED("t", t_hpu_wrap);
  m.impl_UNBOXED("threshold_backward", threshold_backward_hpu_wrap);
  m.impl_UNBOXED("topk", topk_hpu_wrap);
  m.impl_UNBOXED("topk.values", topk_out_hpu_wrap);
  m.impl_UNBOXED("sort", sort_hpu_wrap);
  m.impl_UNBOXED("relu_", relu_hpu_wrap_);
  m.impl_UNBOXED("relu", relu_hpu_wrap);
  m.impl_UNBOXED("sigmoid", sigmoid_hpu_wrap);
  m.impl_UNBOXED("sigmoid_backward", sigmoid_backward_hpu_wrap);
  m.impl_UNBOXED("sqrt", sqrt_hpu_wrap);
  m.impl_UNBOXED("tanh", tanh_hpu_wrap);
  m.impl_UNBOXED("tanh_backward", tanh_backward_hpu_wrap);
  m.impl_UNBOXED("tanh_", tanh_hpu_wrap_);
  m.impl_UNBOXED("tanh.out", tanh_out_hpu_wrap);
  m.impl_UNBOXED("gelu", gelu_hpu_wrap);
  m.impl_UNBOXED("gelu_backward", gelu_backward_hpu_wrap);
  m.impl_UNBOXED("erf_", erf_hpu_wrap_);
  m.impl_UNBOXED("erf", erf_hpu_wrap);
  m.impl_UNBOXED("exp_", exp_hpu_wrap_);
  m.impl_UNBOXED("exp", exp_hpu_wrap);
  m.impl_UNBOXED("neg.out", neg_out_hpu_wrap);
  m.impl_UNBOXED("reciprocal_", reciprocal_hpu_wrap_);
  m.impl_UNBOXED("reciprocal", reciprocal_hpu_wrap);
  m.impl_UNBOXED("reciprocal.out", reciprocal_out_hpu_wrap);
  m.impl_UNBOXED("clamp_min", clamp_min_hpu_wrap);
  m.impl_UNBOXED("clamp_", clamp_hpu_wrap_);
  m.impl_UNBOXED("clamp", clamp_hpu_wrap);
  m.impl_UNBOXED("abs", abs_hpu_wrap);
  m.impl_UNBOXED("round", round_hpu_wrap);
  m.impl_UNBOXED("round_", round_hpu_wrap_);
  m.impl_UNBOXED("neg", neg_hpu_wrap);
  m.impl_UNBOXED("ones_like", ones_like_hpu_new);
  m.impl_UNBOXED("matmul", matmul_hpu_wrap);
  m.impl_UNBOXED("matmul_backward", matmul_backward_hpu_wrap);
  m.impl_UNBOXED("_masked_scale", masked_scale_hpu_wrap);
  m.impl_UNBOXED("floor", floor_hpu_wrap);
  m.impl_UNBOXED("floor_", floor_hpu_wrap_);
}

TORCH_LIBRARY(hpu, m) {
  m.def("mm_t(Tensor mm, Tensor t , bool tr, bool no_tr) -> Tensor");
  m.def("habana_d2d_memcpy_other(Tensor s, Tensor d) -> Tensor");
  m.def(
      "sum_dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
  m.def("habana_d2d_memcpy(Tensor self) -> (Tensor)");
  m.def(
      "habanaOptimizerSparseSgd(Tensor gradients, Tensor weights_in, Tensor moments_in, Tensor indices, Tensor learning_rate, Tensor valid_count_tensor, float mom, bool nesterov) -> (Tensor, Tensor)");
  m.def(
      "habanaOptimizerSparseAdagrad(Tensor gradients, Tensor weights_in, Tensor moments_in, Tensor indices, Tensor learning_rate, Tensor valid_count_tensor) -> (Tensor, Tensor)");
  m.def("cast(Tensor self, Scalar type) -> Tensor(a)");
  m.def(
      "embedding_bag_sum(Tensor input, Tensor indices, Tensor offsets, Tensor valid_count, int kernel_mode) -> (Tensor)");
  m.def(
      "embedding_bag_sum_bwd_out(Tensor out, Tensor input, Tensor indices_bwd, Tensor offsets_bwd, Tensor valid_count_bwd, int kernel_mode) -> (Tensor)");
  m.def(
      "habanaOptimizerFusedAdagrad(Tensor[] gradients, Tensor[] weights_in, Tensor[] variances_in, Tensor epoch_num, Tensor learning_rate, float wd, float lrd, float eps) -> Tensor(a!)");
  m.def(
      "habanaOptimizerFusedSGD(Tensor[] gradients, Tensor[] weights_in, Tensor learning_rate, float wd, float mom, float damp, bool nesterov) -> Tensor(a!)");
  m.def(
      "habanaOptimizerFusedSGDMomentum(Tensor[] gradients, Tensor[] weights_in, Tensor[] momentum_in, Tensor epoch_num, Tensor learning_rate, float wd, float mom, float damp, bool nesterov) -> Tensor(a!)");
  m.def("permute_cl(Tensor(a) self, int[] dims) -> Tensor(a)");
  m.def("control_edge_other_(Tensor self, Tensor other) -> Tensor(a!)");
  m.def("control_edge_(Tensor self)-> Tensor(a!)");
  m.def(
      "hpu::native_batch_norm_rmv(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::native_batch_norm_inf(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor)");
}

TORCH_LIBRARY_IMPL(hpu, HABANATensorId, m) {
  /*m.impl(
      "habanaOptimizerSparseSgd",
      optimizer_sparse_sgd_with_valid_count_hpu_wrap);*/
  m.impl("habana_d2d_memcpy", habana_d2d_memcpy);
  /*m.impl(
      "habanaOptimizerSparseAdagrad",
      optimizer_sparse_adagrad_with_valid_count_hpu_wrap);*/
  m.impl("embedding_bag_sum", embedding_bag_sum_hpu_wrap);
  m.impl(
      "embedding_bag_sum_bwd_out",
      embedding_bag_sum_bwd_out_kernel_mode_hpu_wrap);
  m.impl("sum_dim_IntList", sum_dim_IntList_hpu_wrap);
}
