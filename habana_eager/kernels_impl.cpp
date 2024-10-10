/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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
#include "backend/backend_meta.h"
#include "backend/helpers/eager_pipeline.h"
#include "common/dump_args.h"
#include "generated/eager/wrap_kernels_declarations.h"
#include "habana_eager/eager_context.h"
#include "habana_eager/eager_pipeline_utils.h"
#include "habana_eager/eager_tensor.h"
#include "habana_eager/helpers.h"
#include "habana_eager/ops/as_strided.h"
#include "habana_eager/ops/bincount.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_eager/ops/empty.h"
#include "habana_eager/ops/index_put.h"
#include "habana_eager/ops/masked_select.h"
#include "habana_eager/ops/nonzero.h"
#include "habana_eager/ops/set.h"
#include "habana_eager/ops/unique.h"
#include "habana_eager/ops/unique2.h"
#include "habana_eager/ops/view.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "hpu_ops/cpu_fallback.h"
#include "hpu_ops/op_logger.h"

using namespace at;
using namespace habana;

Tensor hpu_wrap::empty(
    SymIntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<MemoryFormat> optional_memory_format) {
  PT_EAGER_TRACE;
  return habana::eager::empty(
      size, dtype, layout, device, pin_memory, optional_memory_format);
}

Tensor hpu_wrap::empty_strided(
    SymIntArrayRef size,
    SymIntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  PT_EAGER_TRACE;
  return habana::eager::empty_strided(
      size, stride, dtype, layout, device, pin_memory);
}

Tensor hpu_wrap::_reshape_alias(
    const Tensor& self,
    SymIntArrayRef size,
    SymIntArrayRef stride) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "_reshape_alias :",
      " self=",
      to_string(self),
      " size=",
      to_string(size),
      " stride",
      to_string(stride));
  auto result = habana::eager::alias_with_sizes_and_strides(self, size, stride);
  auto pipeline_or_direct_reshape_alias = [](const at::Tensor& self,
                                             const at::Tensor& result) {
    habana::eager::view_propagate_permutation(self, result);
    habana_helpers::set_output_hw_scaling_meta(self, result);
  };
  auto src_backend = habana::eager::HbEagerTensorPool::get_backend_tensor(self);
  auto dst_backend =
      habana::eager::HbEagerTensorPool::get_backend_tensor(result);
  auto dst_hb_tmeta{habana::get_tensor_extra_meta(dst_backend)};
  dst_hb_tmeta->set_tensor_pipelined();
  habana::eager::pipeline_or_direct_generic(
      pipeline_or_direct_reshape_alias,
      std::move(src_backend),
      std::move(dst_backend));
  return result;
}

Tensor hpu_wrap::_unsafe_view(const Tensor& self, SymIntArrayRef size) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "_unsafe_view :", " self=", to_string(self), " size=", to_string(size));
  auto result = habana::eager::view_hpu(self, size);
  return result;
}

at::Tensor fused_norm_hpu_wrap(
    std::vector<at::Tensor>& grad,
    const at::Tensor& max_norm,
    float norm_type) {
  auto FusedNormMeta = [](const at::Stack& stack) {
    OutputMetaDataVector meta_vec;

    auto grads = stack[0].toTensorList();
    meta_vec.reserve(grads.size() + 1);

    {
      // First and second element in the meta_vec vector should be the same
      OutputMetaData meta;
      const at::Tensor& grad = grads[0];
      meta.dtype = grad.scalar_type();
      meta.shape = {1};
      meta_vec.push_back(meta);
    }

    for (const at::Tensor& grad : grads) {
      OutputMetaData meta;
      meta.dtype = grad.scalar_type();
      meta.shape = grad.sizes().vec();
      meta_vec.push_back(meta);
    }

    return meta_vec;
  };

  habana::eager::EagerOp<std::vector<at::Tensor>> hpu_op{
      "hpu::fused_norm_lazy", {grad, max_norm, norm_type}};
  hpu_op.SetOutputMetaFn(FusedNormMeta);
  auto res = hpu_op.call();
  for (size_t i = 0; i < grad.size(); ++i) {
    // grad[i] = res[i + 1]
    // calls operator= with rvalue reference qualifier i.e. deep copy
    // grad_ref = res[i + 1]
    // calls operator= with lvalue reference qualifier i.e. shallow copy
    // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/templates/TensorBody.h
    at::Tensor& grad_ref = grad[i];
    grad_ref = res[i + 1];
  }
  return res[0];
}

at::Tensor hpu_wrap::repeat_interleave(
    const at::Tensor& self,
#if IS_PYTORCH_AT_LEAST(2, 4)
    ::std::optional<SymInt> output_size) {
#elif IS_PYTORCH_AT_LEAST(2, 2)
    c10::optional<SymInt> output_size) {
#else
    c10::optional<int64_t> output_size) {
#endif
  PT_EAGER_TRACE;
  PT_OP_INFO("repeat_interleave:", DUMP_2ARGS(self, output_size));

  // If output_size is not provided in optional, it must be calculated on
  // frontend to get the actual output size.
  if (!output_size) {
    auto out = self.sum();
    output_size = out.item().toInt();
  }
  auto RepeatInterleaveMeta = [](const at::Stack& stack) {
    auto self = stack.at(0).toTensor();
    auto output_size_opt = stack.at(1).toOptional<int64_t>();
    TORCH_CHECK(
        output_size_opt.has_value(),
        "It is expected that output_size is provided after frontend execution.");

    OutputMetaData meta;
    meta.dtype = self.scalar_type();
    meta.shape = std::vector<int64_t>{output_size_opt.value()};

    return OutputMetaDataVector{meta};
  };

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "aten::repeat_interleave", {self, output_size}};
  hpu_op.SetOutputMetaFn(RepeatInterleaveMeta);
  return hpu_op.call();
}

void optimizer_sgd_momentum_hpu_wrap(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& momentum,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    at::Tensor& mom,
    const float damp,
    const bool nesterov) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      " optimizer_sgd_momentum:",
      DUMP_9ARGS(
          gradients,
          weights,
          momentum,
          epoch_num,
          lr,
          wd,
          mom,
          damp,
          nesterov));
  TORCH_CHECK(
      (weights.size() > 0),
      "optimizer_sgd_momentum : can not process empty weight vector");
  eager::EagerOp<void> hpu_op{
      "hpu::optimizer_sgd_momentum",
      {gradients, weights, momentum, epoch_num, lr, mom, wd, damp, nesterov}};
  hpu_op.set_eager_op_info(
      {habana::eager::eagerOpKind::Inplace,
       "hpu::optimizer_sgd_momentum",
       {1, 2}});
  hpu_op.call({weights, momentum});
}

at::Tensor& hpu_wrap::_index_put_impl_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& values,
    bool accumulate,
    bool unsafe) {
  if ((self.scalar_type() != c10::ScalarType::Float) &&
      (self.scalar_type() != c10::ScalarType::Int) &&
      (self.scalar_type() != c10::ScalarType::Long) &&
      (self.scalar_type() != c10::ScalarType::Char) &&
      (self.scalar_type() != c10::ScalarType::Bool) &&
      (self.scalar_type() != c10::ScalarType::BFloat16) &&
      (self.scalar_type() != c10::ScalarType::Short) &&
      (self.scalar_type() != c10::ScalarType::Byte) &&
      (self.scalar_type() != c10::ScalarType::Double) &&
      !(self.scalar_type() == c10::ScalarType::Half &&
        habana::HPURegistrar::get_device().type() !=
            synDeviceType::synDeviceGaudi)) {
    return dispatch_fallback<ATEN_OP(_index_put_impl_)>::call(
        OpSupportLevel::Value::unsupported_dtype,
        PARAMS2(self, indices, values, accumulate, unsafe));
  }

  return habana::eager::_index_put_impl_eager(
      self, indices, values, accumulate, unsafe);
}

::std::tuple<at::Tensor, at::Tensor> hpu_wrap::_unique(
    const at::Tensor& self,
    bool sorted,
    bool return_inverse) {
  return habana::eager::_unique_eager(self, sorted, return_inverse);
}

at::Tensor hpu_wrap::bincount(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& weights,
    int64_t minlength) {
  PT_EAGER_TRACE;
  PT_OP_INFO("bincount :", DUMP_3ARGS(self, weights, minlength));
  static const std::array<c10::ScalarType, 5> valid_self_types = {
      c10::ScalarType::Int,
      c10::ScalarType::Long,
      c10::ScalarType::Short,
      c10::ScalarType::Char,
      c10::ScalarType::Byte,
  };
  static const std::array<c10::ScalarType, 9> valid_weights_types = {
      c10::ScalarType::Float,
      c10::ScalarType::BFloat16,
      c10::ScalarType::Int,
      c10::ScalarType::Long,
      c10::ScalarType::Short,
      c10::ScalarType::Char,
      c10::ScalarType::Byte,
      c10::ScalarType::Double,
      c10::ScalarType::Half,
  };
  bool is_self_valid = std::find(
                           valid_self_types.begin(),
                           valid_self_types.end(),
                           self.scalar_type()) != valid_self_types.end();
  bool is_weights_valid = !weights.has_value() ||
      std::find(
          valid_weights_types.begin(),
          valid_weights_types.end(),
          weights.value().scalar_type()) != valid_weights_types.end();
  if (is_self_valid && is_weights_valid) {
    return habana::eager::bincount_eager(self, weights, minlength);
  }
  return dispatch_fallback<ATEN_OP(bincount)>::call(
      OpSupportLevel::Value::unsupported_dtype,
      PARAMS2(self, weights, minlength));
}

at::Tensor hpu_wrap::nonzero(const at::Tensor& self) {
  if ((self.scalar_type() != c10::ScalarType::Float) &&
      (self.scalar_type() != c10::ScalarType::Int) &&
      (self.scalar_type() != c10::ScalarType::Long) &&
      (self.scalar_type() != c10::ScalarType::Char) &&
      (self.scalar_type() != c10::ScalarType::BFloat16) &&
      (self.scalar_type() != c10::ScalarType::Bool) &&
      !(self.scalar_type() == c10::ScalarType::Half &&
        habana::HPURegistrar::get_device().type() !=
            synDeviceType::synDeviceGaudi &&
        self.dim() >
            4)) { // self.dim()<=4 goes through cguid that doesn't support fp16
    return dispatch_fallback<ATEN_OP(nonzero)>::call(
        OpSupportLevel::Value::unsupported_dtype, PARAMS2(self));
  }
  return habana::eager::nonzero_eager(self);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> hpu_wrap::_unique2(
    const at::Tensor& self,
    bool sorted,
    bool return_inverse,
    bool return_counts) {
  return habana::eager::_unique2_eager(
      self, sorted, return_inverse, return_counts);
}

at::Tensor& hpu_wrap::nonzero_out(const at::Tensor& self, at::Tensor& out) {
  if ((self.scalar_type() != c10::ScalarType::Float) &&
      (self.scalar_type() != c10::ScalarType::Int) &&
      (self.scalar_type() != c10::ScalarType::BFloat16) &&
      (self.scalar_type() != c10::ScalarType::Bool) &&
      !(self.scalar_type() == c10::ScalarType::Half &&
        habana::HPURegistrar::get_device().type() !=
            synDeviceType::synDeviceGaudi)) {
    return dispatch_fallback<ATEN_OP(nonzero_out)>::call(
        OpSupportLevel::Value::unsupported_dtype, PARAMS2(self, out));
  }
  return habana::eager::nonzero_out_eager(self, out);
}

at::Tensor hpu_wrap::masked_select(
    const at::Tensor& self,
    const at::Tensor& mask) {
  PT_EAGER_TRACE;
  if ((self.scalar_type() != c10::ScalarType::Float) &&
      (self.scalar_type() != c10::ScalarType::Double) &&
      (self.scalar_type() != c10::ScalarType::Int) &&
      (self.scalar_type() != c10::ScalarType::Long) &&
      (self.scalar_type() != c10::ScalarType::Char) &&
      (self.scalar_type() != c10::ScalarType::Bool) &&
      (self.scalar_type() != c10::ScalarType::BFloat16) &&
      !(self.scalar_type() == c10::ScalarType::Half &&
        habana::HPURegistrar::get_device().type() !=
            synDeviceType::synDeviceGaudi)) {
    return dispatch_fallback<ATEN_OP(masked_select)>::call(
        OpSupportLevel::Value::unsupported_dtype, PARAMS2(self, mask));
  }
  return habana::eager::masked_select_eager(self, mask);
}

at::Tensor& hpu_wrap::masked_select_out(
    const at::Tensor& self,
    const at::Tensor& mask,
    at::Tensor& out) {
  PT_EAGER_TRACE;
  if ((self.scalar_type() != c10::ScalarType::Float) &&
      (self.scalar_type() != c10::ScalarType::Double) &&
      (self.scalar_type() != c10::ScalarType::Int) &&
      (self.scalar_type() != c10::ScalarType::Long) &&
      (self.scalar_type() != c10::ScalarType::Char) &&
      (self.scalar_type() != c10::ScalarType::Bool) &&
      (self.scalar_type() != c10::ScalarType::BFloat16) &&
      !(self.scalar_type() == c10::ScalarType::Half &&
        habana::HPURegistrar::get_device().type() !=
            synDeviceType::synDeviceGaudi)) {
    return dispatch_fallback<ATEN_OP(masked_select_out)>::call(
        OpSupportLevel::Value::unsupported_dtype, PARAMS2(self, mask, out));
  }
  return habana::eager::masked_select_out_eager(self, mask, out);
}