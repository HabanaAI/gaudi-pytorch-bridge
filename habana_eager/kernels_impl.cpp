
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
#include "common/dump_args.h"
#include "habana_eager/helpers.h"
#include "habana_eager/ops/as_strided.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_eager/ops/empty.h"
#include "habana_eager/ops/index_put.h"
#include "habana_eager/ops/nonzero.h"
#include "habana_eager/ops/set.h"
#include "habana_eager/ops/view.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_kernels_ver/wrap_kernels_declarations.h"
#include "hpu_ops/cpu_fallback.h"
#include "hpu_ops/op_logger.h"

#include "habana_helpers/logging.h"

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
  auto out = habana::eager::alias_with_sizes_and_strides(self, size, stride);
  habana::eager::view_propagate_permutation(self, out);
  return out;
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
      meta.shape = grad.sizes().vec();
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
      {habana::eager::eagerOpKind::InplaceOut,
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
      !(self.scalar_type() == c10::ScalarType::Half &&
        habana::HPURegistrar::get_device().type() !=
            synDeviceType::synDeviceGaudi)) {
    return dispatch_fallback<ATEN_OP(_index_put_impl_)>::call(
        OpSupportLevel::Value::unsupported_dtype,
        PARAMS2(self, indices, values, accumulate, unsafe));
  } else if (accumulate == false) {
    auto indices_size = indices.size();
    for (size_t i = 0; i < indices_size; ++i) {
      auto const& opt_tensor = indices[i];
      if (opt_tensor.has_value() && opt_tensor.value().defined() &&
          opt_tensor.value().numel() > self.sizes()[i]) {
        return dispatch_fallback<ATEN_OP(_index_put_impl_)>::call(
            OpSupportLevel::Value::unsupported_args,
            PARAMS2(self, indices, values, accumulate, unsafe));
      }
    }
  }

  return habana::eager::_index_put_impl_eager(
      self, indices, values, accumulate, unsafe);
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
