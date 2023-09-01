/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include <torch/library.h>
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/device_helpers.h"
#include "backend/synapse_helpers/env_flags.h"
#include "common/dump_args.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_kernels_ver/wrap_kernels_declarations.h"
#include "habana_lazy/lazy_executor.h"
#include "hpu_ops/cpu_fallback.h"
#include "hpu_ops/run_maybe_with_acc_thread.h"
#include "kernel_input_checks.h"
#include "pytorch_helpers/habana_helpers/kernels_accumulation.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

using namespace torch;
using namespace at;
using namespace habana;
using namespace habana_lazy;

bool hpu_wrap::is_pinned(
    const at::Tensor& self,
    c10::optional<at::Device> device) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "is_pinned :", " self=", to_string(self), " device=", to_string(device));
  return is_pinned_hpu(self, device);
}

Tensor hpu_wrap::pin_memory(
    const at::Tensor& self,
    c10::optional<at::Device> device) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "pin_memory :", " self=", to_string(self), " device=", to_string(device));
  return pin_memory_hpu(self, device);
}

Tensor hpu_wrap::_pin_memory(
    const at::Tensor& self,
    c10::optional<at::Device> device) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "_pin_memory :",
      " self=",
      to_string(self),
      " device=",
      to_string(device));
  return pin_memory_hpu(self, device);
}
/*
 PT1.12 introduced a change in linear() to use addmm instead of matmul
 in case of 3d input. This caused a perf regression on HPU. In order to
 circumvent this regression we register linear() ( a compound op) so that
 it gets dispatched as such to HPU. We use essentially the same linear()
 impl. as in PyTorch but removing the PT1.12 change and other code not
 relevant to HPU. Ref. aten/src/ATen/native/Linear.cpp
 Ref. https://jira.habana-labs.com/browse/SW-93519 for details.
*/
Tensor linear_(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "HpuOp linear:",
      " input=",
      to_string(input),
      " weight=",
      to_string(weight),
      " bias_opt=",
      to_string(bias_opt));

  auto bias = bias_opt.has_value()
      ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
      : c10::MaybeOwned<Tensor>::owned(c10::in_place);
  if ((!GET_ENV_FLAG_NEW(PT_DO_NOT_LOWER_LINEAR_OP)) && input.dim() == 2 &&
      bias->defined()) {
    // Fused op is marginally faster.
    return at::addmm(*bias, input, weight.t());
  }
  return linear_non2d_hpu_lazy(input, weight, bias_opt);
}

Tensor& hpu_wrap::copy_(Tensor& self, const Tensor& src, bool non_blocking) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "copy_ :",
      " self=",
      to_string(self),
      " src=",
      to_string(src),
      " non_blocking=",
      to_string(non_blocking));
  return copy_hpu_lazy_(self, src, non_blocking);
}

Tensor hpu_wrap::_reshape_alias(
    const Tensor& self,
    SymIntArrayRef size,
    SymIntArrayRef stride) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "_reshape_alias :",
      " self=",
      to_string(self),
      " size=",
      to_string(size),
      " stride",
      to_string(stride));
  FALLBACK_IF_UNSUPPORTED_OP(
      _reshape_alias, PARAMS1(self), PARAMS2(self, size, stride))
  // TODO: In order to align the changes of bert with Pytorchv1.9 we used
  // view inplace of as_strided implementation for the reshape of tensor
  // with no-change.
  // We need to revert existing change and use only as_strided once we
  // establish the convergence with below changes.
  // Pytorch change: https://github.com/pytorch/pytorch/pull/61466
  //
  // Note: as_strided_hpu_lazy is enabled only for lazy eager mode '2'
  const auto& mode = GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
  if (mode == 2) {
    return as_strided_hpu_lazy(
        self,
        C10_AS_INTARRAYREF_SLOW(size),
        C10_AS_INTARRAYREF_SLOW(stride),
        self.storage_offset());
  }

  return view_hpu(self, size);
}

Tensor embedding_bag_sum_hpu_wrap(
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "embedding_bag_sum :",
      " input=",
      to_string(input),
      " indices=",
      to_string(indices),
      " offsets=",
      to_string(offsets),
      " valid_count=",
      to_string(valid_count),
      " kernel_mode=",
      to_string(kernel_mode));
  return embedding_bag_sum_hpu_lazy(
      input, indices, offsets, valid_count, kernel_mode);
}

Tensor& embedding_bag_sum_bwd_out_kernel_mode_hpu_wrap(
    Tensor& out,
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "embedding_bag_sum_bwd_out :",
      " out=",
      to_string(out),
      " input=",
      to_string(input),
      " indices=",
      to_string(indices),
      " offsets=",
      to_string(offsets),
      " valid_count=",
      to_string(valid_count),
      " kernel_mode=",
      to_string(kernel_mode));
  return embedding_bag_sum_bwd_out_kernel_mode_hpu_lazy(
      out, input, indices, offsets, valid_count, kernel_mode);
}

Tensor hpu_wrap::masked_select(const Tensor& self, const Tensor& mask) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "masked_select :", " self=", to_string(self), " mask=", to_string(mask));
  FALLBACK_IF_UNSUPPORTED_OP(
      masked_select, PARAMS1(self, mask), PARAMS2(self, mask))
  return masked_select_hpu_lazy(self, mask);
}

Tensor& hpu_wrap::masked_select_out(
    const Tensor& self,
    const Tensor& mask,
    Tensor& out) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "masked_select_out :",
      " self=",
      to_string(self),
      " mask=",
      to_string(mask),
      " out=",
      to_string(out));
  FALLBACK_IF_UNSUPPORTED_OP_O(
      masked_select, PARAMS1(self, mask, out), PARAMS2(self, mask, out), out)
  return masked_select_out_hpu_lazy(self, mask, out);
}

Tensor& hpu_wrap::scatter_add_(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "scatter_add_ :",
      " self=",
      to_string(self),
      " dim_=",
      to_string(dim_),
      " index=",
      to_string(index),
      " src=",
      to_string(src));
  FALLBACK_IF_UNSUPPORTED_OP(
      scatter_add_, PARAMS1(self, index, src), PARAMS2(self, dim_, index, src))

  return scatter_add_inplace_src_hpu_lazy(self, dim_, index, src);
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
        HPURegistrar::get_device().type() != synDeviceType::synDeviceGaudi)) {
    return dispatch_fallback<ATEN_OP(_index_put_impl_)>::call(
        OpSupportLevel::Value::unsupported_dtype,
        PARAMS2(self, indices, values, accumulate, unsafe));
  }
  return _index_put_impl_hpu_lazy_(self, indices, values, accumulate, unsafe);
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
  return nonzero_hpu_lazy(self);
}

Tensor& hpu_wrap::index_add_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    const Scalar& alpha,
    Tensor& out) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "index_add_out :",
      " self=",
      to_string(self),
      " dim=",
      to_string(dim),
      " index=",
      to_string(index),
      " source=",
      to_string(source),
      " alpha=",
      to_string(alpha),
      " out=",
      to_string(out));
  FALLBACK_IF_UNSUPPORTED_OP(
      index_add_out,
      PARAMS1(self, index, source, out),
      PARAMS2(self, dim, index, source, alpha, out))

  return index_add_hpu_lazy_out(self, dim, index, source, alpha, out);
}

Tensor& hpu_wrap::index_fill_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& value) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "index_fill_ :",
      " self=",
      to_string(self),
      " dim=",
      to_string(dim),
      "index=",
      to_string(index),
      "value=",
      to_string(value));
  return index_fill_hpu_lazy_(self, dim, index, value);
}

Tensor& hpu_wrap::nonzero_out(const Tensor& self, Tensor& out) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "nonzero_out :", " self=", to_string(self), " out=", to_string(out));
  FALLBACK_IF_UNSUPPORTED_OP(
      nonzero_out, PARAMS1(self, out), PARAMS2(self, out))

  return nonzero_out_hpu_lazy(self, out);
}

::std::tuple<at::Tensor, at::Tensor> hpu_wrap::batch_norm_stats(
    const at::Tensor& input,
    double eps) {
  return batch_norm_stats_lazy(input, eps);
}
at::Tensor hpu_wrap::batch_norm_elemt(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    double eps) {
  return batch_norm_elemt_lazy(input, weight, bias, mean, invstd, eps);
}
at::Tensor hpu_wrap::batch_norm_backward_elemt(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const c10::optional<at::Tensor>& weight,
    const at::Tensor& mean_dy,
    const at::Tensor& mean_dy_xmu,
    const at::Tensor& count) {
  return batch_norm_backward_elemt_lazy(
      grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu, count);
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hpu_wrap::
    batch_norm_backward_reduce(
        const at::Tensor& grad_out,
        const at::Tensor& input,
        const at::Tensor& mean,
        const at::Tensor& invstd,
        const c10::optional<at::Tensor>& weight,
        bool input_g,
        bool weight_g,
        bool bias_g) {
  return batch_norm_backward_reduce_lazy(
      grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
}

::std::tuple<at::Tensor, at::Tensor> hpu_wrap::
    batch_norm_gather_stats_with_counts(
        const at::Tensor& input,
        const at::Tensor& mean,
        const at::Tensor& invstd,
        const c10::optional<at::Tensor>& running_mean,
        const c10::optional<at::Tensor>& running_var,
        double momentum,
        double eps,
        const at::Tensor& counts) {
  return batch_norm_gather_stats_with_counts_lazy(
      input, mean, invstd, running_mean, running_var, momentum, eps, counts);
}

Tensor hpu_wrap::instance_norm(
    const Tensor& input,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    [[maybe_unused]] bool use_input_stats,
    [[maybe_unused]] double momentum,
    double eps,
    [[maybe_unused]] bool cudnn_enabled) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "instance_norm :",
      " input=",
      to_string(input),
      " weight_opt=",
      to_string(weight_opt),
      " bias_opt=",
      to_string(bias_opt),
      " running_mean_opt=",
      to_string(running_mean_opt),
      " running_var_opt=",
      to_string(running_var_opt),
      " use_input_stats=",
      to_string(use_input_stats),
      " momentum=",
      to_string(momentum),
      " eps=",
      to_string(eps),
      " cudnn_enabled=",
      to_string(cudnn_enabled));
  // Note: Legacy eager mode is not supported
  auto weight = weight_opt.value_or(
      at::ones(input.sizes().vec()[1]).to(torch::kFloat32).to(torch::kHPU));
  auto bias = bias_opt.value_or(
      at::zeros(input.sizes().vec()[1]).to(torch::kFloat32).to(torch::kHPU));

  auto running_mean = running_mean_opt.value_or(Tensor());
  auto running_var = running_var_opt.value_or(Tensor());

  if (GET_ENV_FLAG_NEW(PT_HPU_DISABLE_INSTANCE_NORM)) {
    return at::native::instance_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        use_input_stats,
        momentum,
        eps,
        cudnn_enabled);
  }

  struct InstanceNorm : public torch::autograd::Function<InstanceNorm> {
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const Tensor& input,
        const Tensor& weight, // gamma
        const Tensor& bias, // beta
        double eps) {
      auto input_maybe_reshaped = input;
      const auto is_3d = input.dim() == 3;
      if (is_3d) {
        auto new_shape = input.sizes().vec();
        new_shape.push_back(1);
        input_maybe_reshaped = at::reshape(input, new_shape);
      }
      Tensor output_maybe_reshaped, output, mean, istd;
      std::tie(output_maybe_reshaped, mean, istd) =
          instance_norm_hpu_lazy(input_maybe_reshaped, weight, bias, eps);

      if (is_3d) {
        output = at::reshape(output_maybe_reshaped, input.sizes());
      } else {
        output = output_maybe_reshaped;
      }

      ctx->save_for_backward({input, mean, istd, weight});

      return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_in) {
      auto saved = ctx->get_saved_variables();
      auto input = saved[0];
      auto mean = saved[1];
      auto istd = saved[2];
      auto gamma = saved[3];

      Tensor grad_out, grad_out_maybe_reshaped, grad_beta, grad_gamma;
      auto input_maybe_reshaped = input;
      auto grad_in_maybe_reshaped = grad_in[0];
      const auto is_3d = input.dim() == 3;
      if (is_3d) {
        auto new_shape = input.sizes().vec();
        new_shape.push_back(1);
        input_maybe_reshaped = at::reshape(input, new_shape);
        grad_in_maybe_reshaped = at::reshape(grad_in[0], new_shape);
      }

      std::tie(grad_out_maybe_reshaped, grad_beta, grad_gamma) =
          instance_norm_backward_hpu_lazy(
              input_maybe_reshaped, grad_in_maybe_reshaped, mean, istd, gamma);

      if (is_3d) {
        grad_out = at::reshape(grad_out_maybe_reshaped, input.sizes());
      } else {
        grad_out = grad_out_maybe_reshaped;
      }

      // Autograds same number of gradients as the number of forward inputs and
      // in the same order
      //  grad_eps
      auto grad_eps = Tensor();
      return {grad_out, grad_gamma, grad_beta, grad_eps};
    }
  };

  return InstanceNorm::apply(input, weight, bias, eps);
}

at::Tensor hpu_wrap::repeat_interleave(
    const at::Tensor& repeats,
    c10::optional<int64_t> output_size) {
  habana_lazy::NoAccThread no_acc_thread;
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "repeat_interleave:",
      "repeats=",
      to_string(repeats),
      " output_size=",
      to_string(output_size));
  FALLBACK_IF_UNSUPPORTED_OP_O(
      repeat_interleave,
      PARAMS1(repeats),
      PARAMS2(repeats, output_size),
      Tensor)
  return repeat_inlv_hpu_lazy(repeats, output_size);
}

struct SoftmaxFunction : public torch::autograd::Function<SoftmaxFunction> {
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor input,
      int64_t dim,
      c10::optional<at::ScalarType> dtype) {
    Tensor converted = dtype.has_value() ? input.toType(dtype.value()) : input;
    auto result = torch::_softmax(converted, dim, false);
    ctx->save_for_backward({result, input});
    ctx->saved_data["dim"] = dim;
    return result;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    torch::autograd::variable_list saved_vars = ctx->get_saved_variables();
    auto output = saved_vars[0];
    auto input = saved_vars[1];
    auto dim = ctx->saved_data["dim"].toInt();
    auto result = torch::_softmax_backward_data(
        grad_output[0], output, dim, input.scalar_type());
    return {result, torch::Tensor(), torch::Tensor()};
  }
};

Tensor hpu_wrap::softmax(
    const Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "softmax :",
      " self=",
      to_string(self),
      " dim=",
      to_string(dim),
      " dtype=",
      to_string(dtype));
  return SoftmaxFunction::apply(self, dim, dtype);
}

Tensor hpu_wrap::empty(
    SymIntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<MemoryFormat> optional_memory_format) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "empty :",
      " size=",
      to_string(size),
      " dtype=",
      to_string(dtype),
      " layout=",
      to_string(layout),
      " device=",
      to_string(device),
      " pin_memory=",
      to_string(pin_memory),
      " optional_memory_format=",
      to_string(optional_memory_format));
  at::TensorOptions options = at::TensorOptions()
                                  .dtype(dtype)
                                  .layout(layout)
                                  .pinned_memory(pin_memory)
                                  .device(device);

  return empty_hpu_lazy(
      C10_AS_INTARRAYREF_SLOW(size), options, optional_memory_format);
}

Tensor hpu_wrap::empty_strided(
    SymIntArrayRef size,
    SymIntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "empty_strided :",
      " size=",
      to_string(size),
      " stride=",
      to_string(stride),
      " dtype=",
      to_string(dtype),
      " layout=",
      to_string(layout),
      " device=",
      to_string(device),
      " pin_memory=",
      to_string(pin_memory));
  at::TensorOptions options = at::TensorOptions()
                                  .dtype(std::move(dtype))
                                  .layout(std::move(layout))
                                  .pinned_memory(std::move(pin_memory))
                                  .device(std::move(device));
  return empty_strided_hpu_lazy(
      C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride), options);
}

std::vector<Tensor> hpu_wrap::split_with_sizes(
    const Tensor& self,
    c10::SymIntArrayRef split_sizes,
    int64_t dim) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "split_with_sizes :",
      " self=",
      to_string(self),
      " split_sizes=",
      to_string(split_sizes),
      " dim=",
      to_string(dim));
  FALLBACK_IF_UNSUPPORTED_OP(
      split_with_sizes, PARAMS1(self), PARAMS2(self, split_sizes, dim))

  return split_with_sizes_hpu_lazy(
      self, C10_AS_INTARRAYREF_SLOW(split_sizes), dim);
}

std::tuple<Tensor, Tensor> hpu_wrap::sort(
    const Tensor& self,
    int64_t dim,
    bool descending) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "sort :",
      " self=",
      to_string(self),
      " dim =",
      to_string(dim),
      " descending=",
      to_string(descending));
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(self), IValue(dim), IValue(descending)};
  check_handle->hpu_check_ivalues("sort", op_stack);
  FALLBACK_IF_UNSUPPORTED_OP1_RT(
      self.scalar_type(), sort, PARAMS1(self), PARAMS2(self, dim, descending))

  return sort_hpu_lazy(self, dim, descending);
}

Tensor hpu_wrap::_unsafe_view(
    const at::Tensor& self,
    c10::SymIntArrayRef size) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "_unsafe_view:", " self=", to_string(self), " size=", to_string(size));
  FALLBACK_IF_UNSUPPORTED_OP(_unsafe_view, PARAMS1(self), PARAMS2(self, size))

  return view_hpu(self, size);
}

std::vector<at::Tensor> hpu_wrap::split(
    const at::Tensor& self,
    c10::SymInt split_size_symint,
    int64_t dim) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "split :",
      " self=",
      to_string(self),
      " split_size=",
      to_string(split_size_symint),
      " dim=",
      to_string(dim));
  FALLBACK_IF_UNSUPPORTED_OP_O(
      split, PARAMS1(self), PARAMS2(self, split_size_symint, dim), Tensor)

  // lower aten::split as split_with_sizes using the logic used in Fork
  int64_t dim_size = self.size(dim);
  auto split_size = split_size_symint.expect_int();
  TORCH_CHECK(
      split_size > 0 || self.size(dim) == 0,
      "split_size can only be 0 if dimension size is 0, "
      "but got dimension size of ",
      dim_size);

  // if split_size is 0 and dimension size is 0, there is 1 split.
  int64_t num_splits = 1;
  if (split_size != 0) {
    // ensuring num_splits is at least 1 makes consistent the case where
    // split_size > dim_size (returns a single split).  We might want to error
    // here, but keep it for BC.
    num_splits = std::max<int64_t>((dim_size + split_size - 1) / split_size, 1);
  }

  std::vector<c10::SymInt> splits(num_splits);
  int64_t last_split_size = split_size - (split_size * num_splits - dim_size);

  for (int64_t i = 0; i < num_splits; ++i) {
    auto length = i < num_splits - 1 ? split_size : last_split_size;
    splits[i] = c10::SymInt(length);
  }

  c10::SymIntArrayRef split_sizes(splits);
  return hpu_wrap::split_with_sizes(self, split_sizes, dim);
}

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
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "optimizer_sparse_sgd_with_valid_count :",
      " gradients=",
      to_string(gradients),
      " weights_in=",
      to_string(weights_in),
      " moments_in=",
      to_string(moments_in),
      " indices=",
      to_string(indices),
      " learning_rate=",
      to_string(learning_rate),
      " valid_count_tensor",
      to_string(valid_count_tensor),
      " mom",
      to_string(mom),
      " nesterov",
      to_string(nesterov));
  return optimizer_sparse_sgd_with_valid_count_hpu_lazy(
      gradients,
      weights_in,
      moments_in,
      indices,
      learning_rate,
      valid_count_tensor,
      mom,
      nesterov);
}

std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_adagrad_with_valid_count_hpu_wrap(
    const Tensor& gradients,
    Tensor& weights_in,
    Tensor& moments_in,
    const Tensor& indices,
    const Tensor& learning_rate,
    const Tensor& valid_count_tensor) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "optimizer_sparse_adagrad_with_valid_count :",
      " gradients=",
      to_string(gradients),
      " weights_in=",
      to_string(weights_in),
      " moments_in=",
      to_string(moments_in),
      " indices=",
      to_string(indices),
      " learning_rate=",
      to_string(learning_rate),
      " valid_count_tensor",
      to_string(valid_count_tensor));
  return optimizer_sparse_adagrad_with_valid_count_hpu_lazy(
      gradients,
      weights_in,
      moments_in,
      indices,
      learning_rate,
      valid_count_tensor);
}

void optimizer_adamw_hpu_wrap(
    const at::TensorList gradient_vec,
    at::TensorList weight_vec,
    at::TensorList exp_avg_vec,
    at::TensorList exp_avg_sq_vec,
    const at::Tensor& neg_step_t,
    const double beta1,
    const double beta2,
    const double epsilon,
    const double weight_decay) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "optimizer_adamw :",
      DUMP_9ARGS(
          gradient_vec,
          weight_vec,
          exp_avg_vec,
          exp_avg_sq_vec,
          neg_step_t,
          beta1,
          beta2,
          epsilon,
          weight_decay));

  TORCH_CHECK(
      (weight_vec.size() > 0),
      "optimizer_adamw : can not process empty weight vector");

  optimizer_adamw_hpu_lazy(
      gradient_vec,
      weight_vec,
      exp_avg_vec,
      exp_avg_sq_vec,
      neg_step_t,
      beta1,
      beta2,
      epsilon,
      weight_decay);
}

Tensor fused_norm_hpu_wrap(
    std::vector<at::Tensor>& grad,
    const Tensor& max_norm,
    float norm_type) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "fused_norm :",
      " grad=",
      to_string(grad),
      " max_norm=",
      to_string(max_norm),
      " norm_type=",
      to_string(norm_type));
  TORCH_CHECK((grad.size() > 0), "Can not process empty grad vector");
  return fused_norm_hpu_lazy(grad, max_norm, norm_type);
}

void optimizer_adagrad_hpu_wrap(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& variances,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float lrd,
    const float epsilon) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " optimizer_adagrad:",
      " gradients=",
      to_string(gradients),
      " weights=",
      to_string(weights),
      " variances=",
      to_string(variances),
      " epoch_num=",
      to_string(epoch_num),
      " lr=",
      to_string(lr),
      " wd=",
      to_string(wd),
      " lrd=",
      to_string(lrd),
      " epsilon=",
      to_string(epsilon));
  optimizer_adagrad_hpu_lazy(
      gradients, weights, variances, epoch_num, lr, wd, lrd, epsilon);
}

void optimizer_ema_hpu_wrap(
    const TensorList model_inputs,
    TensorList updated_ema,
    const at::Tensor& decay) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(" optimizer_ema :", DUMP_3ARGS(model_inputs, updated_ema, decay));

  LazyOp<void> hpu_op{
      "hpu::optimizer_ema",
      {model_inputs, updated_ema, decay},
      [](const at::Stack&) { return std::vector<std::vector<int64_t>>{}; },
      -1};

  runInplaceMaybeWithAccThread(
      "hpu::optimizer_ema", std::move(hpu_op), updated_ema);
}

void optimizer_sgd_hpu_wrap(
    const TensorList& gradients,
    TensorList& weights,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " optimizer_sgd:",
      " gradients=",
      to_string(gradients),
      " weights=",
      to_string(weights),
      " lr=",
      to_string(lr),
      " wd=",
      to_string(wd),
      " mom=",
      to_string(mom),
      " damp=",
      to_string(damp),
      " nesterov=",
      to_string(nesterov));
  optimizer_sgd_hpu_lazy(gradients, weights, lr, wd, mom, damp, nesterov);
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
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " optimizer_sgd_momentum:",
      " gradients=",
      to_string(gradients),
      " weights=",
      to_string(weights),
      " momentum=",
      to_string(momentum),
      " epoch_num=",
      to_string(epoch_num),
      " lr=",
      to_string(lr),
      " wd=",
      to_string(wd),
      " mom=",
      to_string(mom),
      " damp=",
      to_string(damp),
      " nesterov=",
      to_string(nesterov));
  optimizer_sgd_momentum_hpu_lazy(
      gradients, weights, momentum, epoch_num, lr, mom, wd, damp, nesterov);
}

void optimizer_lars_hpu_wrap(
    const at::TensorList params,
    at::TensorList grads,
    const std::vector<int64_t> skipMasks,
    const float eeta,
    const float weight_decay,
    const float eps,
    const float lr) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " optimizer_lars_hpu_wrap:",
      " param=",
      to_string(params),
      " grad=",
      to_string(grads),
      " skipMasks=",
      to_string(skipMasks),
      "eeta=",
      to_string(eeta),
      "weight_decay=",
      to_string(weight_decay),
      "eps=",
      to_string(eps),
      "lr=",
      to_string(lr));
  return optimizer_lars_hpu_lazy(
      params, grads, skipMasks, eeta, weight_decay, eps, lr);
}

void optimizer_resource_apply_momentum_hpu_wrap(
    at::TensorList params_momentum_buf_list,
    const at::TensorList dp_list,
    const double momentum) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "optimizer_resource_apply_momentum :",
      DUMP_3ARGS(params_momentum_buf_list, dp_list, momentum));

  LazyOp<void> hpu_op{
      "hpu::optimizer_resource_apply_momentum",
      {params_momentum_buf_list, dp_list, momentum},
      [](const at::Stack&) { return std::vector<std::vector<int64_t>>{}; },
      -1};

  runInplaceMaybeWithAccThread(
      "hpu::optimizer_resource_apply_momentum",
      std::move(hpu_op),
      params_momentum_buf_list);
}

Tensor torchvision_nms_hpu_wrap(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    double iou_threshold) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " torchvision_nms:",
      " boxes=",
      to_string(boxes),
      "scores=",
      to_string(scores),
      "iou_threshold=",
      to_string(iou_threshold));

  return habana_nms_hpu_lazy(boxes, scores, iou_threshold);
}

Tensor batched_nms_hpu_wrap(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    const at::Tensor& indices,
    float iou_threshold) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " batched_nms:",
      " boxes=",
      to_string(boxes),
      "scores=",
      to_string(scores),
      "indices=",
      to_string(indices),
      "iou_threshold=",
      to_string(iou_threshold));
  return batched_nms_hpu_lazy(boxes, scores, indices, iou_threshold);
}

std::tuple<Tensor&, Tensor&> cast_to_fp8_wrap(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& amax) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " cast_to_fp8:",
      " input=",
      to_string(input),
      " scale=",
      to_string(scale),
      ", stochastic_rounding=",
      to_string(stochastic_rounding));
  if (synapse_helpers::device_supports_fp8(HPURegistrar::get_device().type())) {
    return cast_to_fp8_lazy(input, scale, stochastic_rounding, out, amax);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}

std::tuple<Tensor, Tensor> cast_to_fp8_v2_wrap(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    bool is_amax,
    c10::optional<at::ScalarType> dtype) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " cast_to_fp8_v2:",
      DUMP_5ARGS(input, scale, stochastic_rounding, is_amax, dtype));
  if (synapse_helpers::device_supports_fp8(
          habana::HPURegistrar::get_device().type())) {
    return cast_to_fp8_v2_lazy(
        input, scale, stochastic_rounding, is_amax, dtype);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}

std::tuple<Tensor&, Tensor&, Tensor&> fp8_cast_transpose_wrap(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& transposed,
    at::Tensor& amax) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " fp8_cast_transpose:",
      " input=",
      to_string(input),
      " scale=",
      to_string(scale),
      ", stochastic_rounding=",
      to_string(stochastic_rounding));
  if (synapse_helpers::device_supports_fp8(HPURegistrar::get_device().type())) {
    return fp8_cast_transpose_lazy(
        input, scale, stochastic_rounding, out, transposed, amax);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> fp8_cast_transpose_bgrad_wrap(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& transposed,
    at::Tensor& bgrad,
    at::Tensor& amax) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " fp8_cast_transpose_bgrad:",
      " input=",
      to_string(input),
      " scale=",
      to_string(scale),
      ", stochastic_rounding=",
      to_string(stochastic_rounding));
  if (synapse_helpers::device_supports_fp8(HPURegistrar::get_device().type())) {
    return fp8_cast_transpose_bgrad_lazy(
        input, scale, stochastic_rounding, out, transposed, bgrad, amax);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
std::tuple<Tensor&, Tensor&, Tensor&, Tensor&>
fp8_cast_transpose_bgrad_dgelu_wrap(
    const at::Tensor& grad,
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    const c10::optional<at::Tensor>& retain,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& transposed,
    at::Tensor& bgrad,
    at::Tensor& amax) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " fp8_cast_transpose_bgrad:",
      " input=",
      to_string(input),
      " scale=",
      to_string(scale),
      ", stochastic_rounding=",
      to_string(stochastic_rounding));
  if (synapse_helpers::device_supports_fp8(HPURegistrar::get_device().type())) {
    return fp8_cast_transpose_bgrad_dgelu_lazy(
        grad,
        input,
        scale,
        retain,
        stochastic_rounding,
        out,
        transposed,
        bgrad,
        amax);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
Tensor cast_from_fp8_wrap(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    at::ScalarType out_dtype) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " cast_from_fp8:",
      " input=",
      to_string(input),
      " scale=",
      to_string(scale),
      " out_dtype=",
      to_string(out_dtype));
  if (synapse_helpers::device_supports_fp8(HPURegistrar::get_device().type())) {
    return cast_from_fp8_lazy(input, scale, out_dtype);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
std::tuple<Tensor, Tensor, Tensor> fp8_dropout_wrap(
    const at::Tensor& input,
    double p,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    bool is_amax,
    c10::optional<at::ScalarType> dtype) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " fp8_dropout:",
      DUMP_6ARGS(input, p, scale, stochastic_rounding, is_amax, dtype));
  if (synapse_helpers::device_supports_fp8(HPURegistrar::get_device().type())) {
    return fp8_dropout_lazy(
        input, p, scale, stochastic_rounding, is_amax, dtype);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
std::tuple<Tensor&, Tensor&, Tensor&> fp8_gelu_wrap(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& retain,
    at::Tensor& amax) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " fp8_gelu:",
      " input=",
      to_string(input),
      " scale=",
      to_string(scale),
      ", stochastic_rounding=",
      to_string(stochastic_rounding));
  if (synapse_helpers::device_supports_fp8(HPURegistrar::get_device().type())) {
    return fp8_gelu_lazy(input, scale, stochastic_rounding, out, retain, amax);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
std::tuple<Tensor, Tensor, Tensor> fp8_gelu_v2_wrap(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    bool is_amax,
    c10::optional<at::ScalarType> dtype) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " fp8_gelu_v2:",
      DUMP_5ARGS(input, scale, stochastic_rounding, is_amax, dtype));
  if (synapse_helpers::device_supports_fp8(
          habana::HPURegistrar::get_device().type())) {
    return fp8_gelu_v2_lazy(input, scale, stochastic_rounding, is_amax, dtype);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
std::tuple<Tensor, Tensor, Tensor> fp8_bgrad_dgelu_wrap(
    const at::Tensor& grad,
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    const c10::optional<at::Tensor>& retain,
    bool stochastic_rounding,
    bool is_amax,
    c10::optional<at::ScalarType> dtype) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " fp8_bgrad_dgelu:",
      DUMP_7ARGS(
          grad, input, scale, retain, stochastic_rounding, is_amax, dtype));
  if (synapse_helpers::device_supports_fp8(
          habana::HPURegistrar::get_device().type())) {
    return fp8_bgrad_dgelu_lazy(
        grad, input, scale, retain, stochastic_rounding, is_amax, dtype);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
std::tuple<Tensor, Tensor> fp8_fast_softmax_wrap(
    const at::Tensor& input,
    const at::Tensor& mask,
    const c10::optional<at::Tensor>& scale,
    double softmax_scale,
    bool stochastic_rounding,
    bool is_amax,
    c10::optional<at::ScalarType> dtype) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " fp8_fast_softmax:",
      DUMP_7ARGS(
          input,
          mask,
          scale,
          softmax_scale,
          stochastic_rounding,
          is_amax,
          dtype));
  if (synapse_helpers::device_supports_fp8(
          habana::HPURegistrar::get_device().type())) {
    return fp8_fast_softmax_lazy(
        input, mask, scale, softmax_scale, stochastic_rounding, is_amax, dtype);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> fp8_layernorm_wrap(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double eps,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& mean,
    at::Tensor& istd,
    at::Tensor& amax) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " fp8_layernorm:",
      " input=",
      to_string(input),
      " weight=",
      to_string(weight),
      " bias=",
      to_string(bias),
      " eps=",
      to_string(eps),
      " scale=",
      to_string(scale),
      ", stochastic_rounding=",
      to_string(stochastic_rounding));
  if (synapse_helpers::device_supports_fp8(HPURegistrar::get_device().type())) {
    return fp8_layernorm_lazy(
        input,
        weight,
        bias,
        eps,
        scale,
        stochastic_rounding,
        out,
        mean,
        istd,
        amax);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
Tensor& fp8_gemm_wrap(
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    const at::Tensor& D,
    at::ScalarType out_dtype,
    const c10::optional<at::Tensor>& A_scale_inv,
    const c10::optional<at::Tensor>& B_scale_inv,
    const c10::optional<at::Tensor>& bias,
    bool accumulate,
    at::Tensor& out) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " fp8_gemm:",
      " A=",
      to_string(A),
      " A_scale_inv=",
      to_string(A_scale_inv),
      " trans_A=",
      to_string(trans_A),
      " B=",
      to_string(B),
      " B_scale_inv=",
      to_string(B_scale_inv),
      " trans_B=",
      to_string(trans_B),
      " out_dtype=",
      to_string(out_dtype),
      " bias=",
      to_string(bias),
      " accumulate=",
      to_string(accumulate));
  if (synapse_helpers::device_supports_fp8(HPURegistrar::get_device().type())) {
    return fp8_gemm_lazy(
        A,
        trans_A,
        B,
        trans_B,
        D,
        out_dtype,
        A_scale_inv,
        B_scale_inv,
        bias,
        accumulate,
        out);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
Tensor fp8_gemm_v2_wrap(
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    const c10::optional<at::Tensor>& D,
    at::ScalarType out_dtype,
    const c10::optional<at::Tensor>& A_scale_inv,
    const c10::optional<at::Tensor>& B_scale_inv,
    const c10::optional<at::Tensor>& bias,
    bool accumulate) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " fp8_gemm_v2:",
      " A=",
      to_string(A),
      " A_scale_inv=",
      to_string(A_scale_inv),
      " trans_A=",
      to_string(trans_A),
      " B=",
      to_string(B),
      " B_scale_inv=",
      to_string(B_scale_inv),
      " trans_B=",
      to_string(trans_B),
      " out_dtype=",
      to_string(out_dtype),
      " bias=",
      to_string(bias),
      " accumulate=",
      to_string(accumulate));
  if (synapse_helpers::device_supports_fp8(HPURegistrar::get_device().type())) {
    return fp8_gemm_v2_lazy(
        A,
        trans_A,
        B,
        trans_B,
        D,
        out_dtype,
        A_scale_inv,
        B_scale_inv,
        bias,
        accumulate);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
at::Tensor& fp8_transpose_wrap(const at::Tensor& input, at::Tensor& out) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(" fp8_transpose:", " input=", to_string(input));
  if (synapse_helpers::device_supports_fp8(HPURegistrar::get_device().type())) {
    return fp8_transpose_lazy(input, out);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
at::Tensor& fp8_permute_wrap(
    const at::Tensor& input,
    at::IntArrayRef dims,
    at::Tensor& out) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " fp8_permute:", " input=", to_string(input), " dims=", to_string(dims));
  if (synapse_helpers::device_supports_fp8(HPURegistrar::get_device().type())) {
    return fp8_permute_lazy(input, dims, out);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
at::Tensor fp8_reshape_wrap(const at::Tensor& input, at::IntArrayRef shape) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(" fp8_reshape:", " input=", to_string(input));
  if (synapse_helpers::device_supports_fp8(HPURegistrar::get_device().type())) {
    return fp8_reshape_lazy(input, shape);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
at::Tensor matmul_ex_wrap(
    const at::Tensor& self,
    const at::Tensor& other,
    at::ScalarType dtype) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  return matmul_hpu_lazy(self, other, dtype);
}
std::tuple<at::Tensor, at::Tensor> matmul_ex_backward_wrap(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& other,
    at::ScalarType dtype) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  return matmul_backward_hpu_lazy(grad_output, self, other, dtype);
}
at::Tensor linear_ex_wrap(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::ScalarType dtype) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  return linear_non2d_hpu_lazy(input, weight, bias_opt, dtype);
}
std::vector<at::Tensor> linear_ex_backward_wrap(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& bias_grad_opt,
    const at::ScalarType dtype) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  return linear_non2d_bwd_hpu_lazy(
      grad_output, input, weight, bias_opt, bias_grad_opt, dtype);
}

Tensor habana_random_seed_wrap(const at::Tensor& input) {
  PT_OP_TRACE;
  PT_OP_INFO(" habana_random_seed:", " input=", to_string(input));
  return habana_random_seed_lazy(input);
}

std::vector<at::Tensor> habana_permute_1D_sparse_data_wrap(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& weights) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " permute_1D_sparse_data:",
      " permute=",
      to_string(permute),
      " lengths=",
      to_string(lengths),
      " indices=",
      to_string(indices),
      " weights=",
      to_string(weights));

  return habana_permute_1D_sparse_data_lazy(permute, lengths, indices, weights);
}

std::vector<at::Tensor> habana_permute_2D_sparse_data_wrap(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& weights) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " permute_2D_sparse_data:",
      " permute=",
      to_string(permute),
      " lengths=",
      to_string(lengths),
      " indices=",
      to_string(indices),
      " weights=",
      to_string(weights));

  return habana_permute_2D_sparse_data_lazy(permute, lengths, indices, weights);
}

at::Tensor habana_expand_into_jagged_permute_wrap(
    const at::Tensor& permute,
    const at::Tensor& input_offsets,
    const at::Tensor& output_offsets,
    int64_t output_size) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " expand_into_jagged_permute:",
      " permute=",
      to_string(permute),
      " input_offsets=",
      to_string(input_offsets),
      " output_offsets=",
      to_string(output_offsets),
      " output_size=",
      to_string(output_size));

  return habana_expand_into_jagged_permute_lazy(
      permute, input_offsets, output_offsets, output_size);
}

at::Tensor habana_split_permute_cat_wrap(
    const at::Tensor& input,
    const at::Tensor& indices,
    int64_t batch_size,
    int64_t num_features,
    int64_t dims) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " split_permute_cat:",
      " input=",
      to_string(input),
      " indices=",
      to_string(indices),
      " batch_size=",
      to_string(batch_size),
      " num_features=",
      to_string(num_features),
      " dims=",
      to_string(dims));

  return habana_split_permute_cat_lazy(
      input, indices, batch_size, num_features, dims);
}

at::Tensor scaled_masked_softmax_wrap(
    const at::Tensor& input,
    const at::Tensor& mask,
    double scale) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " scaled_masked_softmax:",
      " input=",
      to_string(input),
      " mask=",
      to_string(mask),
      " scale=",
      to_string(scale));

  return scaled_masked_softmax_lazy(input, mask, scale);
}

at::Tensor custom_softmax_wrap(const at::Tensor& input, int64_t flavor) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " custom_softmax:",
      " input=",
      to_string(input),
      " flavor=",
      to_string(flavor));

  return custom_softmax_lazy(input, flavor);
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>
habana_bounds_check_indices_wrap(
    at::Tensor& indices,
    at::Tensor& offsets,
    at::Tensor& warning,
    const at::Tensor& rows_per_table,
    int64_t bounds_check_mode,
    const c10::optional<at::Tensor>& weights) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " bounds_check_indices:",
      " indices=",
      to_string(indices),
      " offsets=",
      to_string(offsets),
      " warning=",
      to_string(warning),
      " rows_per_table=",
      to_string(rows_per_table),
      " bounds_check_mode=",
      to_string(bounds_check_mode),
      " weights=",
      to_string(weights));

  return habana_bounds_check_indices_lazy(
      indices, offsets, warning, rows_per_table, bounds_check_mode, weights);
}

at::Tensor rotary_pos_embedding_wrap(
    const at::Tensor& input,
    const at::Tensor& sin,
    const at::Tensor& cos,
    const c10::optional<at::Tensor>& position_ids,
    const int64_t offset,
    const int64_t mode) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " rotary_pos_embedding :",
      DUMP_6ARGS(input, sin, cos, position_ids, offset, mode));

  return rotary_pos_embedding_lazy(input, sin, cos, position_ids, offset, mode);
}

at::Tensor rotary_pos_embedding_backward_wrap(
    const at::Tensor& grad_in,
    const at::Tensor& sin,
    const at::Tensor& cos,
    const int64_t offset) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " rotary_pos_embedding_backward :",
      DUMP_4ARGS(grad_in, sin, cos, offset));

  return rotary_pos_embedding_backward_lazy(grad_in, sin, cos, offset);
}

std::tuple<at::Tensor, at::Tensor> rms_norm_wrap(
    const at::Tensor& data_in,
    const at::Tensor& gamma,
    double epsilon) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(" rms_norm :", DUMP_3ARGS(data_in, gamma, epsilon));

  return rms_norm_lazy(data_in, gamma, epsilon);
}

std::tuple<at::Tensor, at::Tensor> rms_norm_backward_wrap(
    const at::Tensor& grad_in,
    const at::Tensor& data_in,
    const at::Tensor& gamma,
    const at::Tensor& inverse_rms) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " rms_norm_backward :", DUMP_4ARGS(grad_in, data_in, gamma, inverse_rms));

  return rms_norm_backward_lazy(grad_in, data_in, gamma, inverse_rms);
}

at::Tensor masked_batch_gemm_wrap(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& mask_a,
    const at::Tensor& mask_b,
    bool trans_a,
    bool trans_b) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "masked_batch_gemm :",
      DUMP_6ARGS(a, b, mask_a, mask_b, trans_a, trans_b));

  TORCH_CHECK(
      HPURegistrar::get_device().type() == synDeviceGaudi2,
      "masked_batch_gemm is supported only on Gaudi2.");
  return masked_batch_gemm_lazy(a, b, mask_a, mask_b, trans_a, trans_b);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_fwd_wrap(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " sdpa_fwd :", DUMP_7ARGS(q, k, v, attention_mask, p, scale, is_causal));

  return sdpa_fwd_lazy(q, k, v, attention_mask, p, scale, is_causal);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_bwd_wrap(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& P,
    const c10::optional<at::Tensor>& dm,
    const double p,
    const double scale) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(" sdpa_bwd :", DUMP_8ARGS(grad, q, k, v, P, dm, p, scale));

  return sdpa_bwd_lazy(grad, q, k, v, P, dm, p, scale);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> retain_softmax_producer_wrap(
    const at::Tensor& self) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO("retain_softmax_producer :", DUMP_ARG(self));

  return retain_softmax_producer_lazy(self);
}

at::Tensor retain_softmax_consumer_wrap(
    const at::Tensor& self,
    const at::Tensor& max,
    const at::Tensor& exp_sum_recpr) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " retain_softmax_consumer :", DUMP_3ARGS(self, max, exp_sum_recpr));

  return retain_softmax_consumer_lazy(self, max, exp_sum_recpr);
}

at::Tensor& fp8_copy_wrap(at::Tensor& self, const at::Tensor& src) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(DUMP_2ARGS(self, src));
  if (synapse_helpers::device_supports_fp8(HPURegistrar::get_device().type())) {
    return fp8_copy_lazy(self, src);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}

at::Tensor& fp8_kv_reorder_wrap(
    at::Tensor& self,
    const at::Tensor start,
    const at::Tensor end,
    const at::Tensor beam_idx) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(DUMP_4ARGS(self, start, end, beam_idx));
  if (synapse_helpers::device_supports_fp8(HPURegistrar::get_device().type())) {
    return fp8_kv_reorder_lazy(self, start, end, beam_idx);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}

at::Tensor& fp8_index_copy_wrap(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(DUMP_4ARGS(self, dim, index, source));
  if (synapse_helpers::device_supports_fp8(HPURegistrar::get_device().type())) {
    return fp8_index_copy_lazy(self, dim, index, source);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}

at::Tensor fp8_repeat_v2_wrap(
    const at::Tensor& self,
    c10::SymIntArrayRef repeats) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(DUMP_2ARGS(self, repeats));
  if (synapse_helpers::device_supports_fp8(HPURegistrar::get_device().type())) {
    return fp8_repeat_v2_lazy(self, repeats);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}

at::Tensor fp8_index_select_v2_wrap(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(DUMP_3ARGS(self, dim, index));
  if (synapse_helpers::device_supports_fp8(HPURegistrar::get_device().type())) {
    return fp8_index_select_v2_lazy(self, dim, index);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}

at::Tensor& kv_reorder_wrap(
    at::Tensor& self,
    const at::Tensor start,
    const at::Tensor end,
    const at::Tensor beam_idx) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(DUMP_4ARGS(self, start, end, beam_idx));

  return kv_reorder_lazy(self, start, end, beam_idx);
}

at::Tensor scaled_masked_triangular_softmax_wrap(
    const at::Tensor& self,
    const at::Tensor& start_end,
    double inv_scale_attn,
    int64_t grouped_batch_size,
    bool use_max,
    int64_t mode) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(DUMP_6ARGS(
      self, start_end, inv_scale_attn, grouped_batch_size, use_max, mode));

  return scaled_masked_triangular_softmax_lazy(
      self, start_end, inv_scale_attn, grouped_batch_size, use_max, mode);
}

at::Tensor& in_place_interleave_wrap(at::Tensor& self) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(DUMP_ARG(self));

  return in_place_interleave_lazy(self);
}

at::Tensor conv2d_fp8_wrap(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    c10::optional<at::ScalarType> out_dtype) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(DUMP_8ARGS(
      input, weight, bias, stride, padding, dilation, groups, out_dtype));

  if (synapse_helpers::device_supports_fp8(HPURegistrar::get_device().type())) {
    return conv2d_fp8_lazy(
        input, weight, bias, stride, padding, dilation, groups, out_dtype);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}

/***********************************************************************************
 * Kernels requiring autograd override
 **********************************************************************************/
using namespace torch::autograd;

struct MatmulFunction : public torch::autograd::Function<MatmulFunction> {
  static at::Tensor forward(
      AutogradContext* ctx,
      const at::Tensor& self,
      const at::Tensor& other) {
    at::Tensor result;
    ctx->save_for_backward({self, other});
    return matmul_hpu_lazy(self, other);
  }

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_output) {
    std::tuple<Tensor, Tensor> result;
    variable_list saved_vars = ctx->get_saved_variables();

    result =
        matmul_backward_hpu_lazy(grad_output[0], saved_vars[0], saved_vars[1]);

    return {std::get<0>(result), std::get<1>(result)};
  }
};

Tensor hpu_wrap::matmul(const Tensor& self, const Tensor& other) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO("matmul:", " self=", to_string(self), "other=", to_string(other));
  return MatmulFunction::apply(self, other);
}

struct LinearFunction : public torch::autograd::Function<LinearFunction> {
  static at::Tensor forward(
      AutogradContext* ctx,
      Tensor input,
      Tensor weight,
      c10::optional<Tensor> bias_opt) {
    auto bias = bias_opt.has_value()
        ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
        : c10::MaybeOwned<Tensor>::owned(c10::in_place);
    ctx->save_for_backward({input, weight, *bias});
    at::Tensor result;
    result = linear_(input, weight, bias_opt);
    return result;
  }

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_output) {
    std::tuple<Tensor, Tensor> result;
    variable_list saved_vars = ctx->get_saved_variables();
    Tensor input = saved_vars[0];
    Tensor weight = saved_vars[1];
    Tensor bias_opt = saved_vars[2];
    return linear_non2d_bwd_hpu_lazy(grad_output[0], input, weight, bias_opt);
  }
};

Tensor hpu_wrap::linear(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt) {
  PT_KERNEL_DEBUG(
      "HpuOp linear:",
      " input=",
      to_string(input),
      " weight=",
      to_string(weight),
      " bias_opt=",
      to_string(bias_opt));
  if (false == GET_ENV_FLAG_NEW(PT_HPU_ENABLE_COMPOUND_LOWERING_OPS)) {
    auto bias = bias_opt.has_value()
        ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
        : c10::MaybeOwned<Tensor>::owned(c10::in_place);
    if (input.dim() == 2 && bias->defined()) {
      // Fused op is marginally faster.
      return at::addmm(*bias, input, weight.t());
    }
    auto output = at::matmul(input, weight.t());
    if (bias->defined()) {
      output.add_(*bias);
    }
    return output;
  } else {
    return LinearFunction::apply(input, weight, bias_opt);
  }
}

Tensor hpu_wrap::slice(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<c10::SymInt> start,
    c10::optional<c10::SymInt> end,
    c10::SymInt step) {
  auto temp_start = start.has_value() ? start.value().expect_int() : 0;
  auto temp_end = end.has_value() ? end.value().expect_int() : INT64_MAX;
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " slice:",
      " self=",
      to_string(self),
      "dim=",
      to_string(dim),
      "start=",
      to_string(start),
      "end=",
      to_string(end),
      "step=",
      to_string(step));

  return slice_hpu_lazy(self, dim, temp_start, temp_end, step.expect_int());
}

struct DropoutFunction : public Function<DropoutFunction> {
  static at::Tensor forward(
      AutogradContext* ctx,
      at::Tensor input,
      double p,
      bool train) {
    ctx->saved_data["p"] = p;
    if ((p == 0) || !train || (input.numel() == 0)) {
      return input;
    } else if (p == 1) {
      return input * 0.0;
    }
    c10::optional<at::Generator> gen = c10::nullopt;
    at::Tensor result1, result2;
    std::tie(result1, result2) = _fused_dropout(input, p, gen);
    ctx->save_for_backward({result2});
    return result1;
  }

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_output) {
    auto p = ctx->saved_data["p"].toDouble();
    if (p == 0) {
      return {grad_output[0], torch::Tensor(), torch::Tensor()};
    } else if (p == 1) {
      return {grad_output[0] * 0.0, torch::Tensor(), torch::Tensor()};
    }
    variable_list saved_vars = ctx->get_saved_variables();
    auto mask = saved_vars[0];
    at::Tensor result;
    result = at::_masked_scale(grad_output[0], mask, 1.0 / p);
    return {result, torch::Tensor(), torch::Tensor()};
  }
};

Tensor hpu_wrap::dropout(const Tensor& input, double p, bool train) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " dropout:",
      " input=",
      to_string(input),
      "p=",
      to_string(p),
      "train=",
      to_string(train));
  return DropoutFunction::apply(input, p, train);
}

at::Tensor _ragged_softmax_wrap(
    const at::Tensor& self,
    int64_t dim,
    bool half_to_float,
    const at::Tensor& valid_count) {
  return habana_lazy::_ragged_softmax(self, dim, half_to_float, valid_count);
}

namespace vision {
namespace ops {
at::Tensor roi_align_fwd_wrap(
    const at::Tensor& images,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t output_h,
    int64_t output_w,
    int64_t sampling_ratio,
    bool aligned) {
  int mode = 0;
  // rois from torchvision are of shape {K, 5} where 1st column contain the
  // index of corresponding element in the batch, whereas remaining columns
  // contain the roi co-ordinates. Since "roi_align" TPC kernels expect
  // these indices and co-ordinates as separate tensors, therefore split
  // operation is being done here. TPC kernel expects a 1D Int tensor for
  // num_rois, therefore a reshape and conversion to Int is also done here.
  auto out = rois.split_with_sizes({1, 4}, 1);
  auto num_rois = out[0].view(-1).to(torch::kInt);
  auto roi = out[1];
  return roi_align_fwd_hpu_lazy(
      images,
      roi,
      num_rois,
      static_cast<int>(output_h),
      static_cast<int>(output_w),
      mode,
      static_cast<int>(sampling_ratio),
      static_cast<float>(spatial_scale),
      aligned);
}

at::Tensor roi_align_bwd_wrap(
    const at::Tensor& grad_out,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t output_h,
    int64_t output_w,
    int64_t bs,
    int64_t ch,
    int64_t h,
    int64_t w,
    int64_t sampling_ratio,
    bool aligned) {
  static_cast<void>(output_h);
  static_cast<void>(output_w);
  // Refer to comment in roi_align_fwd_wrap for same operations
  auto out = rois.split_with_sizes({1, 4}, 1);
  auto num_rois = out[0].view(-1).to(torch::kInt);
  auto roi = out[1];
  return roi_align_bwd_hpu_lazy(
      grad_out,
      roi,
      num_rois,
      static_cast<int>(bs),
      static_cast<int>(ch),
      static_cast<int>(h),
      static_cast<int>(w),
      static_cast<int>(sampling_ratio),
      static_cast<float>(spatial_scale),
      aligned);
}

TORCH_LIBRARY_IMPL(torchvision, HPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::roi_align"),
      TORCH_FN(roi_align_fwd_wrap));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_roi_align_backward"),
      TORCH_FN(roi_align_bwd_wrap));
}
} // namespace ops
} // namespace vision

TORCH_LIBRARY(hpu, m) {
  m.def("cat(Tensor[] tensors, int dim, Tensor out_shape) -> Tensor");
  m.def(
      "repeat_inlv(Tensor input, Tensor repeats, int dim, Tensor out_shape) -> Tensor");
  m.def("repeat_inlv_ht(Tensor input, Tensor repeats, int dim) -> Tensor");
  m.def(
      "nonzero(Tensor self, Tensor? nonzero_input_shape_tensor) -> (Tensor, Tensor)");
  m.def(
      "index_put(Tensor self, Tensor where_tensor, Tensor shape_tensor, Tensor value, Tensor value_upd_dim, Tensor zero_shape_tensor, bool accumulate=False) -> Tensor");
  m.def("mul_out(Tensor self, Tensor other, Tensor(a!) out) -> Tensor(a!)");
  m.def("div_out(Tensor self, Tensor other, Tensor(a!) out) -> Tensor(a!)");
  m.def("mm_t(Tensor mm, Tensor t , bool tr, bool no_tr) -> Tensor");
  m.def("habana_d2d_memcpy_other(Tensor s, Tensor(a!) d) -> Tensor(a!)");
  m.def(
      "prod_dim_Int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
  m.def("diag_out(Tensor self, int diagonal, Tensor(a!) out) -> Tensor(a!)");
  m.def("randperm_out(int n, Tensor seed, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "randperm_out_ds_ht(Tensor ht, Tensor seed, Tensor output) -> Tensor(a!)");
  m.def("habana_d2d_memcpy(Tensor self) -> Tensor");
  m.def(
      "habanaOptimizerSparseSgd(Tensor gradients, Tensor(a!) weights_in, Tensor(b!) moments_in, Tensor indices, Tensor learning_rate, Tensor valid_count_tensor, float mom, bool nesterov) -> (Tensor(a!), Tensor(b!))");
  m.def(
      "habanaOptimizerSparseAdagrad(Tensor gradients, Tensor(a!) weights_in, Tensor(b!) moments_in, Tensor indices, Tensor learning_rate, Tensor valid_count_tensor) -> (Tensor(a!), Tensor(b!))");
  m.def("cast(Tensor self, Scalar type) -> Tensor(a)");
  m.def(
      "embedding_bag_sum(Tensor input, Tensor indices, Tensor offsets, Tensor valid_count, int kernel_mode) -> Tensor");
  m.def(
      "embedding_bag_sum_bwd_out(Tensor(a!) out, Tensor input, Tensor indices_bwd, Tensor offsets_bwd, Tensor valid_count_bwd, int kernel_mode) -> Tensor(a!)");
  m.def(
      "habanaOptimizerFusedAdagrad(Tensor[] gradients, Tensor(a!)[] weights_in, Tensor(b!)[] variances_in, Tensor epoch_num, Tensor(c!) learning_rate, float wd, float lrd, float eps) -> ()");
  m.def(
      "habanaOptimizerFusedSGD(Tensor[] gradients, Tensor(a!)[] weights_in, Tensor(b!) learning_rate, float wd, float mom, float damp, bool nesterov) -> ()");
  m.def(
      "hpu::habanaOptimizerAdamW(Tensor[] gradient_vec, Tensor(a!)[] weight_vec, Tensor(b!)[] exp_avg_vec, Tensor(c!)[] exp_avg_sq_vec, Tensor neg_step_t, float beta1, float beta2, float epsilon, Tensor weight_decay, bool has_weight_decay) -> ()");
  m.def(
      "hpu::optimizer_adamw(Tensor[] gradient_vec, Tensor(a!)[] weight_vec, Tensor(b!)[] exp_avg_vec, Tensor(c!)[] exp_avg_sq_vec, Tensor neg_step_t, float beta1, float beta2, float epsilon, float weight_decay) -> ()");
  m.def(
      "hpu::optimizer_ema(Tensor[] model_inputs, Tensor(a!)[] updated_ema, Tensor decay) -> ()");
  m.def(
      "hpu::optimizer_lamb_fused_norm(Tensor[] grad, float max_norm) -> Tensor");
  m.def(
      "hpu::optimizer_lamb_phase1(Tensor[] gradients, Tensor[] weights, Tensor(a!)[] exp_avg, Tensor(b!)[] exp_avg_sq, Tensor(c!)[] out_weight_norms, Tensor(d!)[] out_adam_norms, Tensor(e!)[] out_adam_steps, Tensor clip_global_grad_norm, int grad_averaging, float beta1, float beta2, float epsilon, Tensor bias_correction1, Tensor bias_correction2, float weight_decay) -> ()");
  m.def(
      "hpu::optimizer_lamb_phase2(Tensor(a!)[] weights, Tensor[] adam_norms, Tensor[] weight_norms, Tensor[] adam_steps, Tensor neg_step, float wd, bool use_lamb) -> ()");
  m.def(
      "habanaOptimizerLars(Tensor[] params, Tensor(a!)[] grads, Tensor lr_t, int[] skip_masks, float eeta, float weight_decay, float eps) -> ()");
  m.def(
      "optimizer_resource_apply_momentum(Tensor(a!)[] params_momentum_buf_list, Tensor[] dp_list, float momentum) -> ()");
  m.def(
      "habana_nms(Tensor boxes, Tensor scores, float iou_threshold, float score_threshold) -> (Tensor, Tensor, Tensor)");
  m.def(
      "batched_nms(Tensor boxes, Tensor scores, Tensor indexes, float iou_threshold, Tensor shape_tensor1, Tensor shape_tensor2, int max_classes) -> (Tensor, Tensor)");
  m.def(
      "roi_align_fwd(Tensor inputs, Tensor rois, Tensor n_rois, int out_h, int out_w, int mode, int sr, float ss, bool aligned) -> Tensor");
  m.def(
      "roi_align_bwd(Tensor inputs, Tensor rois, Tensor n_rois, Tensor input_shape, int sr, float ss, bool aligned) -> Tensor");
  m.def(
      "_unique(Tensor self, bool sorted, bool return_inverse) -> (Tensor, Tensor)");
  m.def(
      "_unique2(Tensor self, bool sorted, bool return_inverse, bool return_counts) -> (Tensor, Tensor)");
  m.def(
      "unique_dim(Tensor self, int dim, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor)");
  m.def(
      "gather_elements(Tensor self, Tensor index, Tensor? opt, int64_t dim_, bool sorted) -> Tensor");
  m.def("permute(Tensor(a) self, int[] dims) -> Tensor(a)");
  m.def("permute_cl(Tensor(a) self, int[] dims) -> Tensor(a)");
  m.def("restride_cl(Tensor(a) self, int[] dims) -> Tensor(a)");
  m.def("restride(Tensor(a) self, int[] dims) -> Tensor(a)");
  m.def("permute_weight(Tensor self, int[] size) -> (Tensor)");
  m.def("permuted_weight_restride(Tensor self, int[] size) -> (Tensor)");
  m.def("control_edge_other_(Tensor self, Tensor(a) other) -> Tensor(a)");
  m.def("control_edge_(Tensor(a) self)-> Tensor(a)");
  m.def(
      "hpu::native_batch_norm_training(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::native_batch_norm_inf(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor)");
  m.def(
      "hpu::native_batch_norm_backward(Tensor input, Tensor? grad_out, Tensor? weight, Tensor? mean, Tensor? invistd, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::group_norm(Tensor input, Tensor weight, Tensor bias, int[] normalized_shape, int64_t num_groups, double eps) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::group_norm_backward(Tensor grad_out,Tensor input, Tensor mean, Tensor rstd, Tensor weight, int[] normalized_shape, int64_t num_groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)");
  m.def(
      "as_strided_lazy_(Tensor self, int[] size, int[] stride, int offset, bool can_replace) -> (Tensor)");
  m.def(
      "as_strided_lazy_cl_(Tensor self, int[] size, int[] stride, int offset, bool can_replace) -> (Tensor)");
  m.def(
      "strided_view(Tensor self, int[] size, int[] stride, int offset) -> (Tensor)");
  m.def(
      "strided_view_out(Tensor self, int[] size, int[] stride, int offset) -> (Tensor)");
  m.def(
      "strided_view_cl(Tensor self, int[] size, int[] stride, int offset) -> (Tensor)");
  m.def("strided_view_ds(Tensor self, Tensor size, Tensor offset) -> (Tensor)");
  m.def(
      "strided_view_ds_h2d(Tensor self, Tensor size, Tensor stride, Tensor offset) -> (Tensor)");
  m.def(
      "strided_view_out_ds(Tensor self, Tensor size, Tensor offset) -> (Tensor)");
  m.def(
      "strided_view_out_ds_h2d(Tensor self, Tensor size, Tensor stride, Tensor offset) -> (Tensor)");
  m.def(
      "strided_view_cl_ds(Tensor self, Tensor size,Tensor offset) -> (Tensor)");
  m.def("slice_insert(Tensor self, Tensor other, int[] params) -> (Tensor)");
  m.def(
      "slice_insert_ds(Tensor self, Tensor other, Tensor steps, Tensor start) -> (Tensor)");
  m.def(
      "slice_insert_ds_ht(Tensor self, Tensor other, Tensor host_tensor) -> (Tensor)");
  m.def(
      "strided_insert(Tensor self, Tensor other, int[] stride, int offset) -> (Tensor)");
  m.def(
      "strided_insert_cl(Tensor self, Tensor other, int[] stride, int offset) -> (Tensor)");
  m.def(
      "strided_insert_ds(Tensor self, Tensor other, Tensor offset) -> (Tensor)");
  m.def(
      "strided_insert_cl_ds(Tensor self, Tensor other, Tensor offset) -> (Tensor)");
  m.def(
      "strided_view_orig_ds(Tensor self, Tensor size, Tensor stride, Tensor offset) -> (Tensor)");
  m.def(
      "strided_view_orig_ds_h2d(Tensor self, Tensor size, Tensor stride) -> (Tensor)");
  m.def(
      "strided_view_out_orig_ds(Tensor self, Tensor size, Tensor stride, Tensor offset) -> (Tensor)");
  m.def(
      "strided_view_out_orig_ds_h2d(Tensor self, Tensor size, Tensor stride) -> (Tensor)");
  m.def(
      "strided_insert_orig_ds(Tensor self, Tensor other, Tensor stride, Tensor offset) -> (Tensor)");
  m.def(
      "strided_insert_orig_ds_h2d(Tensor self, Tensor other, Tensor stride) -> (Tensor)");
  m.def("as_strided_layout(Tensor self, int[] size) -> (Tensor)");
  m.def("reshape(Tensor self, int[] size) -> (Tensor)");
  m.def(
      "matmul_backward(Tensor grad_out, Tensor self, Tensor other) -> (Tensor, Tensor)");
  m.def(
      "instance_norm(Tensor input, Tensor weight, Tensor bias, float eps) -> (Tensor, Tensor, Tensor)");
  m.def(
      "instance_norm_backward(Tensor input, Tensor grad_in, Tensor mean, Tensor istd, Tensor gamma) -> (Tensor, Tensor, Tensor)");
  m.def("view(Tensor input, Tensor shape) -> Tensor");
  m.def(
      "slice(Tensor input, Tensor shape, Tensor step,  Tensor start) -> (Tensor)");
  m.def("slice_ht(Tensor input, Tensor shape, Tensor host_tensor) -> (Tensor)");
  m.def(
      "hpu::expand(Tensor(a) self, int[] sizes, *, bool implicit=False) -> Tensor(a)");
  m.def(
      "hpu::expand_ds(Tensor(a) self, Tensor shape, *, bool implicit=False) -> Tensor(a)");
  m.def("hpu::repeat_ht(Tensor self, Tensor result_shape) -> Tensor");
  m.def(
      "hpu::constant_pad_nd_ht(Tensor self, Tensor pad_tensor, Tensor output_shape_tensor, Scalar value) -> Tensor");
  m.def(
      "hpu::constant_pad_nd(Tensor self, int[] pad_tensor, Scalar value) -> Tensor");
  m.def(
      "hpu::scatter_nd_onnx(Tensor input, Tensor indices, Tensor values) -> Tensor");
  m.def(
      "hpu::scatter_nd(Tensor input, Tensor indices, Tensor grouped_indices, Tensor update_locations, Tensor updates) -> Tensor");
  m.def("hpu::add.Tensor(Tensor self, Tensor other, Scalar alpha) -> Tensor");
  m.def("hpu::add.Scalar(Tensor self, Scalar other, Scalar alpha) -> Tensor");
  m.def(
      "hpu::add_.Tensor(Tensor(a) self, Tensor other, Scalar alpha) -> Tensor(a)");
  m.def(
      "hpu::add_.Scalar(Tensor(a) self, Scalar other, Scalar alpha) -> Tensor(a)");
  m.def(
      "hpu::linear_bwd(Tensor grad_out, Tensor input, Tensor weight, bool bias_g=False) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::linear_ex_bwd(Tensor grad_out, Tensor input, Tensor weight, bool bias_g=False, Tensor? bias_grad_out=None) -> (Tensor, Tensor, Tensor)");
  m.def("hpu::identity(Tensor self) -> (Tensor)");
  m.def(
      "hpu::habana_cast_sr_mode(Tensor input, Scalar type, bool stochastic_rounding, int seed=0) -> (Tensor)");
  m.def(
      "hpu::cast_to_fp8(Tensor input, Tensor? scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) amax) -> (Tensor(a!), Tensor(b!))");
  m.def(
      "hpu::cast_to_fp8_v2(Tensor input, Tensor? scale=None, bool stochastic_rounding=False, bool is_amax=False, ScalarType? dtype=None) -> (Tensor, Tensor)");
  m.def(
      "hpu::fp8_cast_transpose(Tensor input, Tensor? scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) transposed, Tensor(c!) amax) -> (Tensor(a!), Tensor(b!), Tensor(c!))");
  m.def(
      "hpu::fp8_cast_transpose_bgrad(Tensor input, Tensor? scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) transposed, Tensor(c!) bgrad, Tensor(d!) amax) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))");
  m.def(
      "hpu::fp8_cast_transpose_bgrad_dgelu(Tensor grad, Tensor input, Tensor? scale, Tensor? retain, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) transposed, Tensor(c!) bgrad, Tensor(d!) amax) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))");
  m.def(
      "hpu::cast_from_fp8(Tensor input, Tensor? scale, ScalarType out_dtype) -> Tensor");
  m.def(
      "hpu::fp8_dropout(Tensor input, float p, Tensor? scale=None, bool stochastic_rounding=False, bool is_amax=False, ScalarType? dtype=None) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_gelu(Tensor input, Tensor? scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) retain, Tensor(c!) amax) -> (Tensor(a!), Tensor(b!), Tensor(c!))");
  m.def(
      "hpu::fp8_gelu_v2(Tensor input, Tensor? scale=None, bool stochastic_rounding=False, bool is_amax=False, ScalarType? dtype=None) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_bgrad_dgelu(Tensor grad, Tensor input, Tensor? scale=None, Tensor? retain=None, bool stochastic_rounding=False, bool is_amax=False, ScalarType? dtype=None) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_fast_softmax(Tensor input, Tensor mask, Tensor? scale, float softmax_scale, bool stochastic_rounding, bool is_amax, ScalarType? dtype=None) -> (Tensor, Tensor)");
  m.def(
      "hpu::fp8_layernorm(Tensor input, Tensor weight, Tensor bias, float eps, Tensor? scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) mean, Tensor(c!) istd, Tensor(d!) amax) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))");
  m.def(
      "hpu::fp8_gemm(Tensor A, bool trans_A, Tensor B, bool trans_B, Tensor D, ScalarType out_dtype, Tensor? A_scale_inv, Tensor? B_scale_inv, Tensor? bias, bool accumulate, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "hpu::fp8_gemm_v2(Tensor A, bool trans_A, Tensor B, bool trans_B, Tensor? D, ScalarType out_dtype, Tensor? A_scale_inv, Tensor? B_scale_inv, Tensor? bias, bool accumulate) -> Tensor");
  m.def("hpu::fp8_transpose(Tensor input, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "hpu::fp8_permute(Tensor input, int[] dims, Tensor(a!) out) -> Tensor(a!)");
  m.def("hpu::fp8_reshape(Tensor input, int[] shape) -> Tensor");
  m.def(
      "hpu::index_add(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor");
  m.def("hpu::habana_random_seed(Tensor input) -> (Tensor)");
  m.def(
      "hpu::habana_permute_1D_sparse_data(Tensor permute, Tensor lengths, Tensor indices, Tensor? weights=None) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::habana_permute_1D_sparse_data_without_weights(Tensor permute, Tensor lengths, Tensor indices) -> (Tensor, Tensor)");
  m.def(
      "hpu::habana_permute_2D_sparse_data(Tensor permute, Tensor lengths, Tensor indices, Tensor? weights=None) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::habana_permute_2D_sparse_data_without_weights(Tensor permute, Tensor lengths, Tensor indices) -> (Tensor, Tensor)");
  m.def(
      "hpu::habana_expand_into_jagged_permute(Tensor permute, Tensor input_offsets, Tensor output_offsets, int output_size) -> Tensor");
  m.def(
      "hpu::habana_split_permute_cat(Tensor input, Tensor indices, int batch_size, int num_features, int dims) -> Tensor");
  m.def(
      "hpu::ragged_softmax(Tensor self, int dim, bool half_to_float, Tensor valid_count) -> Tensor");
  m.def(
      "hpu::scaled_masked_softmax(Tensor input, Tensor mask, float scale) -> Tensor");
  m.def("hpu::custom_softmax(Tensor input, int flavor) -> Tensor");
  m.def(
      "hpu::habana_bounds_check_indices(Tensor(a!) indices, Tensor(b!) offsets, Tensor(c!) warning, Tensor rows_per_table, int bounds_check_mode, Tensor? weights) -> (Tensor(a!), Tensor(b!), Tensor(c!))");
  m.def(
      "hpu::rotary_pos_embedding(Tensor input, Tensor sin, Tensor cos, Tensor? position_ids, int offset, int mode) -> Tensor");
  m.def(
      "hpu::rotary_pos_embedding_backward(Tensor grad_in, Tensor sin, Tensor cos, int offset) -> Tensor");
  m.def(
      "hpu::rms_norm(Tensor data_in, Tensor gamma, float epsilon) -> (Tensor, Tensor)");
  m.def(
      "hpu::rms_norm_backward(Tensor grad_in, Tensor data_in, Tensor gamma, Tensor inverse_rms) -> (Tensor, Tensor)");
  m.def(
      "hpu::masked_batch_gemm(Tensor a, Tensor b, Tensor mask_a, Tensor mask_b, bool trans_a, bool trans_b) -> Tensor");

  // Seed is generated at FE and passed to BE. There is no seed at python
  // interface. So the schema with python interface and BE differ. Register the
  // op at python interface directly with the wrapper function to let Pytorch
  // Infer the schema at operator level. So no m.impl() def is needed for this
  // operator. For BE, Register a schema with seed.
  m.def("hpu::sdpa_fwd", sdpa_fwd_wrap);
  // sdpa_fwd_be schema is for sdpa op BE interface which takes an optional seed
  // tensor as well.
  m.def(
      "hpu::sdpa_fwd_be(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, Tensor? seed, float p, float scale, bool is_causal) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::sdpa_bwd(Tensor grad, Tensor q, Tensor k, Tensor v, Tensor P, Tensor? dm, float p, float scale) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::retain_softmax_producer(Tensor self) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::retain_softmax_consumer(Tensor self, Tensor max, Tensor exp_sum_recpr) -> Tensor");
  m.def("hpu::fp8_copy_(Tensor(a!) self, Tensor src) -> Tensor(a!)");
  m.def(
      "hpu::fp8_kv_reorder_(Tensor(a!) self, Tensor start, Tensor end, Tensor beam_idx) -> (Tensor(a!))");
  m.def(
      "hpu::kv_reorder_(Tensor(a!) self, Tensor start, Tensor end, Tensor beam_idx) -> (Tensor(a!))");
  m.def(
      "hpu::fp8_index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)");
  m.def("hpu::fp8_repeat_v2(Tensor self, SymInt[] repeats) -> Tensor");
  m.def(
      "hpu::fp8_index_select_v2(Tensor self, int dim, Tensor index) -> Tensor");
  m.def(
      "hpu::scaled_masked_triangular_softmax(Tensor self, Tensor start_end, float inv_scale_attn, int grouped_batch_size, bool use_max, int mode) -> Tensor");
  m.def("hpu::in_place_interleave_(Tensor(a!) self) -> (Tensor(a!))");
  m.def(
      "hpu::conv2d_fp8(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1, ScalarType? out_dtype=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu, HPU, m) {
  m.impl("hpu::cast_to_fp8", cast_to_fp8_wrap);
  m.impl("hpu::cast_to_fp8_v2", cast_to_fp8_v2_wrap);
  m.impl("hpu::fp8_cast_transpose", fp8_cast_transpose_wrap);
  m.impl("hpu::fp8_cast_transpose_bgrad", fp8_cast_transpose_bgrad_wrap);
  m.impl(
      "hpu::fp8_cast_transpose_bgrad_dgelu",
      fp8_cast_transpose_bgrad_dgelu_wrap);
  m.impl("hpu::cast_from_fp8", cast_from_fp8_wrap);
  m.impl("hpu::fp8_dropout", fp8_dropout_wrap);
  m.impl("hpu::fp8_gelu", fp8_gelu_wrap);
  m.impl("hpu::fp8_gelu_v2", fp8_gelu_v2_wrap);
  m.impl("hpu::fp8_bgrad_dgelu", fp8_bgrad_dgelu_wrap);
  m.impl("hpu::fp8_fast_softmax", fp8_fast_softmax_wrap);
  m.impl("hpu::fp8_layernorm", fp8_layernorm_wrap);
  m.impl("hpu::fp8_gemm", fp8_gemm_wrap);
  m.impl("hpu::fp8_gemm_v2", fp8_gemm_v2_wrap);
  m.impl("hpu::fp8_transpose", fp8_transpose_wrap);
  m.impl("hpu::ragged_softmax", _ragged_softmax_wrap);
  m.impl("hpu::scaled_masked_softmax", scaled_masked_softmax_wrap);
  m.impl("hpu::custom_softmax", custom_softmax_wrap);
  m.impl("hpu::fp8_reshape", fp8_reshape_wrap);
  m.impl("hpu::fp8_permute", fp8_permute_wrap);
  m.impl("hpu::optimizer_lamb_fused_norm", optimizer_lamb_norm_hpu_lazy);
  m.impl(
      "hpu::optimizer_resource_apply_momentum",
      optimizer_resource_apply_momentum_hpu_wrap);
  m.impl("hpu::optimizer_lamb_phase1", optimizer_lamb_phase1);
  m.impl("hpu::optimizer_lamb_phase2", optimizer_lamb_phase2);
  m.impl("hpu::optimizer_ema", optimizer_ema_hpu_wrap);
  m.impl("hpu::optimizer_adamw", optimizer_adamw_hpu_wrap);
  m.impl("hpu::rotary_pos_embedding", rotary_pos_embedding_wrap);
  m.impl(
      "hpu::rotary_pos_embedding_backward", rotary_pos_embedding_backward_wrap);
  m.impl("hpu::rms_norm", rms_norm_wrap);
  m.impl("hpu::rms_norm_backward", rms_norm_backward_wrap);
  m.impl("hpu::masked_batch_gemm", masked_batch_gemm_wrap);
  m.impl("hpu::sdpa_bwd", sdpa_bwd_wrap);
  m.impl("hpu::retain_softmax_producer", retain_softmax_producer_wrap);
  m.impl("hpu::retain_softmax_consumer", retain_softmax_consumer_wrap);
  m.impl("hpu::fp8_copy_", fp8_copy_wrap);
  m.impl("hpu::fp8_kv_reorder_", fp8_kv_reorder_wrap);
  m.impl("hpu::kv_reorder_", kv_reorder_wrap);
  m.impl("hpu::fp8_index_copy_", fp8_index_copy_wrap);
  m.impl("hpu::fp8_repeat_v2", fp8_repeat_v2_wrap);
  m.impl("hpu::fp8_index_select_v2", fp8_index_select_v2_wrap);
  m.impl(
      "hpu::scaled_masked_triangular_softmax",
      scaled_masked_triangular_softmax_wrap);
  m.impl("hpu::in_place_interleave_", in_place_interleave_wrap);
  m.impl("hpu::conv2d_fp8", conv2d_fp8_wrap);
}

TORCH_LIBRARY_IMPL(torchvision, HPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::nms"),
      TORCH_FN(torchvision_nms_hpu_wrap));
}

TORCH_LIBRARY(hccl, m) {
  m.def(
      "broadcast_(Tensor(a!) tensor, int root_rank, int64_t comm_id) -> Tensor(a!)");
  m.def(
      "allreduce_(Tensor(a!) tensor, uint8_t reduceOp, int64_t comm_id) -> Tensor(a!)");
  m.def(
      "reduce_(Tensor(a!) tensor, int64_t dst_rank, uint8_t reduceOp, int64_t comm_id) -> Tensor(a!)");
  m.def(
      "alltoall_out(Tensor input_tensor, int64_t comm_id,int[]  outputSplitSizes, int[] inputSplitSizes, Tensor(a!) output_tensor) -> Tensor(a!)");
  m.def(
      "allgather_out(Tensor input_tensor, int64_t comm_id, Tensor(a!) output_tensor) -> Tensor(a!)");
  m.def(
      "reduce_scatter_out(Tensor input_tensor, uint8_t reduceOp, int64_t comm_id, Tensor(a!) output_tensor) -> Tensor(a!)");
  m.def(
      "send_(Tensor(a!) tensor,  int64_t dst_rank,  int64_t tag, int64_t comm_id) -> Tensor(a!)");
  m.def(
      "recv_(Tensor(a!) tensor,  int64_t src_rank,  int64_t tag, int64_t comm_id) -> Tensor(a!)");
}
