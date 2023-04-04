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
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_kernels_ver/wrap_kernels_declarations.h"
#include "habana_lazy/lazy_executor.h"
#include "hpu_ops/cpu_fallback.h"
#include "kernel_input_checks.h"
#include "pytorch_helpers/habana_helpers/kernels_accumulation.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

using namespace torch;
using namespace at;
using namespace habana;
using namespace habana_lazy;

Tensor hpu_wrap::_to_copy(
    const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    bool non_blocking,
    c10::optional<MemoryFormat> optional_memory_format) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "_to_copy :",
      " self=",
      to_string(self),
      " dtype=",
      to_string(dtype),
      " layout=",
      to_string(layout),
      " device=",
      to_string(device),
      " pin_memory=",
      to_string(pin_memory),
      " non_blocking=",
      to_string(non_blocking),
      " optional_memory_format=",
      to_string(optional_memory_format));
  auto memory_format = optional_memory_format.value_or(MemoryFormat::Preserve);
  auto options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
  options = self.options().merge_in(options);
  if (memory_format == MemoryFormat::Preserve) {
    if (self.is_non_overlapping_and_dense()) {
      // Copy all strides
      auto r = at::empty_strided(
          self.sizes(), self.strides(), options.memory_format(c10::nullopt));
      r.copy_(self, non_blocking);
      return r;
    } else {
      memory_format = self.suggest_memory_format();
    }
  }

  auto r = at::empty(
      self.sizes(), options.memory_format(memory_format), c10::nullopt);
  r.copy_(self, non_blocking);
  return r;
}

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

#if IS_PYTORCH_OLDER_THAN(1, 13)
Tensor hpu_wrap::_reshape_alias(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride) {
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
    return as_strided_hpu_lazy(self, size, stride, self.storage_offset());
  }

  return view_hpu(self, size);
}

#else
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

#if IS_PYTORCH_OLDER_THAN(2, 1)
Tensor hpu_wrap::_efficientzerotensor(
    IntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "efficientzerotensor :",
      " size=",
      to_string(size),
      " dtype=",
      to_string(dtype),
      " layout=",
      to_string(layout),
      " device=",
      to_string(device),
      " pin_memory=",
      to_string(pin_memory));

  at::TensorOptions options = at::TensorOptions()
                                  .dtype(dtype)
                                  .layout(layout)
                                  .pinned_memory(pin_memory)
                                  .device(device);

  auto zero_tensor =
      empty_hpu_lazy(size, options, MemoryFormat::Contiguous, true);
  fill_hpu_lazy_(zero_tensor, 0);
  return zero_tensor;
}
#else
Tensor hpu_wrap::_efficientzerotensor(
    SymIntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "efficientzerotensor :",
      " size=",
      to_string(size),
      " dtype=",
      to_string(dtype),
      " layout=",
      to_string(layout),
      " device=",
      to_string(device),
      " pin_memory=",
      to_string(pin_memory));

  at::TensorOptions options = at::TensorOptions()
                                  .dtype(dtype)
                                  .layout(layout)
                                  .pinned_memory(pin_memory)
                                  .device(device);

  auto zero_tensor = empty_hpu_lazy(
      C10_AS_INTARRAYREF_SLOW(size), options, MemoryFormat::Contiguous, true);
  fill_hpu_lazy_(zero_tensor, 0);
  return zero_tensor;
}
#endif
#endif

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

Tensor& hpu_wrap::index_copy_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& value) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "index_copy_ :",
      " self=",
      to_string(self),
      " dim=",
      to_string(dim),
      "index=",
      to_string(index),
      "value=",
      to_string(value));
  return index_copy_hpu_lazy_(self, dim, index, value);
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

#if IS_PYTORCH_OLDER_THAN(1, 13)
Tensor hpu_wrap::kl_div_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    bool log_target) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "kl_div_backward :",
      " grad=",
      to_string(grad),
      " self=",
      to_string(self),
      " target=",
      to_string(target),
      " reduction=",
      to_string(reduction),
      " log_target=",
      to_string(log_target));
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(grad),
      IValue(self),
      IValue(target),
      IValue(reduction),
      IValue(log_target)};
  check_handle->hpu_check_ivalues("kl_div_backward", op_stack);

  FALLBACK_IF_UNSUPPORTED_OP1(
      kl_div_backward,
      PARAMS1(grad, self, target),
      PARAMS2(grad, self, target, reduction, log_target))

  return kl_div_backward_hpu_lazy(grad, self, target, reduction, log_target);
}
#endif

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

      Tensor grad_out, grad_beta, grad_gamma;

      std::tie(grad_out, grad_beta, grad_gamma) =
          instance_norm_backward_hpu_lazy(input, grad_in[0], mean, istd, gamma);

      // Autograds same number of gradients as the number of forward inputs and
      // in the same order
      //  grad_eps
      auto grad_eps = Tensor();
      return {grad_out, grad_gamma, grad_beta, grad_eps};
    }
  };

  return InstanceNorm::apply(input, weight, bias, eps);
}

Tensor& hpu_wrap::max_pool2d_with_indices_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices,
    Tensor& grad_input) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "max_pool2d_with_indices_backward_out :",
      " grad_output=",
      to_string(grad_output),
      " input=",
      to_string(input),
      " kernel_size=",
      to_string(kernel_size),
      " stride=",
      to_string(stride),
      " padding=",
      to_string(padding),
      " dilation=",
      to_string(dilation),
      " ceil_mode=",
      to_string(ceil_mode),
      " indices=",
      to_string(indices),
      " grad_input=",
      to_string(grad_input));
  FALLBACK_IF_UNSUPPORTED_OP_O(
      max_pool2d_with_indices_backward,
      PARAMS1(grad_input, grad_output, input, indices),
      PARAMS2(
          grad_output,
          input,
          kernel_size,
          stride,
          padding,
          dilation,
          ceil_mode,
          indices,
          grad_input),
      grad_input)

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

#if IS_PYTORCH_OLDER_THAN(1, 13)
Tensor hpu_wrap::empty(
    IntArrayRef size,
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

  return empty_hpu_lazy(size, options, optional_memory_format);
}

Tensor hpu_wrap::empty_strided(
    IntArrayRef size,
    IntArrayRef stride,
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
  return empty_strided_hpu_lazy(size, stride, options);
}

#else
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

#endif

#if IS_PYTORCH_OLDER_THAN(1, 14)
std::vector<Tensor> hpu_wrap::split_with_sizes(
    const Tensor& self,
    IntArrayRef split_sizes,
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

  return split_with_sizes_hpu_lazy(self, split_sizes, dim);
}

#else
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
#endif

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

#if IS_PYTORCH_OLDER_THAN(1, 13)
Tensor hpu_wrap::_unsafe_view(const at::Tensor& self, at::IntArrayRef size) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "_unsafe_view:", " self=", to_string(self), " size=", to_string(size));
  FALLBACK_IF_UNSUPPORTED_OP(_unsafe_view, PARAMS1(self), PARAMS2(self, size))

  return view_hpu_lazy(self, size);
}
#else
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
#endif

#if IS_PYTORCH_OLDER_THAN(1, 14)
std::vector<at::Tensor> hpu_wrap::split(
    const at::Tensor& self,
    int64_t split_size,
    int64_t dim) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "split :",
      " self=",
      to_string(self),
      " split_size=",
      to_string(split_size),
      " dim=",
      to_string(dim));
  FALLBACK_IF_UNSUPPORTED_OP_O(
      split, PARAMS1(self), PARAMS2(self, split_size, dim), Tensor)

  // lower aten::split as split_with_sizes using the logic used in Fork
  int64_t dim_size = self.size(dim);
  TORCH_CHECK(
      split_size > 0 || self.size(dim) == 0,
      "split_size can only be 0 if dimension size is 0, "
      "but got dimension size of ",
      dim_size);

  // if split_size is 0 and dimension size is 0, there is 1 split.
  int64_t num_splits = 1;
  if (split_size != 0) {
    // ensuring num_splits is at least 1 makes consistent the case where
    // split_size > dim_size (returns a single split).  We might want to
    // error here, but keep it for BC.
    num_splits = std::max<int64_t>((dim_size + split_size - 1) / split_size, 1);
  }

  std::vector<int64_t> splits(num_splits);
  int64_t last_split_size = split_size - (split_size * num_splits - dim_size);

  for (int64_t i = 0; i < num_splits; ++i) {
    auto length = i < num_splits - 1 ? split_size : last_split_size;
    splits[i] = length;
  }

  IntArrayRef split_sizes(splits);
  return hpu_wrap::split_with_sizes(self, split_sizes, dim);
}
#else
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
#endif

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
    const TensorList& gradient_vec,
    TensorList& weight_vec,
    TensorList& exp_avg_vec,
    TensorList& exp_avg_sq_vec,
    const float lr,
    at::Tensor& neg_step_t,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weight_decay) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "optimizer_adamw :",
      " gradient_vec=",
      to_string(gradient_vec),
      " weight_vec=",
      to_string(weight_vec),
      " exp_avg_vec=",
      to_string(exp_avg_vec),
      " exp_avg_sq_vec=",
      to_string(exp_avg_sq_vec),
      " lr=",
      to_string(lr),
      " neg_step_t",
      to_string(neg_step_t),
      " beta1",
      to_string(beta1),
      " beta2",
      to_string(beta2),
      " epsilon",
      to_string(epsilon),
      " weight_decay",
      to_string(weight_decay));
  TORCH_CHECK((weight_vec.size() > 0), "Can not process empty weight vector");
  auto lr_t = get_tensor_for_scalar(lr);
  optimizer_adamw_hpu_lazy(
      gradient_vec,
      weight_vec,
      exp_avg_vec,
      exp_avg_sq_vec,
      lr_t,
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
    const TensorList& model_inputs,
    TensorList& updated_ema,
    const at::Tensor& decay) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " optimizer_ema:",
      " model_inputs=",
      to_string(model_inputs),
      " updated_ema=",
      to_string(updated_ema),
      " decay=",
      to_string(decay));
  optimizer_ema_hpu_lazy(model_inputs, updated_ema, decay);
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
    const float mom,
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
  auto mom_t = get_tensor_for_scalar(mom);
  optimizer_sgd_momentum_hpu_lazy(
      gradients, weights, momentum, epoch_num, lr, mom_t, wd, damp, nesterov);
}

Tensor optimizer_lamb_fused_norm_hpu_wrap(
    const std::vector<at::Tensor>& grad,
    float max_grad_norm) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " optimizer_lamb_fused_norm:",
      " grad=",
      to_string(grad),
      "max_grad_norm=",
      to_string(max_grad_norm));
  return optimizer_lamb_fused_norm_hpu_lazy(grad, max_grad_norm);
}

std::tuple<
    std::vector<at::Tensor>,
    std::vector<at::Tensor>,
    std::vector<at::Tensor>>
optimizer_lamb_phase1_hpu_wrap(
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
    const float weight_decay) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " optimizer_lamb_phase1:",
      " gradients=",
      to_string(gradients),
      "weights=",
      to_string(weights),
      "exp_avg=",
      to_string(exp_avg),
      "exp_avg_sq=",
      to_string(exp_avg_sq),
      "clip_global_grad_norm=",
      to_string(clip_global_grad_norm),
      "grad_averaging=",
      to_string(grad_averaging),
      "lr=",
      to_string(lr),
      "beta1=",
      to_string(beta1),
      "beta2=",
      to_string(beta2),
      "epsilon=",
      to_string(epsilon),
      "step=",
      to_string(step),
      "bias_correction=",
      to_string(bias_correction),
      "weight_decay=",
      to_string(weight_decay));
  return optimizer_lamb_phase1_hpu_lazy(
      gradients,
      weights,
      exp_avg,
      exp_avg_sq,
      clip_global_grad_norm,
      grad_averaging,
      lr,
      beta1,
      beta2,
      epsilon,
      step,
      bias_correction,
      weight_decay);
}

void optimizer_lamb_phase2_hpu_wrap(
    std::vector<at::Tensor>& weight_vec,
    const std::vector<at::Tensor>& adam_norm_vec,
    const std::vector<at::Tensor>& weight_norm_vec,
    const std::vector<at::Tensor>& adam_step_vec,
    const float step,
    const float weight_decay,
    const int use_lamb) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " optimizer_lamb_phase2:",
      " weight_vec=",
      to_string(weight_vec),
      "adam_norm_vec=",
      to_string(adam_norm_vec),
      "weight_norm_vec=",
      to_string(weight_norm_vec),
      "adam_step_vec=",
      to_string(adam_step_vec),
      "step=",
      to_string(step),
      "weight_decay=",
      to_string(weight_decay),
      "use_lamb=",
      to_string(use_lamb));
  optimizer_lamb_phase2_hpu_lazy(
      weight_vec,
      adam_norm_vec,
      weight_norm_vec,
      adam_step_vec,
      step,
      weight_decay,
      use_lamb);
}

void optimizer_lars_hpu_wrap(
    const at::TensorList& params,
    at::TensorList& grads,
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

void optimizer_ResourceApplyMomentum_hpu_wrap(
    at::TensorList& params_momentum_buffer_list,
    const at::TensorList& d_p_list,
    const float momentum) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " optimizer_ResourceApplyMomentum_hpu_wrap:",
      " params_momentum_buffer_list =",
      to_string(params_momentum_buffer_list),
      " d_p_list=",
      to_string(d_p_list),
      "momentum=",
      to_string(momentum));
  return optimizer_ResourceApplyMomentum_hpu_lazy(
      params_momentum_buffer_list, d_p_list, momentum);
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

#if IS_PYTORCH_FORK_AT_LEAST(1, 0)
Tensor habana_cast_to_fp8_wrap(
    const at::Tensor& input,
    bool stochastic_rounding,
    int seed) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " habana_cast_to_fp8:",
      " input=",
      to_string(input),
      ", stochastic_rounding=",
      to_string(stochastic_rounding),
      ", seed=",
      to_string(seed));
  if (synapse_helpers::device_supports_fp8(
          synapse_helpers::HPURegistrar::get_device().type())) {
    return habana_cast_to_fp8_lazy(input, stochastic_rounding, seed);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
#endif

std::tuple<Tensor&, Tensor&> cast_to_fp8_wrap(
    const at::Tensor& input,
    const at::Tensor& scale,
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
  if (synapse_helpers::device_supports_fp8(
          synapse_helpers::HPURegistrar::get_device().type())) {
    return cast_to_fp8_lazy(input, scale, stochastic_rounding, out, amax);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
std::tuple<Tensor&, Tensor&, Tensor&> fp8_cast_transpose_wrap(
    const at::Tensor& input,
    const at::Tensor& scale,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& amax,
    at::Tensor& transposed) {
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
  if (synapse_helpers::device_supports_fp8(
          synapse_helpers::HPURegistrar::get_device().type())) {
    return fp8_cast_transpose_lazy(
        input, scale, stochastic_rounding, out, amax, transposed);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> fp8_cast_transpose_bgrad_wrap(
    const at::Tensor& input,
    const at::Tensor& scale,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& amax,
    at::Tensor& transposed,
    at::Tensor& bgrad) {
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
  if (synapse_helpers::device_supports_fp8(
          synapse_helpers::HPURegistrar::get_device().type())) {
    return fp8_cast_transpose_bgrad_lazy(
        input, scale, stochastic_rounding, out, amax, transposed, bgrad);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
std::tuple<Tensor&, Tensor&, Tensor&, Tensor&>
fp8_cast_transpose_bgrad_dgelu_wrap(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& retain,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& amax,
    at::Tensor& transposed,
    at::Tensor& bgrad) {
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
  if (synapse_helpers::device_supports_fp8(
          synapse_helpers::HPURegistrar::get_device().type())) {
    return fp8_cast_transpose_bgrad_dgelu_lazy(
        grad,
        input,
        scale,
        retain,
        stochastic_rounding,
        out,
        amax,
        transposed,
        bgrad);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
Tensor cast_from_fp8_wrap(
    const at::Tensor& input,
    const at::Tensor& scale,
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
  if (synapse_helpers::device_supports_fp8(
          synapse_helpers::HPURegistrar::get_device().type())) {
    return cast_from_fp8_lazy(input, scale, out_dtype);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
std::tuple<Tensor&, Tensor&, Tensor&> fp8_gelu_wrap(
    const at::Tensor& input,
    const at::Tensor& scale,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& amax,
    at::Tensor& retain) {
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
  if (synapse_helpers::device_supports_fp8(
          synapse_helpers::HPURegistrar::get_device().type())) {
    return fp8_gelu_lazy(input, scale, stochastic_rounding, out, amax, retain);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> fp8_layernorm_wrap(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double eps,
    const at::Tensor& scale,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& amax,
    at::Tensor& mean,
    at::Tensor& istd) {
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
  if (synapse_helpers::device_supports_fp8(
          synapse_helpers::HPURegistrar::get_device().type())) {
    return fp8_layernorm_lazy(
        input,
        weight,
        bias,
        eps,
        scale,
        stochastic_rounding,
        out,
        amax,
        mean,
        istd);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
Tensor& fp8_gemm_wrap(
    const at::Tensor& A,
    const at::Tensor& A_scale_inv,
    bool trans_A,
    const at::Tensor& B,
    const at::Tensor& B_scale_inv,
    bool trans_B,
    const at::Tensor& D,
    at::ScalarType out_dtype,
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
  if (synapse_helpers::device_supports_fp8(
          synapse_helpers::HPURegistrar::get_device().type())) {
    return fp8_gemm_lazy(
        A,
        A_scale_inv,
        trans_A,
        B,
        B_scale_inv,
        trans_B,
        D,
        out_dtype,
        bias,
        accumulate,
        out);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
at::Tensor& fp8_transpose_wrap(const at::Tensor& input, at::Tensor& out) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(" fp8_transpose:", " input=", to_string(input));
  if (synapse_helpers::device_supports_fp8(
          synapse_helpers::HPURegistrar::get_device().type())) {
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
  if (synapse_helpers::device_supports_fp8(
          synapse_helpers::HPURegistrar::get_device().type())) {
    return fp8_permute_lazy(input, dims, out);
  } else {
    TORCH_CHECK(false, "FP8 data type is not available on this device.")
  }
}
at::Tensor fp8_reshape_wrap(const at::Tensor& input, at::IntArrayRef shape) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(" fp8_reshape:", " input=", to_string(input));
  if (synapse_helpers::device_supports_fp8(
          synapse_helpers::HPURegistrar::get_device().type())) {
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

#if IS_PYTORCH_OLDER_THAN(1, 13)
Tensor hpu_wrap::slice(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<int64_t> start,
    c10::optional<int64_t> end,
    int64_t step) {
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

  return slice_hpu_lazy(self, dim, start, end, step);
}
#else
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
#endif

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
      "sum_dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
  m.def(
      "prod_dim_Int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
  m.def("diag_out(Tensor self, int diagonal, Tensor(a!) out) -> Tensor(a!)");
  m.def("randperm_out(int n, Tensor seed, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "randperm_out_ds(Tensor idst, Tensor seed, Tensor(a!) out) -> Tensor(a!)");
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
      "habanaOptimizerFusedSGDMomentum(Tensor[] gradients, Tensor(a!)[] weights_in, Tensor(b!)[] momentum_in, Tensor epoch_num, Tensor(c!) learning_rate, Tensor mom, float wd, float damp, bool nesterov) -> ()");
  m.def(
      "hpu::habanaOptimizerAdamW(Tensor[] gradient_vec, Tensor(a!)[] weight_vec, Tensor(b!)[] exp_avg_vec, Tensor(c!)[] exp_avg_sq_vec, Tensor(d!) lr_t, Tensor(e!) neg_step_t, float beta1, float beta2, float epsilon, Tensor(f!) weight_decay, bool is_wd_modified) -> ()");
  m.def(
      "hpu::habanaOptimizerFusedEMA(Tensor[] model_inputs, Tensor(a!)[] updated_ema, Tensor decay) -> ()");
  m.def(
      "fused_norm_(Tensor(a!)[] grad, Tensor max_norm, float norm_type) -> Tensor");
  m.def(
      "fused_norm_lazy(Tensor(a!)[] grad, Tensor max_norm, float norm_type) -> Tensor");
  m.def(
      "habanaOptimizerLambFusedNorm(Tensor[] grad, float max_norm, Tensor clip_norm) -> Tensor");
  m.def(
      "habanaOptimizerLambPhase1(Tensor[] grad, Tensor[] weights, Tensor[] exp_avg, Tensor[] exp_avg_sq, Tensor clip_global_grad_norm, float beta1, float beta2, float beta3, float epsilon, Tensor bias_corection1, Tensor bias_correction2, float weight_decay) -> (Tensor[], Tensor[], Tensor[])");
  m.def(
      "habanaOptimizerLambPhase2(Tensor(a!)[] weights, Tensor[] adam_norm, Tensor[] wt_norm, Tensor[] adam_step, Tensor neg_step, float wd, int use_lamb) -> ()");
  m.def(
      "habanaOptimizerLars(Tensor[] params, Tensor(a!)[] grads, Tensor lr_t, int[] skip_masks, float eeta, float weight_decay, float eps) -> ()");
  m.def(
      "habanaOptimizerResourceApplyMomentum(Tensor(a!)[] params_momentum_buf_list, Tensor[] dp_list, float momentum) -> ()");
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
      "instance_norm(Tensor input, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)");
  m.def(
      "instance_norm_backward(Tensor input, Tensor grad_in, Tensor? mean, Tensor? istd, Tensor gamma) -> (Tensor, Tensor, Tensor)");
  m.def("view(Tensor input, Tensor shape) -> Tensor");
  m.def(
      "slice(Tensor input, Tensor shape, Tensor step,  Tensor start) -> (Tensor)");
  m.def(
      "hpu::expand(Tensor(a) self, int[] sizes, *, bool implicit=False) -> Tensor(a)");
  m.def(
      "hpu::expand_ds(Tensor(a) self, Tensor shape, *, bool implicit=False) -> Tensor(a)");
  m.def("hpu::repeat(Tensor self, Tensor repeats_shape) -> Tensor");
  m.def(
      "hpu::constant_pad_nd(Tensor self, Tensor pad_before_tensor, Tensor pad_after_tensor, Scalar value) -> Tensor");
  m.def("hpu::repeat_ht(Tensor self, Tensor result_shape) -> Tensor");
  m.def(
      "hpu::constant_pad_nd_ht(Tensor self, Tensor pad_tensor, Tensor output_shape_tensor, Scalar value) -> Tensor");
  m.def(
      "hpu::scatter_nd_onnx(Tensor input, Tensor indices, Tensor values) -> Tensor");
  m.def(
      "hpu::scatter_nd(Tensor input, Tensor indices, Tensor grouped_indices, Tensor update_locations, Tensor updates) -> Tensor");
  m.def(
      "hpu::linear_bwd(Tensor grad_out, Tensor input, Tensor weight, bool bias_g=False) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::linear_ex_bwd(Tensor grad_out, Tensor input, Tensor weight, bool bias_g=False, Tensor? bias_grad_out=None) -> (Tensor, Tensor, Tensor)");
  m.def("hpu::identity(Tensor self) -> (Tensor)");
  m.def(
      "hpu::habana_cast_sr_mode(Tensor input, Scalar type, bool stochastic_rounding, int seed=0) -> (Tensor)");
  m.def(
      "hpu::cast_to_fp8(Tensor input, Tensor scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) amax) -> (Tensor(a!), Tensor(b!))");
  m.def(
      "hpu::fp8_cast_transpose(Tensor input, Tensor scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) amax, Tensor(c!) transposed) -> (Tensor(a!), Tensor(b!), Tensor(c!))");
  m.def(
      "hpu::fp8_cast_transpose_bgrad(Tensor input, Tensor scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) amax, Tensor(c!) transposed, Tensor(d!) bgrad) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))");
  m.def(
      "hpu::fp8_cast_transpose_bgrad_dgelu(Tensor grad, Tensor input, Tensor scale, Tensor? retain, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) amax, Tensor(c!) transposed, Tensor(d!) bgrad) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))");
  m.def(
      "hpu::cast_from_fp8(Tensor input, Tensor scale, ScalarType out_dtype) -> Tensor");
  m.def(
      "hpu::fp8_gelu(Tensor input, Tensor scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) amax, Tensor(c!) retain) -> (Tensor(a!), Tensor(b!), Tensor(c!))");
  m.def(
      "hpu::fp8_layernorm(Tensor input, Tensor weight, Tensor bias, float eps, Tensor scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) amax, Tensor(c!) mean, Tensor(d!) istd) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))");
  m.def(
      "hpu::fp8_gemm(Tensor A, Tensor A_scale_inv, bool trans_A, Tensor B, Tensor B_scale_inv, bool trans_B, Tensor D, ScalarType out_dtype, Tensor? bias, bool accumulate, Tensor(a!) out) -> Tensor(a!)");
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
      "hpu::ragged_softmax(Tensor self, int dim, bool half_to_float, Tensor valid_count) -> Tensor");
  m.def(
      "hpu::scaled_masked_softmax(Tensor input, Tensor mask, float scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu, HPU, m) {
  m.impl("hpu::cast_to_fp8", cast_to_fp8_wrap);
  m.impl("hpu::fp8_cast_transpose", fp8_cast_transpose_wrap);
  m.impl("hpu::fp8_cast_transpose_bgrad", fp8_cast_transpose_bgrad_wrap);
  m.impl(
      "hpu::fp8_cast_transpose_bgrad_dgelu",
      fp8_cast_transpose_bgrad_dgelu_wrap);
  m.impl("hpu::cast_from_fp8", cast_from_fp8_wrap);
  m.impl("hpu::fp8_gelu", fp8_gelu_wrap);
  m.impl("hpu::fp8_layernorm", fp8_layernorm_wrap);
  m.impl("hpu::fp8_gemm", fp8_gemm_wrap);
  m.impl("hpu::fp8_transpose", fp8_transpose_wrap);
  m.impl("hpu::ragged_softmax", _ragged_softmax_wrap);
  m.impl("hpu::scaled_masked_softmax", scaled_masked_softmax_wrap);
  m.impl("hpu::fp8_reshape", fp8_reshape_wrap);
  m.impl("hpu::fp8_permute", fp8_permute_wrap);
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
