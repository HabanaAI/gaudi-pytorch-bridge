/**
 * Copyright (c) 2021-2024 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <ATen/core/Tensor.h>
#include <torch/library.h>
#include "backend/helpers/habana_types.h"
#include "backend/synapse_helpers/device_helpers.h"
#include "common/dump_args.h"
#include "common/random_utils.h"
#include "generated/lazy/wrap_kernels_declarations.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/instance_norm_utils.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/lazy_optimizer_kernels.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_lazy/hpu_stage_submission.h"
#include "hpu_ops/cpu_fallback.h"
#include "hpu_ops/op_logger.h"
#include "hpu_ops/op_validator.h"
#include "hpu_ops/run_maybe_with_acc_thread.h"
#include "hpu_ops/shared_meta_common.h"
#include "kernel_input_checks.h"
#include "pytorch_helpers/habana_helpers/kernels_accumulation.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

using namespace torch;
using namespace at;
using namespace habana;
using namespace habana_lazy;

#define FP8_CHECK                                 \
  TORCH_CHECK(                                    \
      synapse_helpers::device_supports_fp8(       \
          HPUDeviceContext::get_device().type()), \
      "FP8 data type is not available on this device.")

namespace habana {
static CheckNodeWithSharedLayerValidator validator_matmul(
    "matmul",
    MatmulSharedMeta,
    habana_helpers::HabanaExecutionMode::LAZY);

static CheckNodeWithSharedLayerValidator validator__reshape_alias(
    "_reshape_alias",
    StridedViewSharedMeta,
    habana_helpers::HabanaExecutionMode::LAZY);

static CheckNodeWithSharedLayerValidator validator__unsafe_view(
    "_unsafe_view",
    StridedViewSharedMeta,
    habana_helpers::HabanaExecutionMode::LAZY);

static CheckNodeWithSharedLayerValidator validator_instance_norm(
    "instance_norm",
    InstanceNormSharedMeta,
    habana_helpers::HabanaExecutionMode::LAZY);
} // namespace habana

bool hpu_wrap::is_pinned(
    const at::Tensor& self,
    c10::optional<at::Device> device) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "is_pinned :", " self=", to_string(self), " device=", to_string(device));
  if (!device.has_value()) {
    return false;
  }
  return is_pinned_hpu(self, *device);
}

Tensor hpu_wrap::pin_memory(
    const at::Tensor& self,
    ::std::optional<at::Device> device) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "pin_memory :", " self=", to_string(self), " device=", to_string(device));
  HABANA_ASSERT(device.has_value(), "Unable to pin memory to an null device");
  return pin_memory_hpu(self, *device);
}

Tensor hpu_wrap::_pin_memory(
    const at::Tensor& self,
    ::std::optional<at::Device> device) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "_pin_memory :",
      " self=",
      to_string(self),
      " device=",
      to_string(device));
  HABANA_ASSERT(device.has_value(), "Unable to pin memory to an null device");
  return pin_memory_hpu(self, *device);
}

Tensor hpu_wrap::bincount(
    const Tensor& self,
    const c10::optional<Tensor>& weights,
    int64_t minlength) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
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
    return bincount_hpu_lazy(self, weights, minlength);
  }
  return dispatch_fallback<ATEN_OP(bincount)>::call(
      OpSupportLevel::Value::unsupported_dtype,
      PARAMS2(self, weights, minlength));
}

Tensor& hpu_wrap::copy_(Tensor& self, const Tensor& src, bool non_blocking) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "copy_ :",
      " self=",
      to_string(self),
      " src=",
      to_string(src),
      " non_blocking=",
      to_string(non_blocking));
  TORCH_CHECK(
      self.dim() <= 8 && src.dim() <= 8, "HPU doesn't support rank > 8D");
  return copy_hpu_lazy_(self, src, non_blocking);
}

Tensor hpu_wrap::_reshape_alias(
    const Tensor& self,
    SymIntArrayRef size,
    SymIntArrayRef stride) {
  PT_LAZY_OP_TRACE;
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
  [[maybe_unused]] bool require_h2d = false;
  [[maybe_unused]] bool require_st = false;
  VAL_CUSTOM_FALLBACK_IF_UNSUPPORTED_DTYPE(
      _reshape_alias, false, self, size, stride)
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
  PT_LAZY_OP_TRACE;
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
  PT_LAZY_OP_TRACE;
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
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "masked_select :", " self=", to_string(self), " mask=", to_string(mask));
  FALLBACK_IF_UNSUPPORTED_OP(
      masked_select, PARAMS1(self, mask), PARAMS2(self, mask))
  if (self.dim() > 5) {
    return dispatch_fallback<ATEN_OP(masked_select)>::call(
        OpSupportLevel::Value::unsupported_rank, PARAMS2(self, mask));
  }
  return masked_select_hpu_lazy(self, mask);
}

Tensor& hpu_wrap::masked_select_out(
    const Tensor& self,
    const Tensor& mask,
    Tensor& out) {
  PT_LAZY_OP_TRACE;
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
  if (self.dim() > 5) {
    return dispatch_fallback<ATEN_OP2(masked_select, out)>::call(
        OpSupportLevel::Value::unsupported_rank, PARAMS2(self, mask, out));
  }
  return masked_select_out_hpu_lazy(self, mask, out);
}

Tensor& hpu_wrap::scatter_add_(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  PT_LAZY_OP_TRACE;
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
  if (self.dim() > 5 || index.dim() > 5 || src.dim() > 5) {
    return dispatch_fallback<ATEN_OP(scatter_add_)>::call(
        OpSupportLevel::Value::unsupported_rank,
        PARAMS2(self, dim_, index, src));
  }
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
      (self.scalar_type() != c10::ScalarType::Float8_e5m2) &&
      (self.scalar_type() != c10::ScalarType::Float8_e4m3fn) &&
      !(self.scalar_type() == c10::ScalarType::Half &&
        HPUDeviceContext::get_device().type() !=
            synDeviceType::synDeviceGaudi)) {
    return dispatch_fallback<ATEN_OP(_index_put_impl_)>::call(
        OpSupportLevel::Value::unsupported_dtype,
        PARAMS2(self, indices, values, accumulate, unsafe));
  }
  if (self.dim() > 5) {
    return dispatch_fallback<ATEN_OP(_index_put_impl_)>::call(
        OpSupportLevel::Value::unsupported_rank,
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
        habana::HPUDeviceContext::get_device().type() !=
            synDeviceType::synDeviceGaudi &&
        self.dim() >
            4)) { // self.dim()<=4 goes through cguid that doesn't support fp16
    return dispatch_fallback<ATEN_OP(nonzero)>::call(
        OpSupportLevel::Value::unsupported_dtype, PARAMS2(self));
  }
  if (self.dim() > 5) {
    return dispatch_fallback<ATEN_OP(nonzero)>::call(
        OpSupportLevel::Value::unsupported_rank, PARAMS2(self));
  }
  return nonzero_hpu_lazy(self);
}

::std::tuple<at::Tensor, at::Tensor> hpu_wrap::_unique(
    const at::Tensor& self,
    bool sorted,
    bool return_inverse) {
  FALLBACK_IF_UNSUPPORTED_OP(
      _unique, PARAMS1(self), PARAMS2(self, sorted, return_inverse))
  if (self.dim() > 4) {
    return dispatch_fallback<ATEN_OP(_unique)>::call(
        OpSupportLevel::Value::unsupported_rank,
        PARAMS2(self, sorted, return_inverse));
  }
  return _unique_hpu_lazy(self, sorted, return_inverse);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> hpu_wrap::_unique2(
    const at::Tensor& self,
    bool sorted,
    bool return_inverse,
    bool return_counts) {
  FALLBACK_IF_UNSUPPORTED_OP(
      _unique2,
      PARAMS1(self),
      PARAMS2(self, sorted, return_inverse, return_counts))
  if (self.dim() > 4) {
    return dispatch_fallback<ATEN_OP(_unique2)>::call(
        OpSupportLevel::Value::unsupported_rank,
        PARAMS2(self, sorted, return_inverse, return_counts));
  }
  return unique2_hpu_lazy(self, sorted, return_inverse, return_counts);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> hpu_wrap::unique_dim(
    const at::Tensor& self,
    int64_t dim,
    bool sorted,
    bool return_inverse,
    bool return_counts) {
  return unique_dim_hpu_lazy(self, dim, sorted, return_inverse, return_counts);
}

Tensor& hpu_wrap::index_add_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    const Scalar& alpha,
    Tensor& out) {
  PT_LAZY_OP_TRACE;
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
  if (self.dim() > 5 || index.dim() > 5 || source.dim() > 5) {
    return dispatch_fallback<ATEN_OP2(index_add, out)>::call(
        OpSupportLevel::Value::unsupported_rank,
        PARAMS2(self, dim, index, source, alpha, out));
  }
  return index_add_hpu_lazy_out(self, dim, index, source, alpha, out);
}

Tensor& hpu_wrap::nonzero_out(const Tensor& self, Tensor& out) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "nonzero_out :", " self=", to_string(self), " out=", to_string(out));
  FALLBACK_IF_UNSUPPORTED_OP(
      nonzero_out, PARAMS1(self, out), PARAMS2(self, out))
  if (self.dim() > 5) {
    return dispatch_fallback<ATEN_OP2(nonzero, out)>::call(
        OpSupportLevel::Value::unsupported_rank, PARAMS2(self, out));
  }
  return nonzero_out_hpu_lazy(self, out);
}

::std::tuple<at::Tensor, at::Tensor> hpu_wrap::batch_norm_stats(
    const at::Tensor& input,
    double eps) {
  return batch_norm_stats_lazy(input, eps);
}
at::Tensor hpu_wrap::batch_norm_elemt(
    const at::Tensor& input,
    const ::std::optional<at::Tensor>& weight,
    const ::std::optional<at::Tensor>& bias,
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
    const ::std::optional<at::Tensor>& weight,
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
        const ::std::optional<at::Tensor>& weight,
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
        const ::std::optional<at::Tensor>& running_mean,
        const ::std::optional<at::Tensor>& running_var,
        double momentum,
        double eps,
        const at::Tensor& counts) {
  return batch_norm_gather_stats_with_counts_lazy(
      input, mean, invstd, running_mean, running_var, momentum, eps, counts);
}

struct InstanceNormBackward
    : public torch::autograd::Function<InstanceNormBackward> {
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& input,
      const Tensor& weight, // gamma
      const Tensor& grad_in,
      const Tensor& mean, // save_mean
      const Tensor& istd, // save_invstd
      const double eps) {
    auto input_maybe_reshaped = input;
    auto grad_in_maybe_reshaped = grad_in;
    const auto is_3d = input.dim() == 3;
    if (is_3d) {
      auto new_shape = input.sizes().vec();
      new_shape.push_back(1);
      input_maybe_reshaped = at::reshape(input, new_shape);
      grad_in_maybe_reshaped = at::reshape(grad_in, new_shape);
    }
    const auto [grad_out_maybe_reshaped, grad_beta, grad_gamma] =
        instance_norm_backward_hpu_lazy(
            input_maybe_reshaped, grad_in_maybe_reshaped, mean, istd, weight);

    const Tensor grad_out = is_3d
        ? at::reshape(grad_out_maybe_reshaped, input.sizes())
        : grad_out_maybe_reshaped;
    ctx->save_for_backward({input, weight, grad_in, mean, istd});
    ctx->saved_data["eps"] = eps;
    return {grad_out, grad_gamma, grad_beta};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_in) {
    const auto saved = ctx->get_saved_variables();
    auto input = saved[0]; // Input always as [N, C, X1, ... ,Xn]
    auto weight = saved[1];
    auto gO = saved[2];
    auto save_mean = saved[3];
    auto save_invstd = saved[4];
    const double eps = ctx->saved_data["eps"].toDouble();
    const std::array<bool, 3> mask{true, true, true};

    const auto input_shape = input.sizes();
    const auto dim0 = input_shape[0];
    const auto dim1 = input_shape[1];
    const auto is_3d = input.dim() == 3;
    const auto is_5d = input.dim() == 5;

    auto input_batch_norm_shape = is_3d
        ? std::vector<
              int64_t>{1, input_shape[0] * input_shape[1], input_shape[2], 1}
        : std::vector<int64_t>{
              1,
              input_shape[0] * input_shape[1],
              input_shape[2],
              input_shape[3]};
    if (is_5d) {
      input_batch_norm_shape.push_back(input_shape[4]);
    }

    const auto input_reshaped = input.reshape(input_batch_norm_shape);
    const auto weight_reshaped = weight.repeat(dim0);
    const auto gO_reshaped = gO.reshape(input_batch_norm_shape);

    const auto save_mean_reshaped = save_mean.reshape(-1);
    const auto save_invstd_reshaped = save_invstd.reshape(-1);

    const auto ggI_reshaped = grad_in[0].reshape(input_batch_norm_shape);
    const auto ggG_reshaped = grad_in[1].repeat(dim0);
    const auto ggB_reshaped = grad_in[2].repeat(dim0);

    const auto [gI, gG, ggP] = batchnorm_double_backward(
        input_reshaped,
        weight_reshaped,
        ggI_reshaped,
        ggG_reshaped,
        ggB_reshaped,
        gO_reshaped,
        std::nullopt, /*running_mean*/
        std::nullopt, /*running_var*/
        true, /*train*/
        eps,
        save_mean_reshaped,
        save_invstd_reshaped,
        mask);

    const auto index =
        torch::arange(dim1, torch::TensorOptions().device(torch::kHPU));

    return {
        gI.reshape(input_shape),
        gG.index_select(0, index),
        ggP.reshape(input_shape),
        at::Tensor(),
        at::Tensor(),
        at::Tensor()};
  }
};

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
    auto [output_maybe_reshaped, mean, istd] =
        instance_norm_hpu_lazy(input_maybe_reshaped, weight, bias, eps);
    const Tensor output = is_3d
        ? at::reshape(output_maybe_reshaped, input.sizes())
        : output_maybe_reshaped;
    ctx->save_for_backward({input, mean, istd, weight});
    ctx->saved_data["eps"] = eps;
    return output;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_in) {
    const auto eps = ctx->saved_data["eps"].toDouble();
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto mean = saved[1];
    auto istd = saved[2];
    auto weight = saved[3];
    const auto res =
        InstanceNormBackward::apply(input, weight, grad_in[0], mean, istd, eps);

    return {res[0], res[1], res[2], at::Tensor()};
  }
};

Tensor hpu_wrap::instance_norm(
    const Tensor& input,
    const ::std::optional<Tensor>& weight_opt,
    const ::std::optional<Tensor>& bias_opt,
    const ::std::optional<Tensor>& running_mean_opt,
    const ::std::optional<Tensor>& running_var_opt,
    [[maybe_unused]] bool use_input_stats,
    [[maybe_unused]] double momentum,
    double eps,
    [[maybe_unused]] bool cudnn_enabled) {
  PT_LAZY_OP_TRACE;
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
  auto weight = weight_opt.has_value()
      ? weight_opt.value()
      : at::ones(
            input.sizes().vec()[1],
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kHPU));
  auto bias = bias_opt.has_value()
      ? bias_opt.value()
      : at::zeros(
            input.sizes().vec()[1],
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kHPU));

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
  [[maybe_unused]] bool require_h2d = false;
  [[maybe_unused]] bool require_st = false;
  VAL_CUSTOM_FALLBACK_IF_UNSUPPORTED_DTYPE(
      instance_norm,
      false,
      input,
      weight,
      bias,
      running_mean,
      running_var,
      use_input_stats,
      momentum,
      eps,
      cudnn_enabled)
  return InstanceNorm::apply(input, weight, bias, eps);
}

at::Tensor hpu_wrap::repeat_interleave(
    const at::Tensor& repeats,
    c10::optional<c10::SymInt> output_size) {
  habana_lazy::NoAccThread no_acc_thread;
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "repeat_interleave:",
      " repeats=",
      to_string(repeats),
      " output_size=",
      to_string(output_size));
  FALLBACK_IF_UNSUPPORTED_OP_O(
      repeat_interleave,
      PARAMS1(repeats),
      PARAMS2(repeats, output_size),
      Tensor)
  std::optional<int64_t> out_size;
  if (output_size.has_value()) {
    out_size = output_size.value().expect_int();
  }
  return repeat_inlv_hpu_lazy(repeats, out_size);
}

struct SoftmaxFunction : public torch::autograd::Function<SoftmaxFunction> {
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor input,
      int64_t dim,
      c10::optional<at::ScalarType> dtype) {
    bool need_fp8_to_fp32_cast =
        input.scalar_type() == at::ScalarType::Float8_e5m2 ||
        input.scalar_type() == at::ScalarType::Float8_e4m3fn;
    Tensor converted = dtype.has_value() ? input.toType(dtype.value())
        : need_fp8_to_fp32_cast          ? input.toType(at::ScalarType::Float)
                                         : input;
    auto result = torch::_softmax(converted, dim, false);
    ctx->save_for_backward({result, input});
    ctx->saved_data["dim"] = dim;
    return need_fp8_to_fp32_cast ? result.toType(input.scalar_type()) : result;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    torch::autograd::variable_list saved_vars = ctx->get_saved_variables();
    auto output = saved_vars[0];
    auto input = saved_vars[1];
    auto dim = ctx->saved_data["dim"].toInt();
    bool need_fp8_to_fp32_cast =
        input.scalar_type() == at::ScalarType::Float8_e5m2 ||
        input.scalar_type() == at::ScalarType::Float8_e4m3fn;
    auto result = torch::_softmax_backward_data(
        need_fp8_to_fp32_cast ? grad_output[0].toType(at::ScalarType::Float)
                              : grad_output[0],
        output,
        dim,
        input.scalar_type());
    return {
        need_fp8_to_fp32_cast ? result.toType(input.scalar_type()) : result,
        torch::Tensor(),
        torch::Tensor()};
  }
};

Tensor hpu_wrap::softmax(
    const Tensor& self,
    int64_t dim,
    ::std::optional<at::ScalarType> dtype) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "softmax :",
      " self=",
      to_string(self),
      " dim=",
      to_string(dim),
      " dtype=",
      to_string(dtype));
  const auto selfDtype = self.scalar_type();
  auto computeDtype = dtype.has_value() ? dtype.value() : selfDtype;
  const bool needFp8ToFp32Cast = selfDtype == at::ScalarType::Float8_e5m2 ||
      selfDtype == at::ScalarType::Float8_e4m3fn;
  if (needFp8ToFp32Cast)
    computeDtype = at::ScalarType::Float;

  if (computeDtype != at::ScalarType::Float &&
      computeDtype != at::ScalarType::BFloat16 &&
      !((computeDtype == at::ScalarType::Half ||
         computeDtype == at::ScalarType::Float8_e5m2 ||
         computeDtype == at::ScalarType::Float8_e4m3fn) &&
        habana::HPUDeviceContext::get_device().type() !=
            synDeviceType::synDeviceGaudi)) {
    return dispatch_fallback<ATEN_OP2(softmax, int)>::call(
        OpSupportLevel::Value::unsupported_dtype, PARAMS2(self, dim, dtype));
  }
  if (self.dim() > 5) {
    return dispatch_fallback<ATEN_OP2(softmax, int)>::call(
        OpSupportLevel::Value::unsupported_rank, PARAMS2(self, dim, dtype));
  }
  return SoftmaxFunction::apply(self, dim, dtype);
}

Tensor hpu_wrap::empty(
    SymIntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<MemoryFormat> optional_memory_format) {
  PT_LAZY_OP_TRACE;
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
  PT_LAZY_OP_TRACE;
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
  PT_LAZY_OP_TRACE;
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
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "sort :",
      " self=",
      to_string(self),
      " dim =",
      to_string(dim),
      " descending=",
      to_string(descending));
  if (self.dim() > 5) {
    return dispatch_fallback<ATEN_OP(sort)>::call(
        OpSupportLevel::Value::unsupported_rank,
        PARAMS2(self, dim, descending));
  }
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
  PT_LAZY_OP_TRACE;
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
  PT_LAZY_OP_TRACE;
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
  PT_LAZY_OP_TRACE;
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
  PT_LAZY_OP_TRACE;
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
    const double weight_decay,
    c10::optional<at::TensorList> exp_avg_scales,
    c10::optional<at::TensorList> exp_avg_sq_scales) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "optimizer_adamw :",
      DUMP_11ARGS(
          gradient_vec,
          weight_vec,
          exp_avg_vec,
          exp_avg_sq_vec,
          neg_step_t,
          beta1,
          beta2,
          epsilon,
          weight_decay,
          exp_avg_scales,
          exp_avg_sq_scales));

  TORCH_CHECK(
      (weight_vec.size() > 0),
      "optimizer_adamw : can not process empty weight vector");
  TORCH_CHECK(
      exp_avg_scales.has_value() == exp_avg_sq_scales.has_value(),
      "optimizer_adamw : expects both or neighter scales to be set");

  optimizer_adamw_hpu_lazy(
      gradient_vec,
      weight_vec,
      exp_avg_vec,
      exp_avg_sq_vec,
      neg_step_t,
      beta1,
      beta2,
      epsilon,
      weight_decay,
      exp_avg_scales,
      exp_avg_sq_scales);
}

Tensor fused_norm_hpu_wrap(
    std::vector<at::Tensor>& grad,
    const Tensor& max_norm,
    float norm_type) {
  PT_LAZY_OP_TRACE;
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
  PT_LAZY_OP_TRACE;
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
  PT_LAZY_OP_TRACE;
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
    const at::TensorList gradients,
    at::TensorList weights,
    at::Tensor& lr,
    double wd,
    double mom,
    double damp,
    bool nesterov) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  optimizer_sgd_hpu_lazy(gradients, weights, lr, wd, mom, damp, nesterov);
}

void optimizer_sgd_momentum_hpu_wrap(
    const at::TensorList gradients,
    at::TensorList weights,
    at::TensorList momentum,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const at::Tensor& mom,
    double wd,
    double damp,
    bool nesterov) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
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
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "optimizer_lars_hpu_wrap:",
      " param=",
      to_string(params),
      " grad=",
      to_string(grads),
      " skipMasks=",
      to_string(skipMasks),
      " eeta=",
      to_string(eeta),
      " weight_decay=",
      to_string(weight_decay),
      " eps=",
      to_string(eps),
      " lr=",
      to_string(lr));
  return optimizer_lars_hpu_lazy(
      params, grads, skipMasks, eeta, weight_decay, eps, lr);
}

void optimizer_resource_apply_momentum_hpu_wrap(
    at::TensorList params_momentum_buf_list,
    const at::TensorList dp_list,
    const double momentum) {
  PT_LAZY_OP_TRACE;
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
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " torchvision_nms:",
      " boxes=",
      to_string(boxes),
      " scores=",
      to_string(scores),
      " iou_threshold=",
      to_string(iou_threshold));

  return habana_nms_hpu_lazy(boxes, scores, iou_threshold);
}

Tensor batched_nms_hpu_wrap(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    const at::Tensor& indices,
    float iou_threshold) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "batched_nms:",
      " boxes=",
      to_string(boxes),
      " scores=",
      to_string(scores),
      " indices=",
      to_string(indices),
      " iou_threshold=",
      to_string(iou_threshold));
  return batched_nms_hpu_lazy(boxes, scores, indices, iou_threshold);
}

at::Tensor matmul_ex_wrap(
    const at::Tensor& self,
    const at::Tensor& other,
    at::ScalarType dtype) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  return matmul_hpu_lazy(self, other, dtype);
}

std::tuple<at::Tensor, at::Tensor> matmul_ex_backward_wrap(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& other,
    at::ScalarType dtype) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  return matmul_backward_hpu_lazy(grad_output, self, other, dtype);
}

Tensor habana_random_seed_wrap(const at::Tensor& input) {
  PT_LAZY_OP_TRACE;
  PT_OP_INFO(" habana_random_seed:", " input=", to_string(input));
  return habana_random_seed_lazy(input);
}

std::vector<at::Tensor> habana_permute_1D_sparse_data_wrap(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& weights) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "permute_1D_sparse_data:",
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
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "permute_2D_sparse_data:",
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
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "expand_into_jagged_permute:",
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
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "split_permute_cat:",
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

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>
habana_bounds_check_indices_wrap(
    at::Tensor& indices,
    at::Tensor& offsets,
    at::Tensor& warning,
    const at::Tensor& rows_per_table,
    int64_t bounds_check_mode,
    const c10::optional<at::Tensor>& weights) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "bounds_check_indices:",
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_fwd_wrap(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    c10::string_view softmax_mode,
    const c10::optional<at::Tensor>& valid_seq_len,
    c10::string_view seq_padding_type) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "sdpa_fwd :",
      DUMP_10ARGS(
          q,
          k,
          v,
          attention_mask,
          p,
          scale,
          is_causal,
          softmax_mode,
          valid_seq_len,
          seq_padding_type));

  return sdpa_fwd_lazy(
      q,
      k,
      v,
      attention_mask,
      p,
      scale,
      is_causal,
      softmax_mode,
      valid_seq_len,
      seq_padding_type);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fp8_sdpa_fwd_wrap(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    c10::string_view softmax_mode,
    const c10::optional<at::Tensor>& d_scale_q,
    const c10::optional<at::Tensor>& d_scale_k,
    const c10::optional<at::Tensor>& d_scale_v,
    const c10::optional<at::Tensor>& q_scale_s,
    const c10::optional<at::Tensor>& q_scale_o,
    const c10::optional<at::Tensor>& d_scale_s,
    const bool is_amax_s,
    const c10::optional<at::Tensor>& valid_seq_len,
    c10::string_view seq_padding_type) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "fp8_sdpa_fwd :",
      DUMP_17ARGS(
          q,
          k,
          v,
          attention_mask,
          p,
          scale,
          is_causal,
          softmax_mode,
          d_scale_q,
          d_scale_k,
          d_scale_v,
          q_scale_s,
          q_scale_o,
          d_scale_s,
          is_amax_s,
          valid_seq_len,
          seq_padding_type));

  return fp8_sdpa_fwd_lazy(
      q,
      k,
      v,
      attention_mask,
      p,
      scale,
      is_causal,
      softmax_mode,
      d_scale_q,
      d_scale_k,
      d_scale_v,
      q_scale_s,
      q_scale_o,
      d_scale_s,
      is_amax_s,
      valid_seq_len,
      seq_padding_type);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> sdpa_recomp_fwd_wrap(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    const bool requires_backward,
    c10::string_view softmax_mode,
    const c10::optional<at::Tensor>& valid_seq_len,
    c10::string_view seq_padding_type) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "sdpa_recomp_fwd :",
      DUMP_11ARGS(
          q,
          k,
          v,
          attention_mask,
          p,
          scale,
          is_causal,
          requires_backward,
          softmax_mode,
          valid_seq_len,
          seq_padding_type));

  return sdpa_recomp_fwd_lazy(
      q,
      k,
      v,
      attention_mask,
      p,
      scale,
      is_causal,
      requires_backward,
      softmax_mode,
      valid_seq_len,
      seq_padding_type);
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
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO("matmul:", " self=", to_string(self), "other=", to_string(other));
  [[maybe_unused]] bool require_h2d = false;
  [[maybe_unused]] bool require_st = false;
  VAL_CUSTOM_FALLBACK_IF_UNSUPPORTED_DTYPE(matmul, false, self, other)
  return MatmulFunction::apply(self, other);
}

Tensor matmul_inference(const Tensor& self, const Tensor& other) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO("matmul:", " self=", to_string(self), "other=", to_string(other));
  [[maybe_unused]] bool require_h2d = false;
  [[maybe_unused]] bool require_st = false;
  VAL_CUSTOM_FALLBACK_IF_UNSUPPORTED_DTYPE(matmul, false, self, other)
  return matmul_hpu_lazy(self, other);
}

Tensor hpu_wrap::slice(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<c10::SymInt> start,
    c10::optional<c10::SymInt> end,
    c10::SymInt step) {
  auto temp_start = start.has_value() ? start.value().expect_int() : 0;
  auto temp_end = end.has_value() ? end.value().expect_int() : INT64_MAX;
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "slice:",
      " self=",
      to_string(self),
      " dim=",
      to_string(dim),
      " start=",
      to_string(start),
      " end=",
      to_string(end),
      " step=",
      to_string(step));

  return slice_hpu_lazy(self, dim, temp_start, temp_end, step.expect_int());
}

struct DropoutFunction : public Function<DropoutFunction> {
  static at::Tensor forward(
      AutogradContext* ctx,
      at::Tensor input,
      double p,
      bool train) {
    ctx->saved_data["p"] = train ? p : 0.0;
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
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "dropout:",
      " input=",
      to_string(input),
      " p=",
      to_string(p),
      " train=",
      to_string(train));
  FALLBACK_IF_UNSUPPORTED_OP(dropout, PARAMS1(input), PARAMS2(input, p, train))
  if (input.dim() > 5) {
    return dispatch_fallback<ATEN_OP(dropout)>::call(
        OpSupportLevel::Value::unsupported_rank, PARAMS2(input, p, train));
  }
  return DropoutFunction::apply(input, p, train);
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
  m.def(
      "index_put_normal_and_neg_indices(Tensor self, Tensor[] indices, Tensor value, bool accumulate=False) -> Tensor");
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
      "hpu::expand(Tensor(a) self, int[] sizes, *, bool implicit=False) -> Tensor(a)");
  m.def(
      "hpu::expand_ds(Tensor(a) self, Tensor shape, *, bool implicit=False) -> Tensor(a)");
  m.def(
      "embedding_bag_sum_bwd_out(Tensor(a!) out, Tensor input, Tensor indices_bwd, Tensor offsets_bwd, Tensor valid_count_bwd, int kernel_mode) -> Tensor(a!)");
  m.def(
      "habanaOptimizerFusedAdagrad(Tensor[] gradients, Tensor(a!)[] weights_in, Tensor(b!)[] variances_in, Tensor epoch_num, Tensor(c!) learning_rate, float wd, float lrd, float eps) -> ()");
  m.def(
      "hpu::habanaOptimizerSgd(Tensor[] gradients, Tensor(a!)[] weights_in, Tensor(b!) learning_rate, float wd, float mom, float damp, bool nesterov) -> ()");
  m.def(
      "hpu::habanaOptimizerSgdMomentum(Tensor[] gradients, Tensor(a!)[] weights_in, Tensor(b!)[] momentum_in, Tensor epoch_num, Tensor(c!) learning_rate, Tensor(d!) mom, float wd, float damp, bool nesterov) -> ()");
  m.def(
      "hpu::habanaOptimizerAdamW(Tensor[] gradient_vec, Tensor(a!)[] weight_vec, Tensor(b!)[] exp_avg_vec, Tensor(c!)[] exp_avg_sq_vec, Tensor neg_step_t, float beta1, float beta2, float epsilon, Tensor weight_decay, bool has_weight_decay) -> ()");
  m.def(
      "hpu::optimizer_adamw(Tensor[] gradient_vec, Tensor(a!)[] weight_vec, Tensor(b!)[] exp_avg_vec, Tensor(c!)[] exp_avg_sq_vec, Tensor neg_step_t, float beta1, float beta2, float epsilon, float weight_decay, Tensor(d!)[]? exp_avg_scales = None, Tensor(e!)[]? exp_avg_sq_scales = None) -> ()");
  m.def(
      "hpu::habanaOptimizerFusedEMA(Tensor[] model_inputs, Tensor(a!)[] updated_ema, Tensor decay) -> ()");
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
      "gather_elements(Tensor self, Tensor index, Tensor? opt, int dim_, bool sorted) -> Tensor");
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
  m.def(
      "hpu::select_scatter(Tensor self, Tensor src, Tensor dim, Tensor index) -> (Tensor)");
  m.def(
      "hpu::slice_scatter(Tensor self, Tensor src, Tensor dim = None, Tensor? start = None, Tensor? end = None, Tensor step = None) -> (Tensor)");
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
  m.def("hpu::repeat_ht(Tensor self, Tensor result_shape) -> Tensor");
  m.def(
      "hpu::constant_pad_nd_ht(Tensor self, Tensor pad_tensor, Tensor output_shape_tensor, Scalar value) -> Tensor");
  m.def(
      "hpu::constant_pad_nd_lazy(Tensor self, SymInt[] pad_array, Scalar value) -> Tensor");
  m.def(
      "hpu::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor");
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
  m.def("hpu::identity(Tensor self) -> (Tensor)");
  m.def(
      "hpu::habana_cast_sr_mode(Tensor input, Scalar type, bool stochastic_rounding, int seed=0) -> (Tensor)");
  m.def(
      "hpu::convert_from_int4(Tensor input, Tensor scale, Tensor? zero_point, ScalarType out_dtype) -> Tensor");
  m.def(
      "hpu::convert_from_uint4(Tensor input, Tensor scale, Tensor? zero_point, ScalarType out_dtype) -> Tensor");
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
      "hpu::mixture_of_experts(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w1, Tensor[] w2, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max) -> Tensor");
  m.def(
      "hpu::mixture_of_experts.fused_weights(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w12, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max) -> Tensor");
  m.def(
      "hpu::mixture_of_experts.fp8_measurement(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w1, Tensor[] w2, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max, bool measurement_mode) -> (Tensor, Tensor)");
  m.def(
      "hpu::mixture_of_experts.fp8_measurement_fused_weights(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w12, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max, bool measurement_mode) -> (Tensor, Tensor)");
  m.def(
      "hpu::mixture_of_experts.fp8(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w1, Tensor[] w2, Tensor[] w3, Tensor d_scale_hidden_states, Tensor[] d_scale_intermediate_hidden_states, Tensor[] d_scale_w1, Tensor[] d_scale_w2, Tensor[] d_scale_w3, bool permuted_weights, str activation, int experts_min, int experts_max) -> Tensor");
  m.def(
      "hpu::mixture_of_experts.fp8_fused_weights(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w12, Tensor[] w3, Tensor d_scale_hidden_states, Tensor[] d_scale_intermediate_hidden_states, Tensor[] d_scale_w12, Tensor[] d_scale_w3, bool permuted_weights, str activation, int experts_min, int experts_max) -> Tensor");
  m.def(
      "hpu::mixture_of_experts.fp8_scalars(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w1, Tensor[] w2, Tensor[] w3, float d_scale_hidden_states, float[] d_scale_intermediate_hidden_states, float[] d_scale_w1, float[] d_scale_w2, float[] d_scale_w3, bool permuted_weights, str activation, int experts_min, int experts_max) -> Tensor");
  m.def(
      "hpu::mixture_of_experts.fp8_fused_weights_scalars(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w12, Tensor[] w3, float d_scale_hidden_states, float[] d_scale_intermediate_hidden_states, float[] d_scale_w12, float[] d_scale_w3, bool permuted_weights, str activation, int experts_min, int experts_max) -> Tensor");
  m.def(
      "hpu::habana_split_permute_cat(Tensor input, Tensor indices, int batch_size, int num_features, int dims) -> Tensor");
  m.def(
      "hpu::habana_bounds_check_indices(Tensor(a!) indices, Tensor(b!) offsets, Tensor(c!) warning, Tensor rows_per_table, int bounds_check_mode, Tensor? weights) -> (Tensor(a!), Tensor(b!), Tensor(c!))");

  // Seed is generated at FE and passed to BE. There is no seed at python
  // interface. So the schema with python interface and BE differ. Register the
  // op at python interface directly with the wrapper function to let Pytorch
  // Infer the schema at operator level. So no m.impl() def is needed for this
  // operator. For BE, Register a schema with seed.
  m.def(
      "hpu::sdpa_fwd(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, str softmax_mode, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::sdpa_fwd_dropout(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, str softmax_mode, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_fwd(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? q_scale_s, Tensor? q_scale_o, Tensor? d_scale_s, bool is_amax_s, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_fwd_non_dropout(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? q_scale_s, Tensor? q_scale_o, Tensor? d_scale_s, bool is_amax_s, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_fwd_dropout_seed(Tensor seed, Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? q_scale_s, Tensor? q_scale_o, Tensor? d_scale_s, bool is_amax_s, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::sdpa_fwd_non_dropout(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, str softmax_mode, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::sdpa_fwd_dropout_seed(Tensor seed, Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, str softmax_mode, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::sdpa_recomp_fwd(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::sdpa_recomp_fwd_dropout(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor, Tensor)");
  // for torch.compile to insert seed tensor
  m.def(
      "hpu::sdpa_recomp_fwd_non_dropout(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::sdpa_recomp_fwd_dropout_seed(Tensor seed, Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? q_scale_s, Tensor? q_scale_o, Tensor? d_scale_s, bool is_amax_s, bool is_amax_0, Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd_non_dropout(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? q_scale_s, Tensor? q_scale_o, Tensor? d_scale_s, bool is_amax_s, bool is_amax_0,  Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd_dropout(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? q_scale_s, Tensor? q_scale_o, Tensor? d_scale_s, bool is_amax_s, bool is_amax_0,  Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd_dropout_seed(Tensor seed, Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? q_scale_s, Tensor? q_scale_o, Tensor? d_scale_s, bool is_amax_s, bool is_amax_o,  Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd.scalar(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, float d_scale_q, float d_scale_k, float d_scale_v, float q_scale_s, float q_scale_o, float d_scale_s, bool is_amax_s, bool is_amax_0, Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd_non_dropout.scalar(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, float d_scale_q, float d_scale_k, float d_scale_v, float q_scale_s, float q_scale_o, float d_scale_s, bool is_amax_s, bool is_amax_0,  Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd_dropout.scalar(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, float d_scale_q, float d_scale_k, float d_scale_v, float q_scale_s, float q_scale_o, float d_scale_s, bool is_amax_s, bool is_amax_0,  Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd_dropout_seed.scalar(Tensor seed, Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, float d_scale_q, float d_scale_k, float d_scale_v, float q_scale_s, float q_scale_o, float d_scale_s, bool is_amax_s, bool is_amax_o,  Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_bwd(Tensor grad, Tensor q, Tensor k, Tensor v, Tensor? attention_mask, Tensor m, Tensor linv, Tensor ? seed, bool is_causal, float p, float scale, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? d_scale_s, Tensor? d_scale_do, Tensor? d_scale_ds, Tensor? q_scale_s, Tensor? q_scale_ds, bool is_amax_ds, Tensor fwd_out) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::habana_seed_generator(Tensor seed, Tensor counter, int size) -> Tensor");
  HABANA_RANDOM_DEF_CHECKPOINT(bernoulli, "Tensor seed, Tensor self")
  HABANA_RANDOM_DEF_CHECKPOINT(poisson, "Tensor seed, Tensor self")
  HABANA_RANDOM_DEF_CHECKPOINT(
      rand,
      "Tensor seed, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None")
  HABANA_RANDOM_DEF_CHECKPOINT(
      randn,
      "Tensor seed, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None")
  HABANA_RANDOM_DEF_CHECKPOINT(
      randint,
      "Tensor seed, SymInt low, SymInt high, SymInt[] size, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None")
  HABANA_RANDOM_DEF_CHECKPOINT(
      multinomial,
      "Tensor seed, Tensor self, int num_samples, bool replacement=False")
  HABANA_RANDOM_DEF_CHECKPOINT(
      uniform, "Tensor seed, Tensor self, float from=0, float to=1")
  HABANA_RANDOM_DEF_CHECKPOINT(
      randperm,
      "Tensor seed, SymInt n, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None")
  HABANA_RANDOM_DEF_CHECKPOINT_2_OUTS(
      native_dropout, "Tensor seed, Tensor input, float p, bool? train")
  m.def(
      "hpu::bincount_backend(Tensor self, int length, Tensor? weights) -> (Tensor)");
}

TORCH_LIBRARY_IMPL(hpu, HPU, m) {
  m.impl("hpu::convert_from_int4", convert_from_int4_lazy);
  m.impl("hpu::convert_from_uint4", convert_from_uint4_lazy);
  m.impl("hpu::mixture_of_experts", mixture_of_experts_lazy);
  m.impl(
      "hpu::mixture_of_experts.fused_weights",
      mixture_of_experts_fused_weights_lazy);
  m.impl(
      "hpu::mixture_of_experts.fp8_measurement",
      mixture_of_experts_fp8_measurement_lazy);
  m.impl(
      "hpu::mixture_of_experts.fp8_measurement_fused_weights",
      mixture_of_experts_fp8_measurement_fused_weights_lazy);
  m.impl("hpu::mixture_of_experts.fp8", mixture_of_experts_fp8_lazy);
  m.impl(
      "hpu::mixture_of_experts.fp8_fused_weights",
      mixture_of_experts_fp8_fused_weights_lazy);
  m.impl(
      "hpu::mixture_of_experts.fp8_scalars",
      mixture_of_experts_fp8_scalars_lazy);
  m.impl(
      "hpu::mixture_of_experts.fp8_fused_weights_scalars",
      mixture_of_experts_fp8_fused_weights_scalars_lazy);
  m.impl("hpu::optimizer_lamb_fused_norm", optimizer_lamb_norm_hpu_lazy);
  m.impl(
      "hpu::optimizer_resource_apply_momentum",
      optimizer_resource_apply_momentum_hpu_wrap);
  m.impl("hpu::optimizer_lamb_phase1", optimizer_lamb_phase1);
  m.impl("hpu::optimizer_lamb_phase2", optimizer_lamb_phase2);
  m.impl("hpu::optimizer_adamw", optimizer_adamw_hpu_wrap);
  m.impl("hpu::sdpa_fwd", sdpa_fwd_wrap);
  m.impl("hpu::sdpa_recomp_fwd", sdpa_recomp_fwd_wrap);
  m.impl("hpu::fp8_sdpa_recomp_fwd", fp8_sdpa_recomp_fwd_lazy);
  m.impl("hpu::fp8_sdpa_recomp_fwd.scalar", fp8_sdpa_recomp_fwd_scalar_lazy);
  m.impl("hpu::fp8_sdpa_fwd", fp8_sdpa_fwd_wrap);
  m.impl("hpu::fp8_sdpa_recomp_bwd", fp8_sdpa_recomp_bwd_lazy);
}

TORCH_LIBRARY_IMPL(torchvision, HPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::nms"),
      TORCH_FN(torchvision_nms_hpu_wrap));
}

// We need to override matmul implementation also for inference,
// to have the same implementation as matmul forward in autograd.
TORCH_LIBRARY_IMPL(aten, HPU, m) {
  m.impl("matmul", matmul_inference);
}

TORCH_LIBRARY(hccl, m) {
  m.def(
      "broadcast_(Tensor(a!) tensor, int root_rank, int comm_id) -> Tensor(a!)");
  m.def(
      "allreduce_(Tensor(a!) tensor, int reduceOp, int comm_id) -> Tensor(a!)");
  m.def(
      "reduce_(Tensor(a!) tensor, int dst_rank, int reduceOp, int comm_id) -> Tensor(a!)");
  m.def(
      "alltoall_out(Tensor input_tensor, int comm_id,int[]  outputSplitSizes, int[] inputSplitSizes, Tensor(a!) output_tensor) -> Tensor(a!)");
  m.def(
      "allgather_out(Tensor input_tensor, int comm_id, Tensor(a!) output_tensor) -> Tensor(a!)");
  m.def(
      "reduce_scatter_out(Tensor input_tensor, int reduceOp, int comm_id, Tensor(a!) output_tensor) -> Tensor(a!)");
  m.def(
      "send_(Tensor(a!) tensor,  int dst_rank,  int tag, int comm_id) -> Tensor(a!)");
  m.def(
      "recv_(Tensor(a!) tensor,  int src_rank,  int tag, int comm_id) -> Tensor(a!)");
}
