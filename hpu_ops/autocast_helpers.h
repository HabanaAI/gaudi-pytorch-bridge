/******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
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

#pragma once

#include <ATen/autocast_mode.h>
#include <string_view>
#include <unordered_set>
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

namespace at {
namespace autocast {

constexpr std::string_view AUTOCAST_LOWER_LIST =
    "PT_HPU_AUTOCAST_LOWER_PRECISION_OPS_LIST";
constexpr std::string_view AUTOCAST_LOWER_LIST_DEPRECATED = "LOWER_LIST";
constexpr std::string_view AUTOCAST_FP32_LIST = "PT_HPU_AUTOCAST_FP32_OPS_LIST";
constexpr std::string_view AUTOCAST_FP32_LIST_DEPRECATED = "FP32_LIST";

std::unordered_set<std::string> load_list(
    const std::string_view list_name,
    const std::unordered_set<std::string>& default_list);

static const std::unordered_set<std::string> default_lower_ops{
    "addmm",
    "addbmm",
    "batch_norm",
    "baddbmm",
    "bmm",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "dot",
    "dropout",
    "feature_dropout",
    "group_norm",
    "instance_norm",
    "layer_norm",
    "leaky_relu",
    "linear",
    "matmul",
    "mean",
    "mm",
    "mul",
    "mv",
    "softmax",
    "log_softmax",
    "scaled_dot_product_attention"};
static const std::unordered_set<std::string> default_fp32_ops{
    "acos",
    "addcdiv",
    "asin",
    "atan2",
    "bilinear",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "cdist",
    "cosh",
    "cosine_embedding_loss",
    "cosine_similarity",
    "cross_entropy_loss",
    "dist",
    "div",
    "divide",
    "embedding",
    "embedding_bag",
    "erfinv",
    "exp",
    "expm1",
    "hinge_embedding_loss",
    "huber_loss",
    "kl_div",
    "l1_loss",
    "log",
    "log10",
    "log1p",
    "log2",
    "logsumexp",
    "margin_ranking_loss",
    "mse_loss",
    "multi_margin_loss",
    "multilabel_margin_loss",
    "nll_loss",
    "pdist",
    "poisson_nll_loss",
    "pow",
    "reciprocal",
    "renorm",
    "rsqrt",
    "sinh",
    "smooth_l1_loss",
    "soft_margin_loss",
    "softplus",
    "tan",
    "topk",
    "triplet_margin_loss",
    "truediv",
    "true_divide"};
static const std::unordered_set<std::string> lower_first_ops{
    "layer_norm",
    "group_norm",
    "instance_norm",
    "batch_norm"};

// Lists of ops for autocast registration are taken from above default lists, or
// from external files, passed with below envs.

static const std::unordered_set<std::string> lower_list =
    load_list(AUTOCAST_LOWER_LIST, default_lower_ops);
static const std::unordered_set<std::string> fp32_list =
    load_list(AUTOCAST_FP32_LIST, default_fp32_ops);
static const std::unordered_set<std::string> promote_list{
    "add",
    "addcmul",
    "addcdiv",
    "cat",
    "div",
    "exp",
    "mul",
    "pow",
    "sub",
    "iadd",
    "truediv",
    "stack"};

Tensor cast(at::ScalarType to_type, const Tensor& arg, DeviceType device_type);

// Overload to process optional<Tensor>
inline c10::optional<Tensor> cast(
    at::ScalarType to_type,
    const c10::optional<Tensor>& arg,
    DeviceType device_type = DeviceType::HPU) {
  if (arg.has_value()) {
    return cast(to_type, *arg, device_type);
  } else {
    return c10::nullopt;
  }
}

// Overload to process TensorLists
inline std::vector<Tensor> cast(
    at::ScalarType to_type,
    const TensorList& arg,
    DeviceType device_type = DeviceType::HPU) {
  std::vector<Tensor> vec;
  vec.reserve(arg.size());
  for (const auto& t : arg) {
    vec.push_back(cast(to_type, t, device_type));
  }
  return vec;
}

// Template to catch non-Tensor args.
template <typename T>
inline T cast(at::ScalarType, T arg, DeviceType = DeviceType::HPU) {
  return arg;
}

// Below structures are taken from pytorch/aten/src/ATen/autocast_mode.cpp
// and adjusted/enhanced for HPU usage

// Policies correspond to op categories that need code-divergent handling.
// Wrapper templates below are specialized based on a policy template parameter.
enum class Hpu_CastPolicy : uint8_t {
  lower_precision_fp = 0, // Cast all inputs to lower_precision_fp
  fp32, // Cast all inputs to at::kFloat
  promote, // Run in the widest dtype among several args.
  lower_first_arg, // Cast first input to lower_precision_fp
};

// Base template for Hpu_WrapFunction_, which is specialized to contain a "call"
// method each Hpu_CastPolicy
template <
    Hpu_CastPolicy policy,
    class Signature,
    Signature* F,
    class Ret,
    class ArgList>
struct Hpu_WrapFunction_ {};

// Hpu_CastPolicy::lower_precision_fp
template <class Signature, Signature* F, class Ret, class... Args>
struct Hpu_WrapFunction_<
    Hpu_CastPolicy::lower_precision_fp,
    Signature,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(DispatchKey::AutocastHPU);
#if IS_PYTORCH_AT_LEAST(2, 4)
    return (*F)(cast(get_autocast_dtype(at::kHPU), args, DeviceType::HPU)...);
#else
    return (*F)(cast(get_autocast_hpu_dtype(), args, DeviceType::HPU)...);
#endif
  }
};

// Hpu_CastPolicy::fp32
template <class Signature, Signature* F, class Ret, class... Args>
struct Hpu_WrapFunction_<
    Hpu_CastPolicy::fp32,
    Signature,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(DispatchKey::AutocastHPU);
    return (*F)(cast(at::kFloat, args, DeviceType::HPU)...);
  }
};

// Hpu_CastPolicy::promote
template <class Signature, Signature* F, class Ret, class... Args>
struct Hpu_WrapFunction_<
    Hpu_CastPolicy::promote,
    Signature,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(DispatchKey::AutocastHPU);
#if IS_PYTORCH_AT_LEAST(2, 4)
    auto to_type =
        promote_type(get_autocast_dtype(at::kHPU), DeviceType::HPU, args...);
#else
    auto to_type =
        promote_type(get_autocast_hpu_dtype(), DeviceType::HPU, args...);
#endif
    return (*F)(cast(to_type, args, DeviceType::HPU)...);
  }
};

template <class Ret, class Signature, class T, class... Args>
inline Ret cast_firstarg(Signature* F, const T& first, Args... args) {
#if IS_PYTORCH_AT_LEAST(2, 4)
  return (*F)(
      cast(get_autocast_dtype(at::kHPU), first, DeviceType::HPU), args...);
#else
  return (*F)(cast(get_autocast_hpu_dtype(), first, DeviceType::HPU), args...);
#endif
}

// Hpu_CastPolicy::lower_first_arg
template <class Signature, Signature* F, class Ret, class... Args>
struct Hpu_WrapFunction_<
    Hpu_CastPolicy::lower_first_arg,
    Signature,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(DispatchKey::AutocastHPU);
    return cast_firstarg<Ret, Signature>(F, args...);
  }
};

// Wrapper to infer return_type and parameter_types for Hpu_WrapFunction_
// (imitating core/boxing/impl/WrapFunctionIntoFunctor.h)
template <
    Hpu_CastPolicy policy,
    class Signature, // The signature for which we're registering.  The
                     // dispatcher's calling code invokes our registered
                     // functions with arguments matching Signature, so we
                     // register Hpu_WrapFunction_::call methods with a matching
                     // signature to properly field those arguments.
                     // guts::function_traits below extracts return_type and
                     // parameter_types from Signature, which Hpu_WrapFunction_
                     // templates above use to declare their call methods.
    Signature* F> // The actual function we're redispatching to.
struct Hpu_WrapFunction final {
  using type = Hpu_WrapFunction_<
      policy,
      Signature,
      F,
      typename guts::function_traits<Signature>::return_type,
      typename guts::function_traits<Signature>::parameter_types>;
};

#define Hpu_ADD_NS(RAW_OP) at::RAW_OP

#define Hpu_KERNEL(FUNC, REGISTER_NAME, SIGNATURE)      \
  if (lower_list.count(#FUNC)) {                        \
    if (lower_first_ops.count(#FUNC)) {                 \
      m.impl(                                           \
          TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME), \
          &Hpu_WrapFunction<                            \
              Hpu_CastPolicy::lower_first_arg,          \
              SIGNATURE,                                \
              &Hpu_ADD_NS(FUNC)>::type::call);          \
    } else {                                            \
      m.impl(                                           \
          TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME), \
          &Hpu_WrapFunction<                            \
              Hpu_CastPolicy::lower_precision_fp,       \
              SIGNATURE,                                \
              &Hpu_ADD_NS(FUNC)>::type::call);          \
    }                                                   \
  } else if (fp32_list.count(#FUNC)) {                  \
    m.impl(                                             \
        TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME),   \
        &Hpu_WrapFunction<                              \
            Hpu_CastPolicy::fp32,                       \
            SIGNATURE,                                  \
            &Hpu_ADD_NS(FUNC)>::type::call);            \
  } else if (promote_list.count(#FUNC)) {               \
    m.impl(                                             \
        TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME),   \
        &Hpu_WrapFunction<                              \
            Hpu_CastPolicy::promote,                    \
            SIGNATURE,                                  \
            &Hpu_ADD_NS(FUNC)>::type::call);            \
  }

} // namespace autocast
} // namespace at
