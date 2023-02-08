/******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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
#include <fstream>
#include <unordered_set>

#include "habana_helpers/logging.h"

namespace at {
namespace autocast {

std::unordered_set<std::string> load_list(
    const char* list_name,
    const std::unordered_set<std::string>& default_list) {
  auto path = std::getenv(list_name);
  if (path == nullptr) {
    return default_list;
  }
  std::ifstream file(path);
  if (!file.is_open()) {
    PT_BRIDGE_WARN(
        "Failed to open file with ops to autocast: ",
        path,
        ". Default list loaded.");
    return default_list;
  }
  std::unordered_set<std::string> list;
  std::string line;
  while (getline(file, line)) {
    list.insert(line);
  }
  return list;
}

// Below lists are based on the hmp lists from
// pytorch-integration/python_packages/habana_frameworks/torch/hpex/hmp/
static const std::unordered_set<std::string> default_lower_ops{
    "addmm",
    "batch_norm",
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
    "log_softmax"};
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
    load_list("LOWER_LIST", default_lower_ops);
static const std::unordered_set<std::string> fp32_list =
    load_list("FP32_LIST", default_fp32_ops);
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

Tensor cast(at::ScalarType to_type, const Tensor& arg, DeviceType device_type) {
  // HPU in lazy mode doesn't benefit from cached casts. Potential optimization
  // are done in GC level. Moreover, it leaves persistent tensors from cast
  // operations, when HPU Graphs are used or .cpu() is called in the scope of
  // autocast. Since torch.autocast has caching enabled by default, to avoid the
  // risk of bad performance, cached casts are permanently disabled from
  // autocast on HPU.
  // TODO Analyze impact of cached casts when graph mode in PT 2.0 is
  // introduced.
#if 0
  return cached_cast(to_type, arg, device_type);
#else
  if (is_eligible(arg, device_type) && (arg.scalar_type() != to_type)) {
    return arg.to(to_type);
  } else {
    return arg;
  }
#endif
}

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
enum class CastPolicy : uint8_t {
  lower_precision_fp = 0, // Cast all inputs to lower_precision_fp
  fp32, // Cast all inputs to at::kFloat
  promote, // Run in the widest dtype among several args.
  lower_first_arg, // Cast first input to lower_precision_fp
};

// Base template for WrapFunction_, which is specialized to contain a "call"
// method each CastPolicy
template <
    CastPolicy policy,
    class Signature,
    Signature* F,
    class Ret,
    class ArgList>
struct WrapFunction_ {};

// CastPolicy::lower_precision_fp
template <class Signature, Signature* F, class Ret, class... Args>
struct WrapFunction_<
    CastPolicy::lower_precision_fp,
    Signature,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(DispatchKey::AutocastHPU);
    return (*F)(cast(get_autocast_hpu_dtype(), args, DeviceType::HPU)...);
  }
};

// CastPolicy::fp32
template <class Signature, Signature* F, class Ret, class... Args>
struct WrapFunction_<
    CastPolicy::fp32,
    Signature,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(DispatchKey::AutocastHPU);
    return (*F)(cast(at::kFloat, args, DeviceType::HPU)...);
  }
};

// CastPolicy::promote
template <class Signature, Signature* F, class Ret, class... Args>
struct WrapFunction_<
    CastPolicy::promote,
    Signature,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(DispatchKey::AutocastHPU);
    auto to_type =
        promote_type(get_autocast_hpu_dtype(), DeviceType::HPU, args...);
    return (*F)(cast(to_type, args, DeviceType::HPU)...);
  }
};

template <class Ret, class Signature, class T, class... Args>
inline Ret cast_firstarg(Signature* F, const T& first, Args... args) {
  return (*F)(cast(get_autocast_hpu_dtype(), first, DeviceType::HPU), args...);
}

// CastPolicy::lower_first_arg
template <class Signature, Signature* F, class Ret, class... Args>
struct WrapFunction_<
    CastPolicy::lower_first_arg,
    Signature,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(DispatchKey::AutocastHPU);
    return cast_firstarg<Ret, Signature>(F, args...);
  }
};

// Wrapper to infer return_type and parameter_types for WrapFunction_ (imitating
// core/boxing/impl/WrapFunctionIntoFunctor.h)
template <
    CastPolicy policy,
    class Signature, // The signature for which we're registering.  The
                     // dispatcher's calling code invokes our registered
                     // functions with arguments matching Signature, so we
                     // register WrapFunction_::call methods with a matching
                     // signature to properly field those arguments.
                     // guts::function_traits below extracts return_type and
                     // parameter_types from Signature, which WrapFunction_
                     // templates above use to declare their call methods.
    Signature* F> // The actual function we're redispatching to.
struct WrapFunction final {
  using type = WrapFunction_<
      policy,
      Signature,
      F,
      typename guts::function_traits<Signature>::return_type,
      typename guts::function_traits<Signature>::parameter_types>;
};

#define ADD_NS(RAW_OP) at::RAW_OP

#define KERNEL(FUNC, REGISTER_NAME, SIGNATURE)                               \
  if (lower_list.count(#FUNC)) {                                             \
    if (lower_first_ops.count(#FUNC)) {                                      \
      m.impl(                                                                \
          TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME),                      \
          &WrapFunction<                                                     \
              CastPolicy::lower_first_arg,                                   \
              SIGNATURE,                                                     \
              &ADD_NS(FUNC)>::type::call);                                   \
    } else {                                                                 \
      m.impl(                                                                \
          TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME),                      \
          &WrapFunction<                                                     \
              CastPolicy::lower_precision_fp,                                \
              SIGNATURE,                                                     \
              &ADD_NS(FUNC)>::type::call);                                   \
    }                                                                        \
  } else if (fp32_list.count(#FUNC)) {                                       \
    m.impl(                                                                  \
        TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME),                        \
        &WrapFunction<CastPolicy::fp32, SIGNATURE, &ADD_NS(FUNC)>::type::    \
            call);                                                           \
  } else if (promote_list.count(#FUNC)) {                                    \
    m.impl(                                                                  \
        TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME),                        \
        &WrapFunction<CastPolicy::promote, SIGNATURE, &ADD_NS(FUNC)>::type:: \
            call);                                                           \
  }

} // namespace autocast
} // namespace at
