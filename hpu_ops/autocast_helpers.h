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

namespace at {
namespace autocast {

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
    return (*F)(
        cached_cast(get_autocast_hpu_dtype(), args, DeviceType::HPU)...);
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
    return (*F)(cached_cast(at::kFloat, args, DeviceType::HPU)...);
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
    return (*F)(cached_cast(to_type, args, DeviceType::HPU)...);
  }
};

template <class Ret, class Signature, class... Args>
inline Ret cast_firstarg(Signature* F, const Tensor& first, Args... args) {
  return (*F)(
      cached_cast(get_autocast_hpu_dtype(), first, DeviceType::HPU), args...);
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

#define KERNEL(FUNC, REGISTER_NAME, SIGNATURE, POLICY) \
  m.impl(                                              \
      TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME),    \
      &WrapFunction<CastPolicy::POLICY, SIGNATURE, &FUNC>::type::call);

} // namespace autocast
} // namespace at
