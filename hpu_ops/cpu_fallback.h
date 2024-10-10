/*******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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
#include <ATen/native/CPUFallback.h>
#include "habana_kernels/kernel_input_checks.h"
#include "hpu_ops/cpu_fallback_internal.h"

#define PARAMS1(...) __VA_ARGS__
#define PARAMS2(...) __VA_ARGS__
#define INQUOTE(x) #x
#define FIRST_ARG_(N, ...) N
#define FIRST_ARG(args) FIRST_ARG_ args
#define GET_FIRST(...) FIRST_ARG((__VA_ARGS__))

// fallback macros for generated ops
//
#define FALLBACK_IF_UNSUPPORTED_DTYPE(input, opname, args...) \
  if (ABSL_PREDICT_FALSE(!supported_dtypes_.count(input))) {  \
    return dispatch_fallback<ATEN_OP(opname)>::call(          \
        OpSupportLevel::Value::unsupported_dtype, args);      \
  }

#define FALLBACK_IF_UNSUPPORTED_DTYPE2(input, opname, overload, args...) \
  if (ABSL_PREDICT_FALSE(!supported_dtypes_.count(input))) {             \
    return dispatch_fallback<ATEN_OP2(opname, overload)>::call(          \
        OpSupportLevel::Value::unsupported_dtype, args);                 \
  }

#define FALLBACK_IF_UNSUPPORTED_DTYPE_ARG(input, dtype, opname, args...) \
  if (ABSL_PREDICT_FALSE(!supported_dtypes_.count(input, dtype))) {      \
    return dispatch_fallback<ATEN_OP(opname)>::call(                     \
        OpSupportLevel::Value::unsupported_dtype, args);                 \
  }

#define FALLBACK_IF_UNSUPPORTED_DTYPE_ARG2(                         \
    input, dtype, opname, overload, args...)                        \
  if (ABSL_PREDICT_FALSE(!supported_dtypes_.count(input, dtype))) { \
    return dispatch_fallback<ATEN_OP2(opname, overload)>::call(     \
        OpSupportLevel::Value::unsupported_dtype, args);            \
  }

#define FALLBACK_IF_UNSUPPORTED_DTYPE_PER_TENSOR(tensor, opname, args...) \
  if (ABSL_PREDICT_FALSE(                                                 \
          tensor.defined() &&                                             \
          !supported_dtypes_##tensor.count(tensor.scalar_type()))) {      \
    return dispatch_fallback<ATEN_OP(opname)>::call(                      \
        OpSupportLevel::Value::unsupported_dtype, args);                  \
  }

#define FALLBACK_IF_UNSUPPORTED_DTYPE_PER_TENSOR2(                   \
    tensor, opname, overload, args...)                               \
  if (ABSL_PREDICT_FALSE(                                            \
          tensor.defined() &&                                        \
          !supported_dtypes_##tensor.count(tensor.scalar_type()))) { \
    return dispatch_fallback<ATEN_OP2(opname, overload)>::call(      \
        OpSupportLevel::Value::unsupported_dtype, args);             \
  }

#define FALLBACK_IF_UNSUPPORTED_INPUTS(check_fn, opname, args...) \
  if (ABSL_PREDICT_FALSE(!check_fn)) {                            \
    return dispatch_fallback<ATEN_OP(opname)>::call(              \
        OpSupportLevel::Value::unsupported_dtype, args);          \
  }

#define FALLBACK_IF_UNSUPPORTED_INPUTS2(check_fn, opname, overload, args...) \
  if (ABSL_PREDICT_FALSE(!check_fn)) {                                       \
    return dispatch_fallback<ATEN_OP2(opname, overload)>::call(              \
        OpSupportLevel::Value::unsupported_dtype, args);                     \
  }

// fallback macros for manual ops
#define FALLBACK_IF_UNSUPPORTED_OP_RT(result_dtype, input, param1, param2) \
  {                                                                        \
    auto is_supported = hpu_check_inputs_impl(INQUOTE(input), {param1});   \
    if (!is_supported)                                                     \
      return dispatch_fallback<ATEN_OP(input)>::call(                      \
          is_supported, result_dtype, param2);                             \
  }

#define FALLBACK_IF_UNSUPPORTED_OP(input, param1, param2)                   \
  {                                                                         \
    auto is_supported = hpu_check_inputs_impl(INQUOTE(input), {param1});    \
    if (!is_supported)                                                      \
      return dispatch_fallback<ATEN_OP(input)>::call(is_supported, param2); \
  }

#define FALLBACK_IF_UNSUPPORTED_OP_O(input, param1, param2, overload)    \
  {                                                                      \
    auto is_supported = hpu_check_inputs_impl(INQUOTE(input), {param1}); \
    if (!is_supported)                                                   \
      return dispatch_fallback<ATEN_OP2(input, overload)>::call(         \
          is_supported, param2);                                         \
  }

#define FALLBACK_IF_UNSUPPORTED_OP1(input, param1, param2)                  \
  {                                                                         \
    auto is_supported = hpu_check_inputs_impl(INQUOTE(input), {param1});    \
    if (is_supported && !check_handle->get_status())                        \
      is_supported = OpSupportLevel::Value::unsupported_args;               \
    if (!is_supported)                                                      \
      return dispatch_fallback<ATEN_OP(input)>::call(is_supported, param2); \
  }

#define FALLBACK_IF_UNSUPPORTED_OP1_RT(result_dtype, input, param1, param2) \
  {                                                                         \
    auto is_supported = hpu_check_inputs_impl(INQUOTE(input), {param1});    \
    if (is_supported && !check_handle->get_status())                        \
      is_supported = OpSupportLevel::Value::unsupported_args;               \
    if (!is_supported)                                                      \
      return dispatch_fallback<ATEN_OP(input)>::call(                       \
          is_supported, result_dtype, param2);                              \
  }

#define FALLBACK_IF_UNSUPPORTED_OP1_O(input, param1, param2, overload)   \
  {                                                                      \
    auto is_supported = hpu_check_inputs_impl(INQUOTE(input), {param1}); \
    if (is_supported && !check_handle->get_status())                     \
      is_supported = OpSupportLevel::Value::unsupported_args;            \
    if (!is_supported)                                                   \
      return dispatch_fallback<ATEN_OP2(input, overload)>::call(         \
          is_supported, param2);                                         \
  }

#define FALLBACK_UNSUPPORTED_OP2(input, param2)   \
  return dispatch_fallback<ATEN_OP(input)>::call( \
      OpSupportLevel::Value::unsupported, param2);

#define FALLBACK_UNSUPPORTED_OP2_DTYPE(input, dtype, param2) \
  return dispatch_fallback<ATEN_OP(input)>::call(            \
      OpSupportLevel::Value::unsupported, dtype, param2);

#define FALLBACK_UNSUPPORTED_OP2_O(input, param2, overload)  \
  return dispatch_fallback<ATEN_OP2(input, overload)>::call( \
      OpSupportLevel::Value::unsupported, param2);

#define VAL_FALLBACK_IF_UNSUPPORTED_DTYPE(opname, args...)        \
  if (ABSL_PREDICT_FALSE(!validator_##opname.Validate({args}))) { \
    return dispatch_fallback<ATEN_OP(opname)>::call(              \
        OpSupportLevel::Value::unsupported_dtype, args);          \
  } else {                                                        \
    require_h2d = validator_##opname.IsRequireH2D();              \
    require_st = validator_##opname.IsRequireST();                \
  }

#define VAL_FALLBACK_IF_UNSUPPORTED_DTYPE2(opname, overload, args...)          \
  if (ABSL_PREDICT_FALSE(!validator_##opname##_##overload.Validate({args}))) { \
    return dispatch_fallback<ATEN_OP2(opname, overload)>::call(                \
        OpSupportLevel::Value::unsupported_dtype, args);                       \
  } else {                                                                     \
    require_h2d = validator_##opname##_##overload.IsRequireH2D();              \
    require_st = validator_##opname##_##overload.IsRequireST();                \
  }

#define VAL_CUSTOM_FALLBACK_IF_UNSUPPORTED_DTYPE(opname, args...)       \
  if (ABSL_PREDICT_FALSE(!validator_##opname.ValidateCustom({args}))) { \
    return dispatch_fallback<ATEN_OP(opname)>::call(                    \
        OpSupportLevel::Value::unsupported_dtype, args);                \
  }

#define VAL_CUSTOM_FALLBACK_IF_UNSUPPORTED_DTYPE2(opname, overload, args...) \
  if (ABSL_PREDICT_FALSE(                                                    \
          !validator_##opname##_##overload.ValidateCustom({args}))) {        \
    return dispatch_fallback<ATEN_OP2(opname, overload)>::call(              \
        OpSupportLevel::Value::unsupported_dtype, args);                     \
  }

namespace habana {

template <class Op>
using dispatch_fallback = detail::_dispatch_fallback<Op, typename Op::schema>;

} // namespace habana
