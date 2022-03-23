/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <ATen/Operators.h>
#include <ATen/native/CPUFallback.h>

#define PARAMS1(...) __VA_ARGS__
#define PARAMS2(...) __VA_ARGS__
#define INQUOTE(x) #x
#define FIRST_ARG_(N, ...) N
#define FIRST_ARG(args) FIRST_ARG_ args
#define GET_FIRST(...) FIRST_ARG((__VA_ARGS__))

#define FALLBACK_IF_UNSUPPORTED_OP(input, param1, param2)                               \
  if (!hpu_check_inputs_impl(INQUOTE(input), {param1}))                                 \
      return at::native::call_fallback_fn<&cpu_fallback, ATEN_OP(input)>::call(param2); \

#define FALLBACK_IF_UNSUPPORTED_OP_O(input, param1, param2,overload)                              \
  if (!hpu_check_inputs_impl(INQUOTE(input), {param1}))                                           \
      return at::native::call_fallback_fn<&cpu_fallback, ATEN_OP2(input,overload)>::call(param2); \


#define FALLBACK_IF_UNSUPPORTED_OP1(input, param1, param2)                              \
  if (!(hpu_check_inputs_impl(INQUOTE(input), {param1 }) &&                             \
        (check_handle->get_status()))) {                                                \
      return at::native::call_fallback_fn<&cpu_fallback, ATEN_OP(input)>::call(param2); \
    }

#define FALLBACK_IF_UNSUPPORTED_OP1_O(input, param1, param2,overload)                             \
  if (!(hpu_check_inputs_impl(INQUOTE(input), {param1 }) &&                                       \
        (check_handle->get_status()))) {                                                          \
      return at::native::call_fallback_fn<&cpu_fallback, ATEN_OP2(input,overload)>::call(param2); \
    }

#define FALLBACK_IF_UNSUPPORTED_OP2(input, param2)                                      \
      return at::native::call_fallback_fn<&cpu_fallback, ATEN_OP(input)>::call(param2); \


#define FALLBACK_IF_UNSUPPORTED_OP2_O(input, param2,overload)                                     \
      return at::native::call_fallback_fn<&cpu_fallback, ATEN_OP2(input,overload)>::call(param2); \

namespace habana {
void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);
} // namespace habana
