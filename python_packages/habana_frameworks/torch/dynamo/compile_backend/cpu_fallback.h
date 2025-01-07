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

#pragma once

#include <ATen/core/function_schema.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/tensorexpr/tensorexpr_init.h>
#include <torch/csrc/utils/python_symnode.h>
#include <torch/extension.h>
#include <tuple>
#include "habana_helpers/dtype_helpers.h"
#include "hpu_ops/op_validator.h"

struct SharedLayerOp {
  virtual bool func(torch::jit::Stack& stack, bool is_dynamic) = 0;
  habana::SharedMetaVector m_shared_meta;
};

#define RETURN_IF_UNSUPPORTED_DTYPE(input, opname, args...)  \
  if (ABSL_PREDICT_FALSE(!supported_dtypes_.count(input))) { \
    return false;                                            \
  }

#define RETURN_IF_UNSUPPORTED_DTYPE2(input, opname, overload, args...) \
  if (ABSL_PREDICT_FALSE(!supported_dtypes_.count(input))) {           \
    return false;                                                      \
  }

#define RETURN_IF_UNSUPPORTED_DTYPE_ARG(input, dtype, opname, args...) \
  if (ABSL_PREDICT_FALSE(!supported_dtypes_.count(input, dtype))) {    \
    return false;                                                      \
  }

#define RETURN_IF_UNSUPPORTED_DTYPE_ARG2(                           \
    input, dtype, opname, overload, args...)                        \
  if (ABSL_PREDICT_FALSE(!supported_dtypes_.count(input, dtype))) { \
    return false;                                                   \
  }

#define RETURN_IF_UNSUPPORTED_DTYPE_PER_TENSOR(tensor, opname, args...) \
  if (ABSL_PREDICT_FALSE(                                               \
          tensor.defined() &&                                           \
          !supported_dtypes_##tensor.count(tensor.scalar_type()))) {    \
    return false;                                                       \
  }

#define RETURN_IF_UNSUPPORTED_DTYPE_PER_TENSOR2(                     \
    tensor, opname, overload, args...)                               \
  if (ABSL_PREDICT_FALSE(                                            \
          tensor.defined() &&                                        \
          !supported_dtypes_##tensor.count(tensor.scalar_type()))) { \
    return false;                                                    \
  }

#define RETURN_IF_UNSUPPORTED_INPUTS(check_fn, opname, args...) \
  if (ABSL_PREDICT_FALSE(!check_fn)) {                          \
    return false;                                               \
  }

#define RETURN_IF_UNSUPPORTED_INPUTS2(check_fn, opname, overload, args...) \
  if (ABSL_PREDICT_FALSE(!check_fn)) {                                     \
    return false;                                                          \
  }

// fallback macros for manual ops
#define RETURN_IF_UNSUPPORTED_OP_RT(result_dtype, input, param1, param2) \
  {                                                                      \
    auto is_supported = hpu_check_inputs_impl(INQUOTE(input), {param1}); \
    if (!is_supported)                                                   \
      return false;                                                      \
  }

#define RETURN_IF_UNSUPPORTED_OP(input, param1, param2)                  \
  {                                                                      \
    auto is_supported = hpu_check_inputs_impl(INQUOTE(input), {param1}); \
    if (!is_supported)                                                   \
      return false;                                                      \
  }

#define RETURN_IF_UNSUPPORTED_OP_O(input, param1, param2, overload)      \
  {                                                                      \
    auto is_supported = hpu_check_inputs_impl(INQUOTE(input), {param1}); \
    if (!is_supported)                                                   \
      return false;                                                      \
  }

#define RETURN_IF_UNSUPPORTED_OP1(input, param1, param2)                 \
  {                                                                      \
    auto is_supported = hpu_check_inputs_impl(INQUOTE(input), {param1}); \
    if (is_supported && !check_handle->get_status())                     \
      is_supported = OpSupportLevel::Value::unsupported_args;            \
    if (!is_supported)                                                   \
      return false;                                                      \
  }

#define RETURN_IF_UNSUPPORTED_OP1_RT(result_dtype, input, param1, param2) \
  {                                                                       \
    auto is_supported = hpu_check_inputs_impl(INQUOTE(input), {param1});  \
    if (is_supported && !check_handle->get_status())                      \
      is_supported = OpSupportLevel::Value::unsupported_args;             \
    if (!is_supported)                                                    \
      return false;                                                       \
  }

#define RETURN_IF_UNSUPPORTED_OP1_O(input, param1, param2, overload)     \
  {                                                                      \
    auto is_supported = hpu_check_inputs_impl(INQUOTE(input), {param1}); \
    if (is_supported && !check_handle->get_status())                     \
      is_supported = OpSupportLevel::Value::unsupported_args;            \
    if (!is_supported)                                                   \
      return false;                                                      \
  }

#define RETURN_UNSUPPORTED_OP2(input, param2) return false;

#define RETURN_UNSUPPORTED_OP2_DTYPE(input, dtype, param2) return false;

#define RETURN_UNSUPPORTED_OP2_O(input, param2, overload) return false;

#define VAL_RETURN_IF_UNSUPPORTED_DTYPE(opname, is_dynamic, args...) \
  if (ABSL_PREDICT_FALSE(!validator_##opname.Validate(               \
          {args}, is_dynamic, false, m_shared_meta))) {              \
    return false;                                                    \
  }

#define VAL_RETURN_IF_UNSUPPORTED_DTYPE2(                           \
    opname, is_dynamic, overload, args...)                          \
  if (ABSL_PREDICT_FALSE(!validator_##opname##_##overload.Validate( \
          {args}, is_dynamic, false, m_shared_meta))) {             \
    return false;                                                   \
  }

#define VAL_CUSTOM_RETURN_IF_UNSUPPORTED_DTYPE(opname, is_dynamic, args...) \
  if (ABSL_PREDICT_FALSE(                                                   \
          !validator_##opname.ValidateCustom({args}, is_dynamic))) {        \
    return false;                                                           \
  }

#define VAL_CUSTOM_RETURN_IF_UNSUPPORTED_DTYPE2(                          \
    opname, is_dynamic, overload, args...)                                \
  if (ABSL_PREDICT_FALSE(!validator_##opname##_##overload.ValidateCustom( \
          {args}, is_dynamic))) {                                         \
    return false;                                                         \
  }

#define HPU_SUPPORTED_DTYPES(dtypes, suffix...) \
  const static SupportedDtypes supported_dtypes_##suffix dtypes;

template <typename SharedOp>
bool check_support(
    c10::FunctionSchema& schema,
    bool allow_numbers_as_tensors,
    bool is_dynamic,
    const py::list& shared_meta,
    py::args& args,
    const py::kwargs& kwargs) {
  torch::jit::Stack stack;
  habana::SharedMetaVector out_meta;
  {
    torch::jit::ToIValueAllowNumbersAsTensors g(allow_numbers_as_tensors);
    //  Acquire GIL for py::args and py::kwargs processing.
    py::gil_scoped_acquire ag;
    stack =
        torch::jit::createStackForSchema(schema, args, kwargs, c10::nullopt);

    for (const auto& sm : shared_meta) {
      const auto& meta = sm.cast<py::tuple>();
      out_meta.emplace_back(
          meta[0].cast<int>(),
          torch::python::detail::py_object_to_dtype(sm.cast<py::tuple>()[1]));
    }
  }
  static SharedOp shared_op;
  shared_op.m_shared_meta = std::move(out_meta);
  return shared_op.func(stack, is_dynamic);
}

template <typename T, size_t N>
std::array<T, N> as_array(const c10::List<c10::IValue>& list) {
  std::array<T, N> res;
  AT_ASSERT(list.size() == N);
  std::vector<T> vec;
  for (c10::IValue elem : list) {
    vec.push_back(elem.to<T>());
  }
  std::copy(vec.begin(), vec.end(), res.begin());
  return res;
}
