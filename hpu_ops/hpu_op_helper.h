/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include <perf_lib_layer_params.h>
#include "cpu_fallback.h"
#include "op_backend.h"
#include "supported_dtypes.h"

namespace habana {

inline at::Tensor& stack_tensor(at::Stack& stack, int index) {
  return stack.at(index).toTensor();
}

inline at::Tensor stack_tensor(const at::Stack& stack, int index) {
  return stack.at(index).toTensor();
}

inline std::string& update_guid_dtype(
    std::string& guid,
    const std::string& dtype_str) {
  guid = guid.substr(0, guid.find_last_of('_') + 1).append(dtype_str);
  return guid;
}

inline std::string& update_guid_dtype(
    std::string& guid,
    c10::ScalarType dtype) {
  return update_guid_dtype(guid, habana_helpers::name_suffix_from_type(dtype));
}

inline int get_dim_in_tpc_order(int64_t dim_, int64_t max_dims) {
  auto dim = at::maybe_wrap_dim(dim_, max_dims, /*wrap_scalar=*/true);
  return static_cast<int>(max_dims - dim - 1);
}

std::string to_string(const at::IValue& ival);

std::vector<at::Tensor> GetMetaTensorList(
    const std::vector<at::Tensor>& tensors);
std::vector<c10::optional<at::Tensor>> GetMetaOptTensorList(
    const std::vector<c10::optional<at::Tensor>>& tensors);

template <typename T>
T& get(fint_t&);

template <>
inline int& get<int>(fint_t& u) {
  return u.i;
}
template <>
inline float& get<float>(fint_t& u) {
  return u.f;
}

} // namespace habana

#define PARAMS_STUB(structname) \
  size = sizeof(structname);    \
  auto params = std::make_shared<structname>()

// Use when you want to define your own size and param var names
#define PARAMS_STUB_VARS(structname, params_size, params) \
  const size_t& params_size = sizeof(structname);         \
  auto params = std::make_shared<structname>()

#define HPU_OP_BACKEND(op)                                            \
  struct op : OpBackend {                                             \
    op(int device_id,                                                 \
       const std::string& guid,                                       \
       c10::ScalarType scalar_type,                                   \
       const std::vector<int>& res_ids,                               \
       const std::vector<int>& inplace_ids,                           \
       const std::vector<int>& scalar_ids,                            \
       bool is_outfn)                                                 \
        : OpBackend(                                                  \
              device_id,                                              \
              guid,                                                   \
              scalar_type,                                            \
              res_ids,                                                \
              inplace_ids,                                            \
              scalar_ids,                                             \
              is_outfn){};                                            \
    void AddNode(synapse_helpers::graph&, const at::Stack&) override; \
  };

#define HPU_OP_FRONTEND(op)                                                    \
  template <typename T>                                                        \
  struct op : habana_lazy::LazyOp<T> {                                         \
    op(const std::string& qualstring,                                          \
       const std::vector<at::IValue>& inputs,                                  \
       const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn = \
           {});                                                                \
    T get_result_overrideable() override;                                      \
  };

#define FILL_PARAMS_DECL(fn) \
  std::shared_ptr<void> fn(const at::Stack&, size_t&);

#define OUTSHAPE_DECL(fn) sizes_vec fn(const at::Stack&, bool = false);

#define HPU_SUPPORTED_DTYPES(fn, supported_dtypes) \
  const static SupportedDtypes fn##_supported_dtypes supported_dtypes;

#define FALLBACK_IF_UNSUPPORTED_DTYPE(input, opname, args...)                  \
  if (ABSL_PREDICT_FALSE(!opname##_supported_dtypes.count(input))) {           \
    return at::native::call_fallback_fn<&cpu_fallback, ATEN_OP(opname)>::call( \
        args);                                                                 \
  }

#define FALLBACK_IF_UNSUPPORTED_DTYPE2(input, opname, overload, args...)   \
  if (ABSL_PREDICT_FALSE(                                                  \
          !opname##_##overload##_supported_dtypes.count(input))) {         \
    return at::native::                                                    \
        call_fallback_fn<&cpu_fallback, ATEN_OP2(opname, overload)>::call( \
            args);                                                         \
  }

#define FALLBACK_IF_UNSUPPORTED_DTYPE_ARG(input, dtype, opname, args...)       \
  if (ABSL_PREDICT_FALSE(!opname##_supported_dtypes.count(input, dtype))) {    \
    return at::native::call_fallback_fn<&cpu_fallback, ATEN_OP(opname)>::call( \
        args);                                                                 \
  }

#define FALLBACK_IF_UNSUPPORTED_DTYPE_ARG2(                                \
    input, dtype, opname, overload, args...)                               \
  if (ABSL_PREDICT_FALSE(                                                  \
          !opname##_##overload##_supported_dtypes.count(input, dtype))) {  \
    return at::native::                                                    \
        call_fallback_fn<&cpu_fallback, ATEN_OP2(opname, overload)>::call( \
            args);                                                         \
  }

#define FALLBACK_IF_UNSUPPORTED_DTYPE_PER_TENSOR(tensor, opname, args...)      \
  if (ABSL_PREDICT_FALSE(                                                      \
          tensor.defined() &&                                                  \
          !opname##_##tensor##_supported_dtypes.count(                         \
              tensor.scalar_type()))) {                                        \
    return at::native::call_fallback_fn<&cpu_fallback, ATEN_OP(opname)>::call( \
        args);                                                                 \
  }

#define FALLBACK_IF_UNSUPPORTED_DTYPE_PER_TENSOR2(                         \
    tensor, opname, overload, args...)                                     \
  if (ABSL_PREDICT_FALSE(                                                  \
          tensor.defined() &&                                              \
          !opname##_##overload##_##tensor##_supported_dtypes.count(        \
              tensor.scalar_type()))) {                                    \
    return at::native::                                                    \
        call_fallback_fn<&cpu_fallback, ATEN_OP2(opname, overload)>::call( \
            args);                                                         \
  }

#define FALLBACK_IF_UNSUPPORTED_INPUTS(check_fn, op, args...)              \
  if (ABSL_PREDICT_FALSE(!check_fn(args))) {                               \
    return at::native::call_fallback_fn<&cpu_fallback, ATEN_OP(op)>::call( \
        args);                                                             \
  }

#define FALLBACK_IF_UNSUPPORTED_INPUTS2(check_fn, op, overload, args...)     \
  if (ABSL_PREDICT_FALSE(!check_fn(args))) {                                 \
    return at::native::                                                      \
        call_fallback_fn<&cpu_fallback, ATEN_OP2(op, overload)>::call(args); \
  }

#define FALLBACK_CHECK(check_fn, signature...)          \
  extern const std::function<bool(signature)> check_fn; \
  const std::function<bool(signature)> check_fn = [](signature)
