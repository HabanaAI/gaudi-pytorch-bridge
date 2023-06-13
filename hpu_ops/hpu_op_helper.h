/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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
#include <perf_lib_layer_params.h>
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/kernels_accumulation.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/lazy_kernels.h"
#include "op_backend.h"
#include "op_logger.h"

#include "supported_dtypes.h"

namespace habana {

template <class T, class InputType>
void scheduleAccTask(T&& lazy_op, InputType tensor) {
  habana_lazy::AccThread::Get().run(
      [op = std::move(lazy_op), tensor = std::move(tensor)]() mutable {
        PT_LAZY_TRACE_WITH_NAME(op.symbol().toUnqualString());
        op.call(tensor);
        habana_lazy::AccThread::Get().PushCleanupTask(
            [op = std::move(op), tensor = std::move(tensor)]() {});
      });
}

template <class T>
void scheduleAccTask(
    T&& lazy_op,
    std::vector<at::Tensor> result // pass explicit as copy to keep alive
) {
  habana_lazy::AccThread::Get().run(
      [op = std::move(lazy_op), result = std::move(result)]() mutable {
        PT_LAZY_TRACE_WITH_NAME(op.symbol().toUnqualString());
        op.call(result);
        habana_lazy::AccThread::Get().PushCleanupTask(
            [op = std::move(op), result = std::move(result)]() {});
      });
}

template <class T>
void scheduleAccTask(
    T&& lazy_op,
    std::vector<at::Tensor> result, // pass explicit as copy to keep alive
    std::vector<at::Tensor>&& tensor_list_copy) {
  habana_lazy::AccThread::Get().run(
      [op = std::move(lazy_op),
       result = std::move(result),
       tensor_list_copy = std::move(tensor_list_copy)]() mutable {
        PT_LAZY_TRACE_WITH_NAME(op.symbol().toUnqualString());
        op.call(result);
        habana_lazy::AccThread::Get().PushCleanupTask(
            [op = std::move(op),
             result = std::move(result),
             tensor_list_copy = std::move(tensor_list_copy)]() {});
      });
}

template <class T, class TupleType>
void scheduleAccTaskTuple(T&& lazy_op, TupleType& tuple) {
  std::vector<at::Tensor> tensors;
  for_each_in_tuple(
      tuple, [&tensors](const auto& result) { tensors.push_back(result); });
  HABANA_ASSERT(tensors.size() <= 5, "Only tuples up to 5 are supported");
  habana_lazy::AccThread::Get().run(
      [op = std::move(lazy_op), tensors = std::move(tensors)]() mutable {
        PT_LAZY_TRACE_WITH_NAME(op.symbol().toUnqualString());
        if (tensors.size() == 2) {
          op.call(std::tie(tensors[0], tensors[1]));
        } else if (tensors.size() == 3) {
          op.call(std::tie(tensors[0], tensors[1], tensors[2]));
        } else if (tensors.size() == 4) {
          op.call(std::tie(tensors[0], tensors[1], tensors[2], tensors[3]));
        } else if (tensors.size() == 5) {
          op.call(std::tie(
              tensors[0], tensors[1], tensors[2], tensors[3], tensors[4]));
        }
        habana_lazy::AccThread::Get().PushCleanupTask(
            [op = std::move(op), tensors = std::move(tensors)]() {});
      });
}

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
  auto string_or_error = synapse_helpers::graph::name_suffix_from_type(
      habana_helpers::pytorch_to_synapse_type(dtype),
      habana_helpers::isLongTypeSupported(guid));
  HABANA_ASSERT(
      absl::holds_alternative<std::string>(string_or_error),
      "Error getting suffix/precision type: ",
      Logger::synStatusToStr(
          absl::get<synapse_helpers::synapse_error>(string_or_error).status));
  return update_guid_dtype(
      guid,
      absl::get<std::string>(synapse_helpers::graph::name_suffix_from_type(
          habana_helpers::pytorch_to_synapse_type(dtype))));
}

inline int get_dim_in_tpc_order(int64_t dim_, int64_t max_dims) {
  auto dim = at::maybe_wrap_dim(dim_, max_dims, /*wrap_scalar=*/true);
  return static_cast<int>(max_dims - dim - 1);
}

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
#define PARAMS_STUB_VARS(structname, params, params_size) \
  const size_t& params_size = sizeof(structname);         \
  auto params = std::make_shared<structname>()

#define REGISTER_HPU_BACKEND(op, backendclass)              \
  add(op, [](const int device_id, c10::ScalarType type) {   \
    return std::make_shared<backendclass>(device_id, type); \
  })

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

#define HPU_OP_FRONTEND(FEServiceClass, op)                                   \
  template <typename T>                                                       \
  struct op : FEServiceClass<T> {                                             \
    op(const std::string& qualstring,                                         \
       const std::vector<at::IValue>& inputs,                                 \
       const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn = {}); \
    T get_result_overrideable() override;                                     \
  };

#define HPU_OP_FRONTEND_CUSTOM_CTOR(FEServiceClass, op, out_index, T...) \
  template <>                                                            \
  op<T>::op(                                                             \
      const std::string& qualstring,                                     \
      const std::vector<at::IValue>& inputs,                             \
      const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)   \
      : FEServiceClass<T>(qualstring, inputs, out_shapes_fn, out_index)

#define HPU_OP_FRONTEND_CREATE_RESULT(FEServiceClass, op, T...) \
  HPU_OP_FRONTEND_CUSTOM_CTOR(FEServiceClass, op, -1, T...)     \
  template <>                                                   \
  T op<T>::get_result_overrideable()

#define HPU_OP_FRONTEND_CREATE_RESULT_ONLY(FEServiceClass, op, T...) \
  template <>                                                        \
  T op<T>::get_result_overrideable()

#define HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(FEServiceClass, op, T...) \
  template <>                                                      \
  T op<T>::get_result_overrideable() {                             \
    return FEServiceClass<T>::get_result_overrideable();           \
  }                                                                \
  HPU_OP_FRONTEND_CUSTOM_CTOR(FEServiceClass, op, 0, T)

#define FILL_PARAMS_DECL(fn) \
  std::shared_ptr<void> fn(const at::Stack&, size_t&);

#define OUTSHAPE_DECL(fn) sizes_vec fn(const at::Stack&);
#define OUTMETA_DECL(fn) OutputMetaDataVector fn(const at::Stack&);

#define HPU_SUPPORTED_DTYPES(dtypes, suffix...) \
  const static SupportedDtypes supported_dtypes_##suffix dtypes;

#define MAYBE_FLUSH_OP(out_tensor_count) habana_lazy::flush_op(out_tensor_count)

#define RUN_MAYBE_WITH_ACC_THREAD(op, lazy_op)                              \
  if (habana_lazy::AccThread::Get().CanUseAccThread()) {                    \
    PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
    auto result = lazy_op.get_result();                                     \
    scheduleAccTask(std::move(lazy_op), result);                            \
    MAYBE_FLUSH_OP(1);                                                      \
    return result;                                                          \
  }                                                                         \
  return lazy_op.call();

#define RUN_MAYBE_WITH_ACC_THREAD_MODIFY_RESULT(op, lazy_op, result_func)   \
  if (habana_lazy::AccThread::Get().CanUseAccThread()) {                    \
    PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
    auto result = lazy_op.get_result();                                     \
    result_func(result);                                                    \
    scheduleAccTask(std::move(lazy_op), result);                            \
    MAYBE_FLUSH_OP(1);                                                      \
    return result;                                                          \
  }                                                                         \
  auto result = lazy_op.call();                                             \
  result_func(result);                                                      \
  return result;

#define RUN_INPLACE_MAYBE_WITH_ACC_THREAD(op, lazy_op, self)                \
  if (habana_lazy::AccThread::Get().CanUseAccThread()) {                    \
    PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
    self = lazy_op.get_result(self);                                        \
    scheduleAccTask(std::move(lazy_op), self);                              \
    MAYBE_FLUSH_OP(1);                                                      \
    return self;                                                            \
  }                                                                         \
  return lazy_op.call(self);

#define RUN_CONST_INPLACE_MAYBE_WITH_ACC_THREAD(op, lazy_op, self)          \
  if (habana_lazy::AccThread::Get().CanUseAccThread()) {                    \
    PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
    lazy_op.get_result(self);                                               \
    scheduleAccTask(std::move(lazy_op), self);                              \
    MAYBE_FLUSH_OP(1);                                                      \
    return self;                                                            \
  }                                                                         \
  return lazy_op.call(self);

template <typename... Args>
inline constexpr size_t tuple_elements(const std::tuple<Args...>&) {
  return sizeof...(Args);
}

#define RUN_TUPLE_MAYBE_WITH_ACC_THREAD(op, lazy_op)                        \
  if (habana_lazy::AccThread::Get().CanUseAccThread()) {                    \
    PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
    auto tuple = lazy_op.get_result();                                      \
    scheduleAccTaskTuple(std::move(lazy_op), tuple);                        \
    MAYBE_FLUSH_OP(tuple_elements(tuple));                                  \
    return tuple;                                                           \
  }                                                                         \
  return lazy_op.call();

#define RUN_INPLACE_TUPLE_MAYBE_WITH_ACC_THREAD(op, lazy_op, tuple)         \
  if (habana_lazy::AccThread::Get().CanUseAccThread()) {                    \
    PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
    tuple = lazy_op.get_result(tuple);                                      \
    scheduleAccTaskTuple(std::move(lazy_op), tuple);                        \
    MAYBE_FLUSH_OP(tuple_elements(tuple));                                  \
    return tuple;                                                           \
  }                                                                         \
  return lazy_op.call(tuple);

#define RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(op, func, out)                  \
  if (habana_lazy::AccThread::Get().CanUseAccThread()) {                    \
    PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
    habana_lazy::AccThread::Get().run([func = std::move(func)]() mutable {  \
      PT_LAZY_TRACE_WITH_NAME(#op);                                         \
      func();                                                               \
      habana_lazy::AccThread::Get().PushCleanupTask(                        \
          [func = std::move(func)]() {});                                   \
    });                                                                     \
    MAYBE_FLUSH_OP(1);                                                      \
  } else {                                                                  \
    func();                                                                 \
  }                                                                         \
  return out;

#define RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD_NO_FLUSH(op, func, out)         \
  if (habana_lazy::AccThread::Get().CanUseAccThread()) {                    \
    PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
    habana_lazy::AccThread::Get().run([func = std::move(func)]() mutable {  \
      PT_LAZY_TRACE_WITH_NAME(#op);                                         \
      func();                                                               \
      habana_lazy::AccThread::Get().PushCleanupTask(                        \
          [func = std::move(func)]() {});                                   \
    });                                                                     \
  } else {                                                                  \
    func();                                                                 \
  }                                                                         \
  return out;

#define RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD_MODIFY_RESULT(                  \
    op, func, out, result_func)                                             \
  if (habana_lazy::AccThread::Get().CanUseAccThread()) {                    \
    PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
    habana_lazy::AccThread::Get().run([func = std::move(func)]() mutable {  \
      PT_LAZY_TRACE_WITH_NAME(#op);                                         \
      func();                                                               \
      habana_lazy::AccThread::Get().PushCleanupTask(                        \
          [func = std::move(func)]() {});                                   \
    });                                                                     \
    result_func(out);                                                       \
    MAYBE_FLUSH_OP(1);                                                      \
  } else {                                                                  \
    func();                                                                 \
    result_func(out);                                                       \
  }                                                                         \
  return out;

#define RUN_MANUAL_OP_NO_RETURN_WITH_ACC_THREAD(op, func)                   \
  if (habana_lazy::AccThread::Get().CanUseAccThread()) {                    \
    PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
    habana_lazy::AccThread::Get().run([func = std::move(func)]() mutable {  \
      PT_LAZY_TRACE_WITH_NAME(#op);                                         \
      func();                                                               \
      habana_lazy::AccThread::Get().PushCleanupTask(                        \
          [func = std::move(func)]() {});                                   \
    });                                                                     \
    MAYBE_FLUSH_OP();                                                       \
  } else {                                                                  \
    func();                                                                 \
  }

#define RUN_MANUAL_OP_NO_RETURN_WITH_ACC_THREAD_NO_FLUSH(op, func)          \
  if (habana_lazy::AccThread::Get().CanUseAccThread()) {                    \
    PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
    habana_lazy::AccThread::Get().run([func = std::move(func)]() mutable {  \
      PT_LAZY_TRACE_WITH_NAME(#op);                                         \
      func();                                                               \
      habana_lazy::AccThread::Get().PushCleanupTask(                        \
          [func = std::move(func)]() {});                                   \
    });                                                                     \
  } else {                                                                  \
    func();                                                                 \
  }

#define RUN_WITH_PREDICATE_VIEW_OP_MAYBE_WITH_ACC_THREAD(                     \
    op, self, out, param_setter, additional_predicate)                        \
  if (habana_lazy::AccThread::Get().CanUseAccThread() &&                      \
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_ACC_VIEW_OPS_MODE) != 0) {                 \
    PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread");   \
    habana_lazy::AccThread::Get().run(                                        \
        [self, out, param_setter, additional_predicate]() {                   \
          lazy_view_fallback_handle(                                          \
              self, out, param_setter, additional_predicate);                 \
          habana_lazy::AccThread::Get().PushCleanupTask(                      \
              [self = std::move(self),                                        \
               out = std::move(out),                                          \
               param_setter = std::move(param_setter),                        \
               additional_predicate = std::move(additional_predicate)]() {}); \
        });                                                                   \
    MAYBE_FLUSH_OP(1);                                                        \
    return out;                                                               \
  }                                                                           \
  lazy_view_fallback_handle(self, out, param_setter, additional_predicate);   \
  return out;

#define RUN_VIEW_OP_MAYBE_WITH_ACC_THREAD(op, self, out, param_setter)      \
  if (habana_lazy::AccThread::Get().CanUseAccThread() &&                    \
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_ACC_VIEW_OPS_MODE) != 0) {               \
    PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
    habana_lazy::AccThread::Get().run([self, out, param_setter]() {         \
      lazy_view_fallback_handle(self, out, param_setter);                   \
      habana_lazy::AccThread::Get().PushCleanupTask(                        \
          [self_in = std::move(self),                                       \
           out = std::move(out),                                            \
           param_setter = std::move(param_setter)]() {});                   \
    });                                                                     \
    MAYBE_FLUSH_OP(1);                                                      \
    return out;                                                             \
  }                                                                         \
  lazy_view_fallback_handle(self, out, param_setter);                       \
  return out;

#define RUN_TENSOR_LIST_MAYBE_WITH_ACC_THREAD(op, lazy_op, tl1)             \
  if (habana_lazy::AccThread::Get().CanUseAccThread()) {                    \
    PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
    std::vector<at::Tensor> tensors_copy;                                   \
    std::copy(tl1.begin(), tl1.end(), std::back_inserter(tensors_copy));    \
    auto result = lazy_op.get_result();                                     \
    scheduleAccTask(std::move(lazy_op), result, std::move(tensors_copy));   \
    MAYBE_FLUSH_OP(1);                                                      \
    return result;                                                          \
  }                                                                         \
  return lazy_op.call();

#define RUN_TENSOR_LIST2_MAYBE_WITH_ACC_THREAD(op, lazy_op, tl1, tl2)       \
  if (habana_lazy::AccThread::Get().CanUseAccThread()) {                    \
    PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
    std::vector<at::Tensor> tensors_copy;                                   \
    std::copy(tl1.begin(), tl1.end(), std::back_inserter(tensors_copy));    \
    std::copy(tl2.begin(), tl2.end(), std::back_inserter(tensors_copy));    \
    auto result = lazy_op.get_result();                                     \
    scheduleAccTask(std::move(lazy_op), result, std::move(tensors_copy));   \
    MAYBE_FLUSH_OP(1);                                                      \
    return result;                                                          \
  }                                                                         \
  return lazy_op.call();

#define RUN_TENSOR_LIST_INPLACE_MAYBE_WITH_ACC_THREAD(op, lazy_op, result)     \
  if (habana_lazy::AccThread::Get().CanUseAccThread()) {                       \
    PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread");    \
    std::vector<at::Tensor> tensors_copy;                                      \
    std::copy(result.begin(), result.end(), std::back_inserter(tensors_copy)); \
    scheduleAccTask(std::move(lazy_op), std::move(tensors_copy));              \
    MAYBE_FLUSH_OP(1);                                                         \
    return;                                                                    \
  }                                                                            \
  return lazy_op.call(result);

#define FALLBACK_CHECK(fn, args...) bool fn(args...)
