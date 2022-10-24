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
#include "op_backend.h"
#include "pytorch_helpers/habana_helpers/kernels_accumulation.h"
#include "pytorch_helpers/habana_helpers/logging.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"

#include "supported_dtypes.h"

namespace habana {

template <class F, class... Ts, std::size_t... Is>
void for_each_in_tuple(
    std::tuple<Ts...>& tuple,
    F func,
    std::index_sequence<Is...>) {
  (void)(int[]){0, ((void)func(std::get<Is>(tuple)), 0)...};
}
template <class F, class... Ts>
void for_each_in_tuple(std::tuple<Ts...>& tuple, F func) {
  for_each_in_tuple(tuple, func, std::make_index_sequence<sizeof...(Ts)>());
}

template <class T, class InputType>
void scheduleAccTask(T&& lazy_op, InputType tensor) {
  habana_lazy::GetAccThreadPool().run(
      [op = std::move(lazy_op), tensor]() mutable {
        PT_LAZY_TRACE;
        op.call(tensor);
        habana_lazy::PushCleanupTask(
            [op = std::move(op), self = std::move(tensor)]() {});
      });
}

template <class T>
void scheduleAccTask(
    T&& lazy_op,
    std::vector<at::Tensor> result // pass explicit as copy to keep alive
) {
  habana_lazy::GetAccThreadPool().run(
      [op = std::move(lazy_op), result]() mutable {
        PT_LAZY_TRACE;
        op.call(result);
        habana_lazy::PushCleanupTask(
            [op = std::move(op), result = std::move(result)]() {});
      });
}

template <class T>
void scheduleAccTask(
    T&& lazy_op,
    std::vector<at::Tensor> result, // pass explicit as copy to keep alive
    std::vector<at::Tensor>&& tensor_list_copy) {
  habana_lazy::GetAccThreadPool().run(
      [op = std::move(lazy_op),
       result,
       tensor_list_copy = std::move(tensor_list_copy)]() mutable {
        PT_LAZY_TRACE;
        op.call(result);
        habana_lazy::PushCleanupTask(
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
  HABANA_ASSERT(tensors.size() <= 3, "Only tuples up to 3 are supported");
  habana_lazy::GetAccThreadPool().run(
      [op = std::move(lazy_op), tensors = std::move(tensors)]() mutable {
        PT_LAZY_TRACE;
        if (tensors.size() == 2) {
          op.call(std::tie(tensors[0], tensors[1]));
        } else if (tensors.size() == 3) {
          op.call(std::tie(tensors[0], tensors[1], tensors[2]));
        }
        habana_lazy::PushCleanupTask(
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

#define HPU_OP_FRONTEND(op)                                                   \
  template <typename T>                                                       \
  struct op : habana_lazy::LazyOp<T> {                                        \
    op(const std::string& qualstring,                                         \
       const std::vector<at::IValue>& inputs,                                 \
       const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn = {}); \
    T get_result_overrideable() override;                                     \
  };

#define HPU_OP_FRONTEND_WITH_TYPE_PROMOTION(op)                               \
  template <typename T>                                                       \
  struct op : habana_lazy::LazyOp<T> {                                        \
    op(const std::string& qualstring,                                         \
       const std::vector<at::IValue>& inputs,                                 \
       bool is_out_fn,                                                        \
       bool safe_cast_check,                                                  \
       const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn = {}); \
    T get_result_overrideable() override;                                     \
  };

#define FILL_PARAMS_DECL(fn) \
  std::shared_ptr<void> fn(const at::Stack&, size_t&);

#define OUTSHAPE_DECL(fn) sizes_vec fn(const at::Stack&);

#define HPU_SUPPORTED_DTYPES(dtypes, suffix...) \
  const static SupportedDtypes supported_dtypes_##suffix dtypes;

#define RUN_MAYBE_WITH_ACC_THREAD(op, lazy_op)                                \
  if (habana_lazy::CanUseAccThread()) {                                       \
    if (habana_lazy::IsAccumulationForAutogenSupported(#op)) {                \
      PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
      auto result = lazy_op.get_result();                                     \
      scheduleAccTask(std::move(lazy_op), result);                            \
      return result;                                                          \
    } else {                                                                  \
      habana_lazy::SyncAccThreadPool();                                       \
    }                                                                         \
  }                                                                           \
  return lazy_op.call();

#define RUN_MAYBE_WITH_ACC_THREAD_MODIFY_RESULT(op, lazy_op, result_func)     \
  if (habana_lazy::CanUseAccThread()) {                                       \
    if (habana_lazy::IsAccumulationForAutogenSupported(#op)) {                \
      PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
      auto result = lazy_op.get_result();                                     \
      result_func(result);                                                    \
      scheduleAccTask(std::move(lazy_op), result);                            \
      return result;                                                          \
    } else {                                                                  \
      habana_lazy::SyncAccThreadPool();                                       \
    }                                                                         \
  }                                                                           \
  auto result = lazy_op.call();                                               \
  result_func(result);                                                        \
  return result;

#define RUN_INPLACE_MAYBE_WITH_ACC_THREAD(op, lazy_op, self)                  \
  if (habana_lazy::CanUseAccThread()) {                                       \
    if (habana_lazy::IsAccumulationForAutogenSupported(#op)) {                \
      PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
      self = lazy_op.get_result(self);                                        \
      scheduleAccTask(std::move(lazy_op), self);                              \
      return self;                                                            \
    } else {                                                                  \
      habana_lazy::SyncAccThreadPool();                                       \
    }                                                                         \
  }                                                                           \
  return lazy_op.call(self);

#define RUN_TUPLE_MAYBE_WITH_ACC_THREAD(op, lazy_op)                          \
  if (habana_lazy::CanUseAccThread()) {                                       \
    if (habana_lazy::IsAccumulationForAutogenSupported(#op)) {                \
      PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
      auto tuple = lazy_op.get_result();                                      \
      scheduleAccTaskTuple(std::move(lazy_op), tuple);                        \
      return tuple;                                                           \
    } else {                                                                  \
      habana_lazy::SyncAccThreadPool();                                       \
    }                                                                         \
  }                                                                           \
  return lazy_op.call();

#define RUN_INPLACE_TUPLE_MAYBE_WITH_ACC_THREAD(op, lazy_op, tuple)           \
  if (habana_lazy::CanUseAccThread()) {                                       \
    if (habana_lazy::IsAccumulationForAutogenSupported(#op)) {                \
      PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
      tuple = lazy_op.get_result(tuple);                                      \
      scheduleAccTaskTuple(std::move(lazy_op), tuple);                        \
      return tuple;                                                           \
    } else {                                                                  \
      habana_lazy::SyncAccThreadPool();                                       \
    }                                                                         \
  }                                                                           \
  return lazy_op.call(tuple);

#define RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(op, func, out)                   \
  if (habana_lazy::CanUseAccThread()) {                                      \
    PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread");  \
    habana_lazy::GetAccThreadPool().run([func = std::move(func)]() mutable { \
      PT_LAZY_TRACE;                                                         \
      func();                                                                \
      habana_lazy::PushCleanupTask([func = std::move(func)]() {});           \
    });                                                                      \
  } else {                                                                   \
    func();                                                                  \
  }                                                                          \
  return out;

#define RUN_TENSOR_LIST_MAYBE_WITH_ACC_THREAD(op, lazy_op, tl1)               \
  if (habana_lazy::CanUseAccThread()) {                                       \
    if (habana_lazy::IsAccumulationForAutogenSupported(#op)) {                \
      PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
      std::vector<at::Tensor> tensors_copy;                                   \
      std::copy(tl1.begin(), tl1.end(), std::back_inserter(tensors_copy));    \
      auto result = lazy_op.get_result();                                     \
      scheduleAccTask(std::move(lazy_op), result, std::move(tensors_copy));   \
      return result;                                                          \
    } else {                                                                  \
      habana_lazy::SyncAccThreadPool();                                       \
    }                                                                         \
  }                                                                           \
  return lazy_op.call();

#define RUN_TENSOR_LIST2_MAYBE_WITH_ACC_THREAD(op, lazy_op, tl1, tl2)         \
  if (habana_lazy::CanUseAccThread()) {                                       \
    if (habana_lazy::IsAccumulationForAutogenSupported(#op)) {                \
      PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
      std::vector<at::Tensor> tensors_copy;                                   \
      std::copy(tl1.begin(), tl1.end(), std::back_inserter(tensors_copy));    \
      std::copy(tl2.begin(), tl2.end(), std::back_inserter(tensors_copy));    \
      auto result = lazy_op.get_result();                                     \
      scheduleAccTask(std::move(lazy_op), result, std::move(tensors_copy));   \
      return result;                                                          \
    } else {                                                                  \
      habana_lazy::SyncAccThreadPool();                                       \
    }                                                                         \
  }                                                                           \
  return lazy_op.call();

#define RUN_TENSOR_LIST_INPLACE_MAYBE_WITH_ACC_THREAD(op, lazy_op, result)    \
  if (habana_lazy::CanUseAccThread()) {                                       \
    if (habana_lazy::IsAccumulationForAutogenSupported(#op)) {                \
      PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
      std::vector<at::Tensor> tensors_copy;                                   \
      std::copy(                                                              \
          result.begin(), result.end(), std::back_inserter(tensors_copy));    \
      scheduleAccTask(std::move(lazy_op), tensors_copy);                      \
      return;                                                                 \
    } else {                                                                  \
      habana_lazy::SyncAccThreadPool();                                       \
    }                                                                         \
  }                                                                           \
  return lazy_op.call(result);

#define FALLBACK_CHECK(fn, args...) bool fn(args...)
