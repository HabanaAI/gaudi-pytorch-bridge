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
#include <ATen/core/Tensor.h>
#include <ATen/core/stack.h>
#include <perf_lib_layer_params.h>
#include "backend/synapse_helpers/device_helpers.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_helpers/kernels_accumulation.h"
#include "habana_helpers/logging.h"
#include "hpu_ops/op_backend.h" // IWYU pragma: keep for HPU_OP_BACKEND

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
  HABANA_ASSERT(tensors.size() <= 6, "Only tuples up to 6 are supported");
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
        } else if (tensors.size() == 6) {
          op.call(std::tie(
              tensors[0],
              tensors[1],
              tensors[2],
              tensors[3],
              tensors[4],
              tensors[5]));
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

inline std::string& update_guid_trunc_mode(
    std::string& guid,
    c10::ScalarType dtype) {
  const std::string trunc_str = "_trunc";
  // remove _trunc if dtype promoted to other dtypes
  std::string::size_type index = guid.find(trunc_str);
  if (index != std::string::npos && dtype != c10::ScalarType::Char &&
      dtype != c10::ScalarType::Byte) {
    guid.erase(index, trunc_str.length());
    return guid;
  }

  using namespace std::literals;
  static const absl::flat_hash_set<std::string_view> guids_support_trunc = {
      "mult"sv,
      "mult_fwd"sv,
  };

  auto device_type{habana::HPURegistrar::get_device().type()};
  if (synapse_helpers::device_supports_trunc(device_type) &&
      guids_support_trunc.contains(guid) &&
      (dtype == c10::ScalarType::Char || dtype == c10::ScalarType::Byte)) {
    if (guid.find_first_of('_') != std::string::npos) {
      guid.insert(guid.find_first_of('_'), trunc_str);
    } else {
      guid.append(trunc_str);
    }
  }

  return guid;
}

inline std::string& update_guid_dtype(
    std::string& guid,
    std::string_view dtype_str) {
  guid = guid.substr(0, guid.find_last_of('_') + 1).append(dtype_str);
  return guid;
}

inline std::string& update_guid_dtype(
    std::string& guid,
    c10::ScalarType dtype) {
  return update_guid_dtype(
      guid,
      synapse_helpers::graph::name_suffix_from_type(
          habana_helpers::pytorch_to_synapse_type(dtype)));
}

inline std::string& update_div_guid_with_precise(
    std::string& guid,
    bool has_rounding = false) {
  const std::string_view div{"div"};
  if (guid.rfind(div, 0) != 0) {
    // Not start with div, no update
    return guid;
  }
  // Use precise when rounding is used or the flag is explicitly set
  bool use_precise = has_rounding;
  if (IS_ENV_FLAG_DEFINED_NEW(PT_HPU_ENABLE_DIV_PRECISE)) {
    use_precise = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_DIV_PRECISE);
  }
  const std::string_view div_precise{"div_precise"};
  bool precise_used = (guid.rfind(div_precise, 0) == 0);
  if (use_precise) {
    if (!precise_used) {
      guid = guid.replace(0, div.size(), div_precise);
    }
  } else {
    if (precise_used) {
      guid = guid.replace(0, div_precise.size(), div);
    }
  }
  return guid;
}

inline int get_dim_in_tpc_order(int64_t dim_, int64_t max_dims) {
  auto dim = at::maybe_wrap_dim(dim_, max_dims, /*wrap_scalar=*/true);
  return std::max(static_cast<int>(max_dims - dim - 1), 0);
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

template <int type_promotion_kind, bool broadcast, int... indices>
OutputMetaDataVector PointwiseMeta(const at::Stack& stack) {
  OutputMetaData meta{};
  at::Stack inputs;
  inputs.reserve(sizeof...(indices));

  bool first = true;
  for (int i : {indices...}) {
    inputs.emplace_back(stack[i]);
    if (stack[i].isTensor()) {
      if (first) {
        meta.shape = stack_tensor(stack, i).sizes().vec();
        first = false;
      } else if (broadcast) {
        meta.shape = at::infer_size(meta.shape, stack_tensor(stack, i).sizes());
      }
    }
  }

  meta.dtype = habana_helpers::DTypeHelper::get_compute_dtype(
      inputs,
      c10::nullopt,
      static_cast<habana_helpers::DTypeHelper::DtypePromoteVariant>(
          type_promotion_kind),
      false,
      c10::nullopt,
      true,
      true);

  return {meta};
}

enum TensorDataIdx { TENSOR_IDX = 0, SHAPE_IDX, STORAGE_IDX };

template <unsigned I>
struct TensorDataGetter {
  auto operator()(synapse_helpers::tensor& arg) {
    if constexpr (I == TENSOR_IDX) {
      return arg.get();
    } else if constexpr (I == SHAPE_IDX) {
      return arg.pt_shape();
    } else if constexpr (I == STORAGE_IDX) {
      return std::variant<synapse_helpers::tensor*, int>{&arg};
    }
  }
  auto operator()(const OpBackend::TensorsPair& arg) {
    if constexpr (I == TENSOR_IDX) {
      return arg.syn_t;
    } else if constexpr (I == SHAPE_IDX) {
      return arg.pt_t.sizes().vec();
    } else if constexpr (I == STORAGE_IDX) {
      return std::variant<synapse_helpers::tensor*, int>{arg.syn_idx};
    }
  }
};

template <unsigned... Is>
auto get_or_create_tensor(
    OpBackend& op,
    synapse_helpers::graph& graph,
    const c10::optional<OpBackend::TensorsPair>& tensor,
    const c10::IntArrayRef& size,
    const c10::ScalarType& scalar_type,
    const at::Scalar& val,
    std::optional<synapse_helpers::tensor>& tensorStorageOpt) {
  if (not tensor.has_value()) {
    tensorStorageOpt = op.BuildConstant(&op, graph, val, scalar_type, size);
    return std::make_tuple(TensorDataGetter<Is>{}(*tensorStorageOpt)...);
  }
  return std::make_tuple(TensorDataGetter<Is>{}(*tensor)...);
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
#define SHARED_LAYER_META_DECL(fn) SharedMetaDataVector fn(const at::Stack&);
#define STMETA_DECL(fn)                   \
  bool fn(                                \
      habana_helpers::IShapeList& inputs, \
      habana_helpers::IShapeList& outputs);

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
  self = lazy_op.get_result(self);                                          \
  if (habana_lazy::AccThread::Get().CanUseAccThread()) {                    \
    PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
    scheduleAccTask(std::move(lazy_op), self);                              \
    MAYBE_FLUSH_OP(1);                                                      \
    return self;                                                            \
  }                                                                         \
  return lazy_op.call(self);

#define RUN_CONST_INPLACE_MAYBE_WITH_ACC_THREAD(op, lazy_op, self)          \
  lazy_op.get_result(self);                                                 \
  if (habana_lazy::AccThread::Get().CanUseAccThread()) {                    \
    PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
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

#define RUN_WITH_PREDICATE_VIEW_OP_MAYBE_WITH_ACC_THREAD(                   \
    op, self, out, param_setter, additional_predicate)                      \
  if (habana_lazy::AccThread::Get().CanUseAccThread() &&                    \
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_ACC_VIEW_OPS_MODE) != 0) {               \
    PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
    habana_lazy::AccThread::Get().run(                                      \
        [self, out, param_setter, additional_predicate]() {                 \
          lazy_view_fallback_handle(                                        \
              self, out, param_setter, additional_predicate);               \
          habana_lazy::AccThread::Get().PushCleanupTask(                    \
              [self = std::move(self), out = std::move(out)]() {            \
                /* Silence lambda capture not used. */                      \
                (void)self;                                                 \
                (void)out;                                                  \
              });                                                           \
        });                                                                 \
    MAYBE_FLUSH_OP(1);                                                      \
    return out;                                                             \
  }                                                                         \
  lazy_view_fallback_handle(self, out, param_setter, additional_predicate); \
  return out;

#define RUN_VIEW_OP_MAYBE_WITH_ACC_THREAD(op, self, out, param_setter)      \
  if (habana_lazy::AccThread::Get().CanUseAccThread() &&                    \
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_ACC_VIEW_OPS_MODE) != 0) {               \
    PT_LAZY_PARALLEL_ACC_DEBUG("Running ", #op, " in accumulation thread"); \
    habana_lazy::AccThread::Get().run([self, out, param_setter]() {         \
      lazy_view_fallback_handle(self, out, param_setter);                   \
      habana_lazy::AccThread::Get().PushCleanupTask(                        \
          [self_in = std::move(self), out = std::move(out)]() {             \
            /* Silence lambda capture not used warning */                   \
            (void)self_in;                                                  \
            (void)out;                                                      \
          });                                                               \
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
