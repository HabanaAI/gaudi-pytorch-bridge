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

#include <tuple>
#include <utility>

#include "habana_kernels/kernel_utils.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hpu_lazy_cache.h"
#include "habana_lazy/lazy_executor.h"
#include "lazy_kernels_declarations.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"
#include "resize.h"

namespace habana_lazy {
void AddMemcpy(const at::Tensor& src, at::Tensor& dst);
void updateDstDependencies(
    habana_lazy::HbLazyTensor& hl_dst,
    const at::Tensor& dst,
    bool in_place = false);

bool HandleViews(const at::Tensor& t, const habana_lazy::HbLazyTensor& hl_t);
habana_lazy::HbLazyTensor HandleViewsOrUpdate(
    const at::Tensor& t,
    habana_lazy::HbLazyTensor& hl_t);
at::Tensor HandleViewsD2H(const at::Tensor& t);
bool HandleViewsD2D(const at::Tensor& src, const at::Tensor& dst);
std::vector<at::Tensor> HandleViewsTensorList(const at::TensorList&);
at::Tensor add_strided_view_node(
    const at::Tensor& self,
    at::IntArrayRef size_in,
    at::IntArrayRef stride_in,
    int64_t storage_offset,
    bool is_update_view,
    c10::optional<at::Tensor> out);
void updateViewTable(HbLazyTensor& hl_view_t, StrideParams& params);
at::Tensor get_parent_tensor(const at::Tensor& self);
const at::Tensor& get_recent_base_tensor(const at::Tensor& self);

void flushWithMarkStep();

at::Tensor get_tensor_for_scalar(
    float alpha,
    const at::TensorOptions& options = {});

void flush_op(at::TensorList tensors, size_t lazy_eager_key = 0);

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

template <class...>
struct conjunction : std::true_type {};

template <class B1>
struct conjunction<B1> : B1 {};

template <class B1, class... Bn>
struct conjunction<B1, Bn...>
    : std::conditional_t<bool(B1::value), conjunction<Bn...>, B1> {};

template <typename Tuple>
struct is_tuple_of_tensor_ref;

template <typename... Ts>
struct is_tuple_of_tensor_ref<std::tuple<Ts...>>
    : conjunction<std::is_same<at::Tensor&, Ts>...> {};

// TODO: Ideally we want a variant of HABANA_ASSERT like
// TORCH_INTERNAL_ASSERT_DEBUG_ONLY

template <typename ReturnType, typename NodeConstruct = void>
class LazyOp {
 public:
  explicit LazyOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      std::set<size_t> metadata_indices = {},
      std::vector<std::vector<int64_t>> out_shapes = {},
      int out_index = 0) noexcept
      : m_symbol{at::Symbol::fromQualString(qualstring)},
        m_metadata_indices{std::move(metadata_indices)},
        m_out_shapes{std::move(out_shapes)},
        m_out_index{out_index} {
    set_inputs(inputs);
  }

  explicit LazyOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      std::vector<std::vector<int64_t>> out_shapes) noexcept
      : m_symbol{at::Symbol::fromQualString(qualstring)},
        m_metadata_indices{},
        m_out_shapes{std::move(out_shapes)},
        m_out_index{} {
    set_inputs(inputs);
  }

  explicit LazyOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      const std::function<
          std::vector<std::vector<int64_t>>(const at::Stack&, bool)>&
          out_shapes_fn,
      int out_index = 0) noexcept
      : m_symbol{at::Symbol::fromQualString(qualstring)},
        m_metadata_indices{},
        m_out_index{out_index} {
    if (out_shapes_fn) {
      m_out_shapes = out_shapes_fn(inputs, false);
    }
    set_inputs(inputs);
  }

  explicit LazyOp(
      ir::NodePtr node,
      const std::vector<at::IValue>& inputs,
      std::vector<std::vector<int64_t>> out_shapes = {},
      int out_index = 0)
      : m_node{std::move(node)},
        m_out_shapes{std::move(out_shapes)},
        m_out_index{out_index} {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        std::is_class<NodeConstruct>::value,
        "This constructor is valid only when NodeConstruct is a class.");
    set_inputs(inputs);
  }

  explicit LazyOp(
      ir::NodePtr node,
      const std::vector<at::IValue>& inputs,
      std::set<size_t> metadata_indices,
      std::vector<std::vector<int64_t>> out_shapes = {},
      int out_index = 0)
      : m_node{std::move(node)},
        m_metadata_indices{std::move(metadata_indices)},
        m_out_shapes{std::move(out_shapes)},
        m_out_index{out_index} {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        std::is_class<NodeConstruct>::value,
        "This constructor is valid only when NodeConstruct is a class.");
    set_inputs(inputs);
  }

  explicit LazyOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      const at::TensorList& output_meta_tensors) noexcept
      : m_symbol{at::Symbol::fromQualString(qualstring)},
        m_out_index{},
        m_out_meta_tensors{output_meta_tensors} {
    set_inputs(inputs);

    for (const auto& out : m_out_meta_tensors) {
      m_out_shapes.emplace_back(out.sizes().vec());
    }
  }

  explicit LazyOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      std::vector<std::vector<int64_t>> out_shapes,
      const c10::ScalarType scalar_type) noexcept
      : m_symbol{at::Symbol::fromQualString(qualstring)},
        m_metadata_indices{},
        m_out_shapes{std::move(out_shapes)},
        m_out_index{},
        m_scalar_type(scalar_type) {
    set_inputs(inputs);
  }

  virtual ~LazyOp() = default;

  template <typename T = ReturnType>
  typename std::enable_if<not is_tuple_of_tensor_ref<T>::value, T>::type
  HandleLazy(size_t lazy_eager_key = 0) {
    auto node = create_node();
    auto results = get_result();
    int i = 0;
    std::vector<at::Tensor> tensors;
    tensors.reserve(std::tuple_size<T>::value);
    for_each_in_tuple(results, [&node, &i, &tensors](const auto& result) {
      auto hl_result = GetHbLazyTensor(result);
      tensors.push_back(result);
      ir::Value& out = hl_result.CurrentIrValue();
      out.SetNode(
          node,
          hl_result.GetDevice(),
          hl_result.GetSizes(),
          hl_result.dtype_optional(),
          i++);
      updateDstDependencies(hl_result, result, false);
    });
    flush_op(tensors, lazy_eager_key);
    return results;
  }

  template <typename T = ReturnType>
  typename std::enable_if<not is_tuple_of_tensor_ref<T>::value, T>::type
  HandleOptimizedLazyEager(size_t lazy_eager_key) {
    PT_LAZY_DEBUG("Optimized Lazy Eager Path Chosen");
    auto results = get_result();
    int i = 0;
    std::vector<HbLazyTensor> hl_tensors;
    for_each_in_tuple(results, [&i, &hl_tensors](const auto& result) {
      auto hl_result = GetHbLazyTensor(result);
      hl_tensors.push_back(hl_result);
      i++;
    });
    std::vector<ir::Value> input_values = prepare_lazy_eager_input_values();
    HbLazyTensor::SyncTensorsGraphFast(
        &hl_tensors, input_values, lazy_eager_key);
    return results;
  }

  template <typename T = ReturnType>
  typename std::enable_if<not is_tuple_of_tensor_ref<T>::value, T>::type call() {
    PT_LAZY_DEBUG("Lazy Call not_Tuple_Of_Tensor :: ", m_symbol.toQualString());
    viewUpdateInputs();
    size_t lazy_eager_key = 0;

    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2 &&
        GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_OPTIM_CACHE) == 1) {
      bool IsOptimizedLazyEagerCached =
          calculate_key_and_check_optimized_lazy_eager_cache(lazy_eager_key);
      if (IsOptimizedLazyEagerCached) {
        return HandleOptimizedLazyEager(lazy_eager_key);
      }
    }

    return HandleLazy(lazy_eager_key);
  }

  template <typename T = ReturnType>
  typename std::enable_if<is_tuple_of_tensor_ref<T>::value, T>::type HandleLazy(
      T results,
      size_t lazy_eager_key = 0) {
    const auto& node = create_node();
    int i = 0;
    std::vector<at::Tensor> tensors;
    tensors.reserve(std::tuple_size<T>::value);
    const auto& out_shapes = m_out_shapes;
    auto context = habana_lazy_executor.getDeviceExecutionContext();

    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        out_shapes.size() == std::tuple_size<T>::value);

    for_each_in_tuple(
        results,
        [&node, &i, &tensors, out_shapes, context](const auto& result) {
          auto hl_result = GetHbLazyTensor(result);
          tensors.push_back(result);
          updateDstDependencies(hl_result, result, true);
          ir::Value& out = hl_result.CurrentIrValue();
          out.SetNode(
              node,
              hl_result.GetDevice(),
              hl_result.GetSizes(),
              hl_result.dtype_optional(),
              i);
          const auto& out_shape = out_shapes.at(i);
          if (result.sizes() != out_shape) {
            auto impl = hl_result.getAttachedTensorImpl();
            THHTensor_resizeNd(
                impl, out_shape.size(), out_shape.data(), nullptr);
            result.unsafeGetTensorImpl()->set_sizes_contiguous(out_shape);
          }
          context->MarkTensorStatus(
              hl_result.getTensorUniqueId(),
              LazyTensorExecutionStatus::kREGISTERED);
          ++i;
        });
    flush_op(tensors, lazy_eager_key);
    return results;
  }

  template <typename T = ReturnType>
  typename std::enable_if<is_tuple_of_tensor_ref<T>::value, T>::type
  HandleOptimizedLazyEager(T results, size_t lazy_eager_key) {
    PT_LAZY_DEBUG("Optimized Lazy Eager Path Chosen");
    int i = 0;
    const auto& out_shapes = m_out_shapes;
    auto context = habana_lazy_executor.getDeviceExecutionContext();

    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        out_shapes.size() == std::tuple_size<T>::value);

    std::vector<HbLazyTensor> hl_tensors;
    for_each_in_tuple(
        results, [&i, &hl_tensors, out_shapes, context](const auto& result) {
          auto hl_result = GetHbLazyTensor(result);
          hl_tensors.push_back(hl_result);
          const auto& out_shape = out_shapes.at(i);
          if (result.sizes() != out_shape) {
            auto impl = hl_result.getAttachedTensorImpl();
            THHTensor_resizeNd(
                impl, out_shape.size(), out_shape.data(), nullptr);
            result.unsafeGetTensorImpl()->set_sizes_contiguous(out_shape);
          }
          context->MarkTensorStatus(
              hl_result.getTensorUniqueId(),
              LazyTensorExecutionStatus::kREGISTERED);
          ++i;
        });
    std::vector<ir::Value> input_values = prepare_lazy_eager_input_values();
    HbLazyTensor::SyncTensorsGraphFast(
        &hl_tensors, input_values, lazy_eager_key);
    return results;
  }

  template <typename T = ReturnType>
  typename std::enable_if<is_tuple_of_tensor_ref<T>::value, T>::type call(
      T results) {
    PT_LAZY_DEBUG("Lazy Call Tuple_Of_Tensor :: ", m_symbol.toQualString());
    viewUpdateInputs();
    size_t lazy_eager_key = 0;

    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2 &&
        GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_OPTIM_CACHE) == 1) {
      bool IsOptimizedLazyEagerCached =
          calculate_key_and_check_optimized_lazy_eager_cache(lazy_eager_key);

      if (IsOptimizedLazyEagerCached) {
        return HandleOptimizedLazyEager(results, lazy_eager_key);
      }
    }

    return HandleLazy(results, lazy_eager_key);
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_fundamental<T>::value, T>::type call() {
    viewUpdateInputs();

    const auto& node = create_node();
    const auto& t = get_inputs().at(m_out_index).toTensor();
    const auto& result =
        empty_hpu_lazy(1, t.options(), t.suggest_memory_format(), false);
    auto hl_result = GetHbLazyTensor(result);
    ir::Value& out = hl_result.CurrentIrValue();
    out.SetNode(
        node,
        hl_result.GetDevice(),
        hl_result.GetSizes(),
        hl_result.dtype_optional());
    updateDstDependencies(hl_result, result, false);

    return result.item().template to<T>();
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor>::value, T>::type
  HandleLazy(size_t lazy_eager_key = 0) {
    const auto& node = create_node();
    const auto& result = get_result();
    auto hl_result = GetHbLazyTensor(result);
    ir::Value& out = hl_result.CurrentIrValue();
    out.SetNode(
        node,
        hl_result.GetDevice(),
        hl_result.GetSizes(),
        hl_result.dtype_optional());
    updateDstDependencies(hl_result, result, false);
    flush_op(result, lazy_eager_key);
    return result;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor>::value, T>::type
  HandleOptimizedLazyEager(size_t lazy_eager_key) {
    PT_LAZY_DEBUG("Optimized Lazy Eager Path Chosen");
    const auto& result = get_result();
    auto hl_result = GetHbLazyTensor(result);
    std::vector<ir::Value> input_values = prepare_lazy_eager_input_values();
    std::vector<HbLazyTensor> hl_tensors = {hl_result};
    HbLazyTensor::SyncTensorsGraphFast(
        &hl_tensors, input_values, lazy_eager_key);
    return result;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor>::value, T>::type call() {
    PT_LAZY_DEBUG("Lazy Call :: ", m_symbol.toQualString());
    viewUpdateInputs();
    size_t lazy_eager_key = 0;

    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2 &&
        GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_OPTIM_CACHE) == 1) {
      bool IsOptimizedLazyEagerCached =
          calculate_key_and_check_optimized_lazy_eager_cache(lazy_eager_key);

      if (IsOptimizedLazyEagerCached) {
        return HandleOptimizedLazyEager(lazy_eager_key);
      }
    }

    return HandleLazy(lazy_eager_key);
  }

  bool is_inplace(at::Symbol symbol) {
    bool is_inplace = false;

    auto node_name = symbol.toQualString();
    /*
  Since as_strided_lazy is now out of place op, we need a control edge to create
  new tensor for fill to avoid GC error
  %5 : Float(*, requires_grad=0,
  device=hpu:0) = hpu::as_strided_lazy(%id:3, %2, %3, %4)
  %6 : Float(*, requires_grad=0, device=hpu:0) = aten::fill_(%5, %1)
  */

    if (strcmp(node_name, "aten::fill_")) {
      size_t len = strlen(node_name);
      char endch = node_name[len - 1];

      if (endch == '_') {
        is_inplace = true;
      }
    }
    return is_inplace;
  }

  void viewUpdateInputs() {
    auto context = habana_lazy_executor.getDeviceExecutionContext();
    size_t idx = 0;
    for (auto ival : m_inputs) {
      if (ival.isTensor()) {
        auto t = ival.toTensor();
        if (t.defined() && (t.device().type() == c10::DeviceType::HPU)) {
          auto hl_t = GetHbLazyTensor(t);

          // if it is base tensor, use the most recent version else check if
          // it is a view
          auto id = hl_t.getTensorUniqueId();
          if (context->orig_tensor_map.find(id) !=
              context->orig_tensor_map.end()) {
            m_inputs[idx] = context->orig_tensor_map[id];
          } else {
            HandleViews(t, hl_t);
          }
        }
      }
      idx++;
    }
  }

  void viewUpdateInputsInplace() {
    auto context = habana_lazy_executor.getDeviceExecutionContext();
    size_t idx = 0;
    for (auto ival : m_inputs) {
      if (ival.isTensor()) {
        auto t = ival.toTensor();
        if (t.defined() && (t.device().type() == c10::DeviceType::HPU)) {
          auto hl_t = GetHbLazyTensor(t);

          // if it is base tensor, use the most recent version else check if
          // it is a view
          auto id = hl_t.getTensorUniqueId();
          if (context->orig_tensor_map.find(id) !=
              context->orig_tensor_map.end()) {
            m_inputs[idx] = context->orig_tensor_map[id];
          } else {
            if (is_inplace(m_symbol)) {
              HandleViews(t, hl_t);
            }
          }
        }
      }
      idx++;
    }
  }

  void HandleViewsInplace(
      const at::Tensor& self,
      habana_lazy::HbLazyTensor& hl_self) {
    auto out_t = empty_hpu_lazy(
        self.sizes(), self.options(), self.suggest_memory_format(), false);

    // optimization for 8x mul_out case. The below logic avoids extra out of
    // place as_strided_lazy call
    // TODO ideally we should also replace inplace op with out of place
    // variant.
    if (!is_inplace(m_symbol)) {
      // out variant needs storage as it is a graph input
      out_t = empty_hpu_lazy(
          self.sizes(), self.options(), self.suggest_memory_format(), true);
      for (size_t idx = 0; idx < m_inputs.size(); idx++) {
        auto t = m_inputs[idx];
        if (t.isTensor() && t.toTensor().is_same(self)) {
          m_inputs[idx] = out_t;
        }
      }
    }

    const auto& node = create_node();

    hl_self = GetHbLazyTensor(out_t);
    ir::Value& out = hl_self.CurrentIrValue();
    out.SetNode(
        node,
        hl_self.GetDevice(),
        hl_self.GetSizes(),
        hl_self.dtype_optional());

    // add strided insert node and update most recent version of original
    // tensor
    flush_op(out_t);
    strided_insert_hpu_lazy(self, out_t);
  }

  // For inplace/out variants
  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor&>::value, T>::type
  HandleLazy(at::Tensor& self, size_t lazy_eager_key = 0) {
    auto context = habana_lazy_executor.getDeviceExecutionContext();
    auto hl_self = GetHbLazyTensor(self);

    auto id = hl_self.getTensorUniqueId();
    auto is_self_view =
        context->view_table.find(id) != context->view_table.end();

    // Handle views or fetch updated tensor for all the inputs
    viewUpdateInputsInplace();

    // special handling for self tensor
    if (is_self_view == false) {
      // use most recent version of the tensor if applicable
      auto self_updated = get_recent_base_tensor(self);
      hl_self = GetHbLazyTensor(self_updated);

      // identify the inplace index and replace it with updated version
      // m_inputs will be used in create_node()
      for (size_t idx = 0; idx < m_inputs.size(); idx++) {
        auto t = m_inputs[idx];
        if (t.isTensor() && t.toTensor().is_same(self)) {
          m_inputs[idx] = self_updated;
        }
      }
      // special handling for self tensor
      // skip ctrl edges for inplace
      // TODO do the same for out variants
      if (!is_inplace(m_symbol)) {
        updateDstDependencies(hl_self, self_updated, true);
      }
      const auto& node = create_node();
      ir::Value& out = hl_self.CurrentIrValue();
      out.SetNode(
          node,
          hl_self.GetDevice(),
          hl_self.GetSizes(),
          hl_self.dtype_optional());
    } else {
      HandleViewsInplace(self, hl_self);
    }

    // numel == 0 is the correct check, need the size check until pytorch fixes
    // it properly
    // https://github.com/pytorch/pytorch/wiki/Developer-FAQ#how-does-out-work-in-pytorch
    auto out_shape = m_out_shapes.empty()
        ? get_inputs().at(m_out_index).toTensor().sizes().vec()
        : m_out_shapes[0];
    if (self.sizes() != out_shape) {
      auto impl = hl_self.getAttachedTensorImpl();
      THHTensor_resizeNd(impl, out_shape.size(), out_shape.data(), nullptr);
      self.unsafeGetTensorImpl()->set_sizes_contiguous(out_shape);
    }

    context->MarkTensorStatus(
        hl_self.getTensorUniqueId(), LazyTensorExecutionStatus::kREGISTERED);

    flush_op(self, lazy_eager_key);
    return self;
  }

  // For inplace/out variants
  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor&>::value, T>::type
  HandleOptimizedLazyEager(at::Tensor& self, size_t lazy_eager_key) {
    PT_LAZY_DEBUG("Optimized Lazy Eager Path Chosen");
    auto hl_self = GetHbLazyTensor(self);
    // numel == 0 is the correct check, need the size check until pytorch
    // fixes
    // it properly
    // https://github.com/pytorch/pytorch/wiki/Developer-FAQ#how-does-out-work-in-pytorch
    auto out_shape = m_out_shapes.empty()
        ? get_inputs().at(m_out_index).toTensor().sizes().vec()
        : m_out_shapes[0];
    if (self.sizes() != out_shape) {
      auto impl = hl_self.getAttachedTensorImpl();
      THHTensor_resizeNd(impl, out_shape.size(), out_shape.data(), nullptr);
      self.unsafeGetTensorImpl()->set_sizes_contiguous(out_shape);
    }
    auto context = habana_lazy_executor.getDeviceExecutionContext();
    context->MarkTensorStatus(
        hl_self.getTensorUniqueId(), LazyTensorExecutionStatus::kREGISTERED);
    std::vector<ir::Value> input_values = prepare_lazy_eager_input_values();
    std::vector<HbLazyTensor> hl_tensors = {hl_self};
    HbLazyTensor::SyncTensorsGraphFast(
        &hl_tensors, input_values, lazy_eager_key);
    return self;
  }

  // For inplace/out variants
  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor&>::value, T>::type call(
      at::Tensor& self) {
    PT_LAZY_DEBUG("Lazy Call Inplace:self :: ", m_symbol.toQualString());
    size_t lazy_eager_key = 0;

    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2 &&
        GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_OPTIM_CACHE) == 1) {
      bool IsOptimizedLazyEagerCached =
          calculate_key_and_check_optimized_lazy_eager_cache(lazy_eager_key);

      if (IsOptimizedLazyEagerCached) {
        return HandleOptimizedLazyEager(self, lazy_eager_key);
      }
    }

    return HandleLazy(self, lazy_eager_key);
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, const at::Tensor&>::value, T>::type
  HandleLazy(size_t lazy_eager_key = 0) {
    const at::Tensor& self = get_inputs().at(0).toTensor();
    auto hl_self = GetHbLazyTensor(self);
    const auto& node = create_node();
    ir::Value& out = hl_self.CurrentIrValue();
    out.SetNode(
        node,
        hl_self.GetDevice(),
        hl_self.GetSizes(),
        hl_self.dtype_optional());
    updateDstDependencies(hl_self, self, false);

    auto out_shape = m_out_shapes.empty()
        ? get_inputs().at(m_out_index).toTensor().sizes().vec()
        : m_out_shapes[0];
    if (self.sizes() != out_shape) {
      auto impl = hl_self.getAttachedTensorImpl();
      THHTensor_resizeNd(impl, out_shape.size(), out_shape.data(), nullptr);
      self.unsafeGetTensorImpl()->set_sizes_contiguous(out_shape);
    }

    auto context = habana_lazy_executor.getDeviceExecutionContext();
    context->MarkTensorStatus(
        hl_self.getTensorUniqueId(), LazyTensorExecutionStatus::kREGISTERED);
    flush_op(self, lazy_eager_key);
    return self;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, const at::Tensor&>::value, T>::type
  HandleOptimizedLazyEager(size_t lazy_eager_key) {
    PT_LAZY_DEBUG("Optimized Lazy Eager Path Chosen");
    const at::Tensor& self = get_inputs().at(0).toTensor();
    auto hl_self = GetHbLazyTensor(self);

    auto out_shape = m_out_shapes.empty()
        ? get_inputs().at(m_out_index).toTensor().sizes().vec()
        : m_out_shapes[0];
    if (self.sizes() != out_shape) {
      auto impl = hl_self.getAttachedTensorImpl();
      THHTensor_resizeNd(impl, out_shape.size(), out_shape.data(), nullptr);
      self.unsafeGetTensorImpl()->set_sizes_contiguous(out_shape);
    }

    auto context = habana_lazy_executor.getDeviceExecutionContext();
    context->MarkTensorStatus(
        hl_self.getTensorUniqueId(), LazyTensorExecutionStatus::kREGISTERED);
    std::vector<ir::Value> input_values = prepare_lazy_eager_input_values();
    std::vector<HbLazyTensor> hl_tensors = {hl_self};
    HbLazyTensor::SyncTensorsGraphFast(
        &hl_tensors, input_values, lazy_eager_key);
    return self;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, const at::Tensor&>::value, T>::type
  call() {
    PT_LAZY_DEBUG("Lazy Call Inplace :: ", m_symbol.toQualString());
    size_t lazy_eager_key = 0;

    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2 &&
        GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_OPTIM_CACHE) == 1) {
      bool IsOptimizedLazyEagerCached =
          calculate_key_and_check_optimized_lazy_eager_cache(lazy_eager_key);

      if (IsOptimizedLazyEagerCached) {
        return HandleOptimizedLazyEager(lazy_eager_key);
      }
    }

    return HandleLazy(lazy_eager_key);
  }

  // wrapped_scalar_tensor in ATen/native/BinaryOps.cpp
  void ConvertWrappedTensorToScalar() {
    m_convert_wrapped_tensor_to_scalar = true;
  }

  bool IsConvertWrappedTensorToScalar() {
    return m_convert_wrapped_tensor_to_scalar;
  }

 private:
  bool isMetadataCandidate(const at::IValue& input) const {
    return input.isBool() || input.isDevice() || input.isIntList() ||
        input.isDoubleList() || input.isBoolList() || input.isString() ||
        input.isNone();
  }

  template <typename T = ReturnType>
  typename std::enable_if<not is_tuple_of_tensor_ref<T>::value, ReturnType>::
      type
      get_result() {
    // Get results from derived class when index is negative
    if (m_out_index < 0) {
      return get_result_overrideable();
    }
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        std::tuple_size<T>::value == m_out_shapes.size());

    unsigned i = 0;
    ReturnType results;

    for_each_in_tuple(results, [&](auto& result) {
      auto t = get_inputs().at(m_out_index).toTensor();
      result = empty_hpu_lazy(
          m_out_shapes[i++], t.options(), t.suggest_memory_format(), false);
    });
    return results;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor>::value, T>::type
  get_result() {
    // Get results from derived class when index is negative
    if (m_out_index < 0) {
      return get_result_overrideable();
    }

    if (m_out_meta_tensors.empty()) {
      const auto& t = get_inputs().at(m_out_index).toTensor();
      const auto& out_shape =
          m_out_shapes.empty() ? t.sizes() : m_out_shapes[0];
      if (m_scalar_type != c10::ScalarType::Undefined) {
        return empty_hpu_lazy(
            out_shape,
            t.options().dtype(m_scalar_type),
            t.suggest_memory_format(),
            false);
      } else {
        return empty_hpu_lazy(
            out_shape, t.options(), t.suggest_memory_format(), false);
      }
    }

    const auto& t = m_out_meta_tensors[0];
    if (m_scalar_type != c10::ScalarType::Undefined) {
      return empty_hpu_lazy(
          t.sizes(),
          t.options().dtype(m_scalar_type),
          t.suggest_memory_format(),
          false);
    } else {
      return empty_hpu_lazy(
          t.sizes(), t.options(), t.suggest_memory_format(), false);
    }
  }

  template <typename N = NodeConstruct>
  std::enable_if_t<std::is_class<N>::value, ir::NodePtr> create_node() {
    return m_node;
  }

  template <typename N = NodeConstruct>
  std::enable_if_t<!std::is_class<N>::value, ir::NodePtr> create_node() {
    ir::ValueList values;
    std::vector<at::Tensor> input_pt_vec;
    ir::MetaData metadata;

    for (size_t i = 0; i < m_inputs.size(); ++i) {
      const at::IValue& input = m_inputs[i];
      if (m_metadata_indices.count(i)) {
        metadata.set(input, i);
        continue;
      }

      if (input.isScalar()) {
        auto val = GetIrValueForScalar(input.toScalar());
        values.emplace_back(val);
      } else if (isMetadataCandidate(input)) {
        metadata.set(input, i);
      } else if (input.isTensor()) {
        const at::Tensor& t = input.toTensor();
        if (t.defined()) {
          HABANA_ASSERT(t.device().type() == c10::DeviceType::HPU)
          auto val = GetHbLazyTensor(t).GetIrValue();
          values.emplace_back(val);
          input_pt_vec.emplace_back(t);
        } else {
          metadata.set(torch::jit::IValue(), i);
        }
      } else if (input.isList()) {
        const auto& list = input.toListRef();
        ir::ValueList opt_tensors;
        std::vector<at::Tensor> list_input_pt_vec;
        bool is_optional = false;

        for (const auto& li : list) {
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
              li.isNone() or li.isTensor(),
              "Got unhandled list item type: ",
              li.tagKind(),
              " for ",
              m_symbol.toQualString(),
              " at index ",
              i,
              ".");
          if (li.isNone()) {
            opt_tensors.emplace_back(GetIrValueForNone());
            is_optional |= true;
          } else {
            const auto& t = li.toTensor();
            opt_tensors.emplace_back(GetHbLazyTensor(t).GetIrValue());
            list_input_pt_vec.emplace_back(t);
          }
        }

        const auto& list_input =
            GetIrValueForListConstruct(opt_tensors, is_optional);
        list_input.mp_node->AddInputPtTensors(list_input_pt_vec);
        values.emplace_back(list_input);
      } else {
        PT_BRIDGE_FATAL(
            "Got unhandled type: ",
            input.tagKind(),
            " for ",
            m_symbol.toQualString(),
            " at index ",
            i);
        HABANA_ASSERT(0);
      }
    }

    auto node = ir::Node::Create(m_symbol, values);

    if (metadata.size()) {
      node->SetMetaData(metadata);
    }

    node->AddInputPtTensors(input_pt_vec);

    return node;
  }

 protected:
  std::vector<at::IValue>& get_inputs() {
    return m_inputs;
  }

  void set_inputs(const std::vector<at::IValue>& inputs) {
    auto inputsHpu = inputs;
    for (auto& t : inputsHpu) { // Any tensor on CPU needs to be moved to HPU
      if (t.isTensor() && t.toTensor().defined() &&
          t.toTensor().device().type() != c10::DeviceType::HPU) {
        const at::Tensor& tensor = t.toTensor();
        at::Tensor tinput;
        // If the CPU tensor is a wrapped number, then use
        // get_tensor_for_scalar method to retrieve cached HPU tensors for
        // the scalar value
        if (!IsConvertWrappedTensorToScalar() &&
            tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
          // is_wrapped_number: True if a tensor was auto-wrapped from a
          // C++ or Python number.
          auto dtype = tensor.scalar_type();
          tinput = get_tensor_for_scalar(
              tensor.item().toFloat(), at::TensorOptions().dtype(dtype));
        } else {
          // Use non_blocking .to()
          tinput = tensor.to(c10::kHPU, true);
        }
        t = c10::IValue(tinput);
      }
    }
    m_inputs = inputsHpu;
  }

  const std::vector<std::vector<int64_t>>& get_out_shapes() const {
    return m_out_shapes;
  }

  virtual ReturnType get_result_overrideable() {
    HABANA_ASSERT(
        0,
        "out_index is negative, implement get_result_overrideable() in your op.");
    // Call std::terminate here to avoid compilation error due to no return
    // statement. return cannot be here because sometimes the type is Tensor& or
    // a tuple of tensors. This terminate is never reachable though.
    std::terminate();
  }

  void set_scalar_type(const c10::ScalarType scalar_type) {
    m_scalar_type = scalar_type;
  }

  // JIT IR Cache key calculation for optimized lazy eager
  size_t calculate_optimized_lazy_eager_key() {
    size_t optimized_key = static_cast<uint32_t>(m_symbol);
    optimized_key = at::hash_combine(optimized_key, m_out_shapes.size());

    std::unordered_set<size_t> input_hash_values;
    for (size_t i = 0; i < m_inputs.size(); ++i) {
      optimized_key = at::hash_combine(optimized_key, i);
      const at::IValue& input = m_inputs[i];
      // Create stack based on input tensors / tensor lists.
      // Metadata and scalars are part of key calculation, so we skip them.
      if (m_metadata_indices.count(i)) {
        if (input.isList()) {
          for (auto& v : input.toListRef()) {
            optimized_key =
                at::hash_combine(optimized_key, at::IValue::hash(v));
          }
        } else {
          optimized_key =
              at::hash_combine(optimized_key, at::IValue::hash(input));
        }
        continue;
      } else if (input.isScalar()) {
        optimized_key =
            at::hash_combine(optimized_key, at::IValue::hash(input.toScalar()));
        continue;
      }

      if (input.isTensor()) {
        const at::Tensor& t = input.toTensor();
        if (t.defined()) {
          if (t.device().type() != c10::DeviceType::HPU) {
            // non HPU tensors to be handled later
            optimized_key = 0;
            break;
          }
          // Calculate hash based on unique tensor inputs.
          size_t input_hash_val = at::IValue::hash(input);
          if (input_hash_values.count(input_hash_val)) {
            continue;
          }
          input_hash_values.emplace(input_hash_val);
          update_hash_key_for_tensor(t, optimized_key);
          if (optimized_key == 0) {
            break;
          }
        } else {
          optimized_key = at::hash_combine(
              optimized_key, at::IValue::hash(torch::jit::IValue()));
        }
      } else if (input.isTensorList()) {
        const auto& tensors = input.toTensorVector();
        for (const auto& t : tensors) {
          update_hash_key_for_tensor(t, optimized_key);
          if (optimized_key == 0) {
            break;
          }
        }
        if (optimized_key == 0) {
          break;
        }
      } else if (isMetadataCandidate(input)) {
        // Not handled so returning null key
        optimized_key = 0;
        break;
      }
    }

    return optimized_key;
  }

  bool calculate_key_and_check_optimized_lazy_eager_cache(
      size_t& lazy_eager_key) {
    bool IsCached = false;
    if (!(std::getenv("PT_HPU_LAZY_CACHE_DISABLE"))) {
      lazy_eager_key = calculate_optimized_lazy_eager_key();
      PT_LAZY_DEBUG("Optimized Lazy Eager Key :: ", lazy_eager_key);
      if (lazy_eager_key != 0) {
        IsCached = habana_lazy::FastLazyGraphCache::GetFastLazyCache().IsCached(
            lazy_eager_key);
      }
    }

    return IsCached;
  }

  std::vector<ir::Value> prepare_lazy_eager_input_values() {
    std::vector<ir::Value> input_values;
    std::vector<ir::Value>::iterator it;
    for (size_t i = 0; i < m_inputs.size(); ++i) {
      const at::IValue& input = m_inputs[i];
      if (m_metadata_indices.count(i) || input.isScalar()) {
        continue;
      } else if (input.isTensor()) {
        const at::Tensor& t = input.toTensor();
        if (t.defined()) {
          if (t.device().type() != c10::DeviceType::HPU) {
            // DMA is default because aten schema may not be happy for most ops
            if (m_convert_wrapped_tensor_to_scalar) {
              continue;
            } else {
              at::Tensor tinput;
              // If the CPU tensor is a wrapped number, then use
              // get_tensor_for_scalar method to retrieve cached HPU tensors for
              // the scalar value
              if ((t.device().type() == c10::DeviceType::CPU)
                  // is_wrapped_number: True if a tensor was auto-wrapped from a
                  // C++ or Python number.
                  && (t.unsafeGetTensorImpl()->is_wrapped_number())) {
                // Set the dtype for the HPU tensor.
                //   Double : Float
                //   Long : Int
                //   Everything else is passed with the dtype of CPU tensor
                at::TensorOptions topt = {};
                auto dtype = t.scalar_type();
                switch (dtype) {
                  case at::ScalarType::Double:
                    topt = at::TensorOptions().dtype(at::ScalarType::Float);
                    break;
                  case at::ScalarType::Long:
                    topt = at::TensorOptions().dtype(at::ScalarType::Int);
                    break;
                  default:
                    topt = at::TensorOptions().dtype(dtype);
                    break;
                }
                tinput = get_tensor_for_scalar(t.item().toFloat(), topt);
              } else {
                // Use non_blocking .to()
                tinput = t.to(c10::kHPU, true);
              }
              auto val = GetHbLazyTensor(tinput).GetIrValue();
              it = find(input_values.begin(), input_values.end(), val);
              if (it == input_values.end()) {
                input_values.emplace_back(val);
                m_input_pt_tensors.emplace_back(tinput);
              }
            }
          } else {
            auto val = GetHbLazyTensor(t).GetIrValue();
            it = find(input_values.begin(), input_values.end(), val);
            if (it == input_values.end()) {
              input_values.emplace_back(val);
            }
          }
        }
      } else if (input.isTensorList()) {
        const auto& tensors = input.toTensorList();
        for (const auto& t : tensors) {
          auto val = GetHbLazyTensor(t).GetIrValue();
          it = find(input_values.begin(), input_values.end(), val);
          if (it == input_values.end()) {
            input_values.emplace_back(val);
          }
        }
      }
    }
    return input_values;
  }

 private:
  std::vector<at::Tensor> m_input_pt_tensors;
  bool m_convert_wrapped_tensor_to_scalar = false;
  ir::NodePtr m_node = nullptr;
  const at::Symbol m_symbol;
  const std::set<size_t> m_metadata_indices;
  std::vector<std::vector<int64_t>> m_out_shapes;
  const int m_out_index;
  at::TensorList m_out_meta_tensors = {};
  std::vector<at::IValue> m_inputs = {};
  c10::ScalarType m_scalar_type = c10::ScalarType::Undefined;
  void update_hash_key_for_tensor(const at::Tensor& t, size_t& optimized_key) {
    auto hl_tensor = TryGetHbLazyTensor(t);
    if (hl_tensor) {
      auto val = hl_tensor->GetIrValue();
      std::shared_ptr<Data> d = val.m_data_ptr.lock();
      torch::jit::IValue hl_tensor_ivalue = d->tensor_data;
      if (hl_tensor_ivalue.isTensor()) {
        auto hb_internal_tensor = hl_tensor_ivalue.toTensor();
        optimized_key =
            at::hash_combine(optimized_key, (size_t)hb_internal_tensor.dim());
        optimized_key = at::hash_combine(
            optimized_key,
            static_cast<size_t>(hb_internal_tensor.scalar_type()));
        optimized_key = at::hash_combine(
            optimized_key,
            static_cast<size_t>(hb_internal_tensor.suggest_memory_format()));
        if (hb_internal_tensor.has_storage()) {
          auto hb_tensor = GetHbInternalTensorImpl(hb_internal_tensor);
          if (hb_tensor) {
            auto lazy_layout_format = hb_tensor->GetTensorLayout();
            optimized_key = at::hash_combine(
                optimized_key, static_cast<size_t>(lazy_layout_format));
          }
        }
      }
      if (!(val.mp_node->is_input())) {
        optimized_key = 0;
      }
    }
  }
};

template <typename T>
class LazyOpWithTypePromotion : public LazyOp<T> {
 public:
  explicit LazyOpWithTypePromotion(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      const std::function<
          std::vector<std::vector<int64_t>>(const at::Stack&, bool)>&
          out_shapes_fn = nullptr) noexcept;

 private:
  T get_result_overrideable() override;
};

template <typename T>
class PromoteIntToFloat : public LazyOp<T> {
 public:
  explicit PromoteIntToFloat(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      const std::function<
          std::vector<std::vector<int64_t>>(const at::Stack&, bool)>&
          out_shapes_fn = nullptr) noexcept;

 private:
  T get_result_overrideable() override;
};

template <typename ReturnType>
class LazyBinaryOp : public LazyOp<ReturnType> {
 public:
  explicit LazyBinaryOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      const std::set<size_t>& metadata_indices = {},
      const std::vector<std::vector<int64_t>>& out_shapes = {},
      int out_index = 0)
      : LazyOp<ReturnType>(
            qualstring,
            inputs,
            metadata_indices,
            out_shapes,
            out_index) {}

  virtual ~LazyBinaryOp() = default;

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor>::value, T>::type call() {
    auto inputs = LazyOp<T>::get_inputs();

    int pos = -1;
    c10::ScalarType dst_dtype = c10::ScalarType::Float;
    habana_helpers::type_promotion_for_two_tensor_inputs(
        inputs, pos, dst_dtype);
    if (pos != -1) {
      auto tensor_promote = inputs[pos].toTensor();
      auto self = empty_hpu_lazy(
          tensor_promote.sizes(),
          tensor_promote.options().dtype(dst_dtype).device(at::kHPU),
          tensor_promote.suggest_memory_format(),
          false);
      self = copy_hpu_lazy_(self, tensor_promote, true);
      inputs[pos] = self;
      LazyOp<T>::set_inputs(inputs);
    }

    auto results = LazyOp<T>::call();
    return results;
  }

  // For inplace binary, the promoted type takes the self's type
  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor&>::value, T>::type call(
      at::Tensor& self) {
    auto inputs = LazyOp<T>::get_inputs();

    c10::ScalarType dst_dtype = self.scalar_type();
    at::Tensor other = inputs.at(1).toTensor();

    if (self.scalar_type() != other.scalar_type()) {
      at::Tensor casted_other = empty_hpu_lazy(
          other.sizes(),
          other.options().dtype(dst_dtype).device(at::kHPU),
          other.suggest_memory_format(),
          false);
      copy_hpu_lazy_(casted_other, other, true);
      inputs[1] = casted_other;
      LazyOp<T>::set_inputs(inputs);
    }

    return LazyOp<T>::call(self);
  }
};

} // namespace habana_lazy
