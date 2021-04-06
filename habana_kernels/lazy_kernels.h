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

#include <utility>

#include "habana_kernels/kernel_utils.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/lazy_executor.h"
#include "lazy_kernels_declarations.h"

void updateDstDependencies(
    habana_lazy::HbLazyTensor& hl_dst,
    const at::Tensor& dst,
    bool in_place = false);

namespace habana_lazy {

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
      int out_index = 0)
      : m_symbol{at::Symbol::fromQualString(qualstring)},
        m_metadata_indices{std::move(metadata_indices)},
        m_out_shapes{std::move(out_shapes)},
        m_out_index{out_index} {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        !std::is_class<NodeConstruct>::value,
        "This constructor is valid only when NodeConstruct is not a class.");
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

  virtual ~LazyOp() = default;

  template <typename T = ReturnType>
  typename std::enable_if<std::tuple_size<T>::value >= 2, T>::type call() {
    auto node = create_node();
    auto results = get_result();
    int i = 0;
    for_each_in_tuple(results, [&node, &i](const auto& result) {
      auto hl_result = GetHbLazyTensor(result);
      ir::Value& out = GetHbLazyTensor(result).CurrentIrValue();
      out.SetNode(node, i++);
      updateDstDependencies(hl_result, result, false);
    });

    return results;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor>::value, T>::type call() {
    const auto& node = create_node();
    const auto& result = get_result();
    auto hl_result = GetHbLazyTensor(result);
    ir::Value& out = GetHbLazyTensor(result).CurrentIrValue();
    out.SetNode(node);
    updateDstDependencies(hl_result, result, false);

    return result;
  }

  // For inplace/out variants
  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor&>::value, T>::type call(
      at::Tensor& self) {
    const auto& node = create_node();
    auto hl_self = GetHbLazyTensor(self);
    updateDstDependencies(hl_self, self, true);
    ir::Value& out = hl_self.CurrentIrValue();
    out.SetNode(node);

    auto context = habana_lazy_executor.getDeviceExecutionContext();
    context->MarkTensorStatus(
        hl_self.getTensorUniqueId(), LazyTensorExecutionStatus::kREGISTERED);
    return self;
  }

  void do_dma_non_first_cpu_tensor() {
    m_dma_non_first_cpu_tensor = true;
  }

 private:
  template <typename T = ReturnType>
  typename std::enable_if<std::tuple_size<T>::value >= 2, ReturnType>::type
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
      result = at::native::empty_hpu_lazy(
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
    auto t = get_inputs().at(m_out_index).toTensor();
    const auto& out_shape = m_out_shapes.empty() ? t.sizes() : m_out_shapes[0];
    return at::native::empty_hpu_lazy(
        out_shape, t.options(), t.suggest_memory_format(), false);
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
      const auto& input = m_inputs[i];
      if (m_metadata_indices.count(i)) {
        metadata.set(input, i);
        continue;
      }

      if (input.isScalar()) {
        auto val = GetIrValueForScalar(input.toScalar());
        values.emplace_back(val);
      } else if (input.isTensor()) {
        const auto& t = input.toTensor();
        if (t.defined()) {
          if (i != 0 && t.device().type() == c10::DeviceType::CPU) {
            // Non first arg can be a non habana tensor.
            // Convert to scalar/DMA such tensor to device and add as node
            // input.
            if (m_dma_non_first_cpu_tensor) {
              auto tinput = t.to(c10::kHABANA);
              auto val = GetOrCreateHbLazyTensor(tinput).GetIrValue();
              values.emplace_back(val);
              input_pt_vec.emplace_back(tinput);
            } else {
              auto val = GetIrValueForScalar(t.item());
              values.emplace_back(val);
            }
          } else {
            auto val = GetOrCreateHbLazyTensor(t).GetIrValue();
            values.emplace_back(val);
            input_pt_vec.emplace_back(t);
          }
        } else {
          metadata.set(torch::jit::IValue(), i);
        }
      } else if (input.isTensorList()) {
        const auto& tensors = input.toTensorList();
        ir::ValueList hl_tensors;
        std::vector<at::Tensor> list_input_pt_vec;
        for (const auto& t : tensors) {
          auto val = GetOrCreateHbLazyTensor(t).GetIrValue();
          hl_tensors.emplace_back(val);
          list_input_pt_vec.emplace_back(t);
        }

        auto list_input = GetIrValueForListConstruct(hl_tensors);
        list_input.mp_node->AddInputPtTensors(list_input_pt_vec);
        values.emplace_back(list_input);
      } else {
        PT_BRIDGE_FATAL("Got unhandled type at index ", i);
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
  const std::vector<at::IValue>& get_inputs() const {
    return m_inputs;
  }

  void set_inputs(const std::vector<at::IValue>& inputs) {
    m_inputs = inputs;
  }

  const std::vector<std::vector<int64_t>>& get_out_shapes() const {
    return m_out_shapes;
  }

  virtual ReturnType get_result_overrideable() {
    HABANA_ASSERT(
        0 &&
        "out_index is negative, implement get_result_overrideable() in your kernel.");
  }

 private:
  bool m_dma_non_first_cpu_tensor = false;
  ir::NodePtr m_node = nullptr;
  const at::Symbol m_symbol;
  const std::set<size_t> m_metadata_indices;
  const std::vector<std::vector<int64_t>> m_out_shapes;
  const int m_out_index;
  std::vector<at::IValue> m_inputs = {};
};

template <typename ReturnType>
class LazyBinaryOp : public LazyOp<ReturnType> {
 public:
  explicit LazyBinaryOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      std::set<size_t> metadata_indices = {},
      std::vector<std::vector<int64_t>> out_shapes = {},
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
      auto self = at::native::empty_hpu_lazy(
          tensor_promote.sizes(),
          tensor_promote.options().dtype(dst_dtype),
          tensor_promote.suggest_memory_format(),
          false);
      self = copy_hpu_lazy_(self, tensor_promote, true);
      inputs[pos] = IValue(self);
      LazyOp<T>::set_inputs(inputs);
    }

    auto results = LazyOp<T>::call();
    return results;
  }
};

template <typename ReturnType>
class LazyCompareOp : public LazyOp<ReturnType> {
 public:
  explicit LazyCompareOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      std::set<size_t> metadata_indices = {},
      std::vector<std::vector<int64_t>> out_shapes = {},
      int out_index = -1)
      : LazyOp<ReturnType>(
            qualstring,
            inputs,
            metadata_indices,
            out_shapes,
            out_index) {}

  virtual ~LazyCompareOp() = default;

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor>::value, T>::type call() {
    auto inputs = LazyOp<T>::get_inputs();

    int pos = -1;
    c10::ScalarType dst_dtype = c10::ScalarType::Float;
    habana_helpers::type_promotion_for_two_tensor_inputs(
        inputs, pos, dst_dtype);
    if (pos != -1) {
      auto tensor_promote = inputs[pos].toTensor();
      auto self = at::native::empty_hpu_lazy(
          tensor_promote.sizes(),
          tensor_promote.options().dtype(dst_dtype),
          tensor_promote.suggest_memory_format(),
          false);
      self = copy_hpu_lazy_(self, tensor_promote, true);
      inputs[pos] = IValue(self);
      LazyOp<T>::set_inputs(inputs);
    }

    auto results = LazyOp<T>::call();
    return results;
  }

 private:
  at::Tensor get_result_overrideable() override {
    auto inputs = LazyOp<ReturnType>::get_inputs();
    auto self = inputs[0].toTensor();
    auto out_shapes = LazyOp<ReturnType>::get_out_shapes()[0];
    auto result = at::native::empty_hpu_lazy(
        out_shapes,
        self.options().dtype(c10::ScalarType::Bool),
        self.suggest_memory_format(),
        false);
    return result;
  }
};

template <
    typename ReturnType = std::tuple<at::Tensor, at::Tensor>,
    typename NodeConstruct = void>
class FusedDropout : public LazyOp<ReturnType> {
 public:
  explicit FusedDropout(
      const std::vector<at::IValue>& inputs,
      std::set<size_t> metadata_indices = {})
      : LazyOp<ReturnType>(
            "aten::_fused_dropout",
            inputs,
            metadata_indices,
            {},
            -1) {}

  virtual ~FusedDropout() = default;

  template <typename T = ReturnType>
  typename std::enable_if<std::tuple_size<T>::value >= 2, T>::type call() {
    return LazyOp<ReturnType>::call();
  }

 protected:
  virtual ReturnType get_result_overrideable() {
    ReturnType results;
    auto t = LazyOp<ReturnType>::get_inputs().at(0).toTensor();
    std::get<0>(results) = at::native::empty_hpu_lazy(
        t.sizes(), t.options(), t.suggest_memory_format(), false);
    std::get<1>(results) = at::native::empty_hpu_lazy(
        t.sizes(),
        t.options().dtype(c10::ScalarType::Char),
        t.suggest_memory_format(),
        false);
    return results;
  }
};

} // namespace habana_lazy
