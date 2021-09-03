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
#include "pytorch_helpers/synapse_helpers/env_flags.h"
#include "resize.h"

namespace habana_lazy {
void updateDstDependencies(
    habana_lazy::HbLazyTensor& hl_dst,
    const at::Tensor& dst,
    bool in_place = false);

void flushWithMarkStep();

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

  virtual ~LazyOp() = default;

  template <typename T = ReturnType>
  typename std::enable_if<std::tuple_size<T>::value >= 2, T>::type call() {
    auto node = create_node();
    auto results = get_result();
    int i = 0;
    std::vector<HbLazyTensor> hl_tensors;
    hl_tensors.reserve(std::tuple_size<T>::value);

    for_each_in_tuple(results, [&node, &i, &hl_tensors](const auto& result) {
      auto hl_result = GetHbLazyTensor(result);
      ir::Value& out = hl_result.CurrentIrValue();
      out.SetNode(
          node,
          hl_result.GetDevice(),
          hl_result.GetSizes(),
          hl_result.dtype_optional(),
          i++);
      updateDstDependencies(hl_result, result, false);
      hl_tensors.push_back(hl_result);
    });
    if (m_flush_op) {
      HbLazyTensor::SyncTensorsGraph(&hl_tensors);
    }
    if (m_random_flush) {
      flushWithMarkStep();
    }

    return results;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor>::value, T>::type call() {
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

    if (m_flush_op) {
      std::vector<HbLazyTensor> hl_tensors = {hl_result};
      HbLazyTensor::SyncTensorsGraph(&hl_tensors);
    }
    if (m_random_flush) {
      flushWithMarkStep();
    }
    return result;
  }

  // For inplace/out variants
  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor&>::value, T>::type call(
      at::Tensor& self) {
    auto hl_self = GetHbLazyTensor(self);
    updateDstDependencies(hl_self, self, true);
    const auto& node = create_node();
    ir::Value& out = hl_self.CurrentIrValue();
    out.SetNode(
        node,
        hl_self.GetDevice(),
        hl_self.GetSizes(),
        hl_self.dtype_optional());

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

    auto context = habana_lazy_executor.getDeviceExecutionContext();
    context->MarkTensorStatus(
        hl_self.getTensorUniqueId(), LazyTensorExecutionStatus::kREGISTERED);

    if (m_flush_op) {
      std::vector<HbLazyTensor> hl_tensors = {hl_self};
      HbLazyTensor::SyncTensorsGraph(&hl_tensors);
    }
    if (m_random_flush) {
      flushWithMarkStep();
    }
    return self;
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
    return input.isBool() || input.isGenerator() || input.isDevice() ||
        input.isIntList() || input.isDoubleList() || input.isBoolList() ||
        input.isNone();
  }

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
      auto t = get_inputs().at(m_out_index).toTensor();
      const auto& out_shape =
          m_out_shapes.empty() ? t.sizes() : m_out_shapes[0];
      return empty_hpu_lazy(
          out_shape, t.options(), t.suggest_memory_format(), false);
    }

    const auto& t = m_out_meta_tensors[0];
    return empty_hpu_lazy(
        t.sizes(), t.options(), t.suggest_memory_format(), false);
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
      } else if (input.isTensor()) {
        const at::Tensor& t = input.toTensor();
        if (t.defined()) {
          if (t.device().type() != c10::DeviceType::HPU) {
            // DMA is default because aten schema may not be happy for most ops
            if (m_convert_wrapped_tensor_to_scalar) {
              auto val = GetIrValueForScalar(t.item());
              values.emplace_back(val);
            } else {
              auto tinput = t.to(c10::kHPU);
              auto val = GetHbLazyTensor(tinput).GetIrValue();
              values.emplace_back(val);
              input_pt_vec.emplace_back(tinput);
            }
          } else {
            auto val = GetHbLazyTensor(t).GetIrValue();
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
          auto val = GetHbLazyTensor(t).GetIrValue();
          hl_tensors.emplace_back(val);
          list_input_pt_vec.emplace_back(t);
        }

        auto list_input = GetIrValueForListConstruct(hl_tensors);
        list_input.mp_node->AddInputPtTensors(list_input_pt_vec);
        values.emplace_back(list_input);
      } else if (isMetadataCandidate(input)) {
        metadata.set(input, i);
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
        0,
        "out_index is negative, implement get_result_overrideable() in your op.");
    // Call std::terminate here to avoid compilation error due to no return
    // statement. return cannot be here because sometimes the type is Tensor& or
    // a tuple of tensors. This terminate is never reachable though.
    std::terminate();
  }

 private:
  bool m_convert_wrapped_tensor_to_scalar = false;
  ir::NodePtr m_node = nullptr;
  const at::Symbol m_symbol;
  const std::set<size_t> m_metadata_indices;
  std::vector<std::vector<int64_t>> m_out_shapes;
  const int m_out_index;
  at::TensorList m_out_meta_tensors = {};
  std::vector<at::IValue> m_inputs = {};
  // PT_HPU_LAZY_MODE=2 will flush the node as soon as it is created, more like
  // eager way of executing using lazy infrastructure.
  const bool m_flush_op = GET_ENV_FLAG(PT_HPU_LAZY_MODE) == 2;
  const bool m_random_flush = GET_ENV_FLAG(PT_HPU_LAZY_MODE) == 3;
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

    if (!LazyOp<T>::IsConvertWrappedTensorToScalar()) {
      for (auto& t : inputs) { // Any tensor on CPU needs to be moved to HPU for
                               // type promotion to work
        if (t.isTensor() &&
            t.toTensor().device().type() != c10::DeviceType::HPU) {
          auto h_tensor = t.toTensor().to(c10::kHPU);
          t = c10::IValue(h_tensor);
        }
      }
    }
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

template <typename ReturnType>
class LazyCompareOp : public LazyOp<ReturnType> {
 public:
  explicit LazyCompareOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      const std::set<size_t>& metadata_indices = {},
      const std::vector<std::vector<int64_t>>& out_shapes = {},
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

 private:
  at::Tensor get_result_overrideable() override {
    auto inputs = LazyOp<ReturnType>::get_inputs();
    auto self = inputs[0].toTensor();
    auto out_shapes = LazyOp<ReturnType>::get_out_shapes()[0];
    auto result = empty_hpu_lazy(
        out_shapes,
        self.options().dtype(c10::ScalarType::Bool),
        self.suggest_memory_format(),
        false);
    return result;
  }
};

template <
    typename ReturnType = std::tuple<at::Tensor, at::Tensor, at::Tensor>,
    typename NodeConstruct = void>
class HabanaNMSLazy : public LazyOp<ReturnType> {
 public:
  explicit HabanaNMSLazy(
      const std::vector<at::IValue>& inputs,
      const std::vector<std::vector<int64_t>>& out_shapes = {})
      : LazyOp<ReturnType>("hpu::habana_nms", inputs, {}, out_shapes, -1) {}

  virtual ~HabanaNMSLazy() = default;

  template <typename T = ReturnType>
  typename std::enable_if<std::tuple_size<T>::value >= 3, T>::type call() {
    return LazyOp<ReturnType>::call();
  }

 protected:
  virtual ReturnType get_result_overrideable() {
    ReturnType results;
    auto inputs = LazyOp<ReturnType>::get_inputs();
    auto scores = inputs[1].toTensor();
    auto box_id_out_shape = LazyOp<ReturnType>::get_out_shapes()[0];
    auto valid_box_id_out_shape = LazyOp<ReturnType>::get_out_shapes()[1];
    auto shape_tensor_shape = LazyOp<ReturnType>::get_out_shapes()[2];
    std::get<0>(results) = empty_hpu_lazy(
        box_id_out_shape,
        scores.options().dtype(c10::ScalarType::Int),
        scores.suggest_memory_format(),
        false);
    std::get<1>(results) = empty_hpu_lazy(
        valid_box_id_out_shape,
        scores.options().dtype(c10::ScalarType::Int),
        scores.suggest_memory_format(),
        false);
    std::get<2>(results) = empty_hpu_lazy(
        shape_tensor_shape,
        scores.options().dtype(c10::ScalarType::Int),
        scores.suggest_memory_format(),
        false);
    return results;
  }
};

template <
    typename ReturnType = std::tuple<at::Tensor, at::Tensor>,
    typename NodeConstruct = void>
class Unique : public LazyOp<ReturnType> {
 public:
  explicit Unique(
      const std::vector<at::IValue>& inputs,
      std::set<size_t> metadata_indices = {},
      const std::vector<std::vector<int64_t>>& out_shapes = {})
      : LazyOp<ReturnType>(
            "hpu::_unique2",
            inputs,
            metadata_indices,
            out_shapes,
            -1) {}

  virtual ~Unique() = default;

  template <typename T = ReturnType>
  typename std::enable_if<std::tuple_size<T>::value >= 2, T>::type call() {
    return LazyOp<ReturnType>::call();
  }

 protected:
  virtual ReturnType get_result_overrideable() {
    ReturnType results;
    auto inputs = LazyOp<ReturnType>::get_inputs();
    auto self = inputs[0].toTensor();
    int elements = self.numel();
    auto output_shape = at::DimVector{elements};
    auto valid_shape = at::DimVector{1};
    std::get<0>(results) = empty_hpu_lazy(
        output_shape, self.options(), self.suggest_memory_format(), false);
    std::get<1>(results) = empty_hpu_lazy(
        valid_shape,
        self.options().dtype(c10::ScalarType::Int),
        self.suggest_memory_format(),
        false);
    return results;
  }
};

} // namespace habana_lazy
