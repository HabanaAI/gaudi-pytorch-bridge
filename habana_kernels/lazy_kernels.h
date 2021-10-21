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
void updateDstDependencies(
    habana_lazy::HbLazyTensor& hl_dst,
    const at::Tensor& dst,
    bool in_place = false);

void flushWithMarkStep();

at::Tensor get_tensor_for_scalar(
    float alpha,
    const at::TensorOptions& options = {});

void flush_op(at::TensorList tensors);

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
  typename std::enable_if<not is_tuple_of_tensor_ref<T>::value, T>::type call() {
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
    flush_op(tensors);
    return results;
  }

  template <typename T = ReturnType>
  typename std::enable_if<is_tuple_of_tensor_ref<T>::value, T>::type call(
      T results) {
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
    flush_op(tensors);
    return results;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_fundamental<T>::value, T>::type call() {
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
  HandleOptimizedLazyEager() {
    std::shared_ptr<torch::jit::Graph> fast_path_jit_graph = nullptr;
    size_t lazy_eager_key = 0;
    if (!(std::getenv("PT_HPU_LAZY_CACHE_DISABLE"))) {
      lazy_eager_key = calculate_optimized_lazy_eager_key();
      PT_LAZY_DEBUG("Optimized Lazy Eager Key :: ", lazy_eager_key);
      if (lazy_eager_key != 0) {
        fast_path_jit_graph =
            habana_lazy::FastLazyGraphCache::GetFastLazyCache()
                .GetOptimizedJITGraph(lazy_eager_key);
      }
    }

    if (fast_path_jit_graph == nullptr) {
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
      std::vector<HbLazyTensor> hl_tensors = {hl_result};
      HbLazyTensor::SyncTensorsGraph(&hl_tensors, lazy_eager_key);
      return result;
    } else {
      PT_LAZY_DEBUG("Optimized Lazy Eager Path Chosen");
      const auto& result = get_result();
      auto hl_result = GetHbLazyTensor(result);
      std::vector<ir::Value> input_values = prepare_lazy_eager_input_values();
      std::vector<HbLazyTensor> hl_tensors = {hl_result};
      HbLazyTensor::SyncTensorsGraphFast(
          &hl_tensors, input_values, lazy_eager_key);
      return result;
    }
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor>::value, T>::type call() {
    PT_LAZY_DEBUG("Lazy Call :: ", m_symbol.toQualString());

    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2 &&
        GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_OPTIM_CACHE) == 1) {
      return (HandleOptimizedLazyEager());
    }

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
    flush_op(result);
    return result;
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

  // For inplace/out variants
  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor&>::value, T>::type call(
      at::Tensor& self) {
    auto hl_self = GetHbLazyTensor(self);
    // skip ctrl edges for inplace
    // TODO do the same for out variants

    if (!is_inplace(m_symbol)) {
      updateDstDependencies(hl_self, self, true);
    }
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
    flush_op(self);
    return self;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, const at::Tensor&>::value, T>::type
  call() {
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
    flush_op(self);
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
      } else if (input.isTensor()) {
        const at::Tensor& t = input.toTensor();
        if (t.defined()) {
          if (t.device().type() != c10::DeviceType::HPU) {
            // DMA is default because aten schema may not be happy for most ops
            if (m_convert_wrapped_tensor_to_scalar) {
              auto val = GetIrValueForScalar(t.item());
              values.emplace_back(val);
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
  std::vector<at::IValue>& get_inputs() {
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
        // Calculate hash based on unique tensor inputs.
        size_t input_hash_val = at::IValue::hash(input);
        if (input_hash_values.count(input_hash_val)) {
          continue;
        }
        input_hash_values.emplace(input_hash_val);
        const at::Tensor& t = input.toTensor();
        update_hash_key_for_tensor(t, optimized_key);
        if (optimized_key == 0) {
          break;
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
    optimized_key = at::hash_combine(optimized_key, (size_t)t.dim());
    optimized_key =
        at::hash_combine(optimized_key, static_cast<size_t>(t.scalar_type()));
    optimized_key = at::hash_combine(
        optimized_key, static_cast<size_t>(t.suggest_memory_format()));
    auto hl_tensor = TryGetHbLazyTensor(t);
    if (hl_tensor) {
      optimized_key =
          at::hash_combine(optimized_key, (size_t)hl_tensor->GetTensorLayout());
      auto val = hl_tensor->GetIrValue();
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
      for (auto& t : inputs) { // Any tensor on CPU needs to be moved to HPU
                               // for type promotion to work
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
} // namespace habana_lazy
