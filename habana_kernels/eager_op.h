/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include <ATen/EmptyTensor.h>
#include <c10/core/DeviceType.h>
#include <tuple>
#include <utility>

#include "backend/jit_graph_cache.h"
#include "habana_eager/eager_exec.h"
#include "habana_eager/eager_tensor.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"
#include "pytorch_helpers/habana_device/HPUStream.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

#if IS_PYTORCH_OLDER_THAN(2, 0)
#define C10_AS_INTARRAYREF_SLOW(_X) c10::asIntArrayRefSlow(_X)
#endif

namespace habana {
namespace eager {

// helper function to create JIT graph with single node defined by input symbol
// TODO remove
std::shared_ptr<torch::jit::Graph> create_simple_JIT(
    const at::Symbol& symbol,
    const SmallTensorVector& inputs,
    const std::vector<OutputSpec>& outputs,
    const habana_lazy::ir::MetaData& metadata);

class EagerOpBase {
 public:
  std::vector<at::IValue>& get_inputs() {
    return m_inputs;
  }

  const std::vector<std::vector<int64_t>>& get_out_shapes() const {
    return m_out_shapes;
  }

  const std::vector<at::IValue>& inputs() const {
    return m_inputs;
  }

  const c10::Symbol& symbol() const {
    return m_symbol;
  }

  [[nodiscard]] c10::ScalarType get_scalar_type() const {
    return m_scalar_type;
  }

  void set_scalar_type(const c10::ScalarType scalar_type) {
    m_scalar_type = scalar_type;
  }

  explicit EagerOpBase(
      const at::Symbol symbol,
      const std::vector<at::IValue>& inputs,
      std::set<size_t> metadata_indices = {},
      std::vector<std::vector<int64_t>> out_shapes = {},
      int out_index = 0,
      const c10::ScalarType scalar_type = c10::ScalarType::Undefined)
      : m_symbol{symbol},
        m_metadata_indices{std::move(metadata_indices)},
        m_out_shapes{std::move(out_shapes)},
        m_out_index{out_index},
        m_scalar_type(scalar_type) {
    set_inputs(inputs);
  }

 protected:
  void convert_inputs_to_backend_tensors(
      SmallTensorVector& input_pt_vec,
      SmallTensorVector& input_backend_pt_vec) {
    std::transform(
        input_pt_vec.begin(),
        input_pt_vec.end(),
        std::back_inserter(input_backend_pt_vec),
        [](at::Tensor& t_) {
          return HbEagerTensorPool::getInstance().get_backend_tensor(t_);
        });
  }

  torch::jit::Stack run(const std::vector<OutputSpec>& out_spec);

  const at::Symbol m_symbol;
  const std::set<size_t> m_metadata_indices;
  std::vector<std::vector<int64_t>> m_out_shapes;
  const int m_out_index;
  std::vector<at::IValue> m_inputs = {};
  c10::ScalarType m_scalar_type = c10::ScalarType::Undefined;

 private:
  void set_inputs(const std::vector<at::IValue>& inputs) {
    auto inputsHpu = inputs;
    size_t idx = 0;
    for (auto& t : inputsHpu) {
      if (t.isTensor() && t.toTensor().defined() &&
          t.toTensor().device().type() != c10::DeviceType::HPU) {
        const at::Tensor& tensor = t.toTensor();
        if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
          // If the CPU tensor is a 0D wrapped number, then transfer it to HPU
          at::Tensor tinput = tensor.to(c10::kHPU, true);
          t = c10::IValue(tinput);
        } else {
          HABANA_ASSERT(
              0,
              "Got unexpected tensor as input at index ",
              idx,
              " to HPU Op. Tensor: ",
              tensor.toString());
        }
      }
      ++idx;
    }
    m_inputs = inputsHpu;
  }

  bool is_metadata_candidate(const at::IValue& input) const {
    return input.isBool() || input.isDevice() || input.isIntList() ||
        input.isDoubleList() || input.isBoolList() || input.isString() ||
        input.isNone() ||
        (input.isList() &&
         !input.toList().elementType()->cast<at::TensorType>());
  }

  void create_inputs(
      SmallTensorVector& input_pt_vec,
      habana::eager::MetaDataMap& metadata) {
    for (size_t i = 0; i < m_inputs.size(); ++i) {
      const at::IValue& input = m_inputs[i];
      if (m_metadata_indices.count(i)) {
        HABANA_ASSERT(metadata.insert({i, input}).second);
        continue;
      }

      if (input.isScalar() || is_metadata_candidate(input)) {
        HABANA_ASSERT(metadata.insert({i, input}).second);
      } else if (input.isTensor()) {
        const at::Tensor& t = input.toTensor();
        if (t.defined()) {
          HABANA_ASSERT(t.device().type() == c10::DeviceType::HPU)
          input_pt_vec.emplace_back(t);
        } else {
          HABANA_ASSERT(metadata.insert({i, torch::jit::IValue()}).second);
        }
      } else if (input.isList()) {
        const auto& list = input.toListRef();

        for (const auto& li : list) {
          HABANA_ASSERT(
              li.isTensor(),
              "Got unhandled list item type: ",
              li.tagKind(),
              " for ",
              m_symbol.toQualString(),
              " at index ",
              i,
              ".");
          input_pt_vec.emplace_back(li.toTensor());
        }
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
  }
};

template <typename ReturnType>
class EagerOp : public EagerOpBase {
 public:
  explicit EagerOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      std::set<size_t> metadata_indices =
          {}, // TODO move away from set for this. inputs are forward-scanned
              // and every non-Tensor and non-TensorList is a metadata, no need
              // to take this from autogeneated api. set is also very
              // compile-time unfriendly, but that's exactly how we use it.
      std::vector<std::vector<int64_t>> out_shapes = {},
      int out_index = 0)
      : EagerOpBase(
            at::Symbol::fromQualString(qualstring),
            inputs,
            std::move(metadata_indices),
            std::move(out_shapes),
            out_index) {}

  explicit EagerOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      std::vector<std::vector<int64_t>> out_shapes)
      : EagerOpBase(
            at::Symbol::fromQualString(qualstring),
            inputs,
            {},
            std::move(out_shapes),
            {}) {}

  explicit EagerOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      const std::function<std::vector<std::vector<int64_t>>(const at::Stack&)>&
          out_shapes_fn,
      int out_index = 0)
      : EagerOpBase(
            at::Symbol::fromQualString(qualstring),
            inputs,
            {},
            out_shapes_fn(inputs),
            out_index) {}

  explicit EagerOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      std::vector<std::vector<int64_t>> out_shapes,
      const c10::ScalarType scalar_type)
      : EagerOpBase(
            at::Symbol::fromQualString(qualstring),
            inputs,
            {},
            std::move(out_shapes),
            {},
            scalar_type) {}

  EagerOp(EagerOp&) = default;
  EagerOp(const EagerOp&) = default;
  EagerOp(EagerOp&&) = default;
  EagerOp& operator=(const EagerOp&) = default;
  EagerOp& operator=(EagerOp&) = default;
  EagerOp& operator=(EagerOp&&) = default;

  virtual ~EagerOp() = default;

  // For inplace/out variants
  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor&>::value, T>::type call(
      at::Tensor& self) {
    PT_LAZY_DEBUG("Eager Call Inplace/out :: ", m_symbol.toQualString());

    HABANA_ASSERT(
        self.device().type() == at::kHPU,
        "Got a non-HPU tensor, expecting an HPU tensor");

    auto out_shape = m_out_shapes.empty()
        ? get_inputs().at(m_out_index).toTensor().sizes().vec()
        : m_out_shapes[0];
    if (self.sizes() != out_shape) {
      HABANA_ASSERT(
          self.numel() == 0,
          "Got a non-empty out tensor for out operation. Out shape: ",
          self.sizes());
      THHTensor_resizeNd(
          self.unsafeGetTensorImpl(),
          out_shape.size(),
          out_shape.data(),
          nullptr);
    }

    auto out_spec = OutputSpec{self.scalar_type(), self.device(), self.sizes()};
    auto stack = run({out_spec});
    HABANA_ASSERT(stack.size() == 1); // single output only
    return self;
  }

  // For regular variants
  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor>::value, T>::type call() {
    PT_EAGER_DEBUG("Eager Call regular :: ", m_symbol.toQualString());

    // TODO avoid calling get_result
    auto result = get_result();
    auto out_spec =
        OutputSpec{result.scalar_type(), result.device(), result.sizes()};
    auto stack = run({out_spec});
    HABANA_ASSERT(stack.size() == 1); // single output only
    return stack.at(0).toTensor();
  }

 private:
  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor>::value, T>::type
  get_result() {
    PT_EAGER_TRACE;
    // Get results from derived class when index is negative
    if (m_out_index < 0) {
      return get_result_overrideable();
    }

    const auto& t = get_inputs().at(m_out_index).toTensor();
    const auto& out_shape = m_out_shapes.empty() ? t.sizes() : m_out_shapes[0];
    if (m_scalar_type != c10::ScalarType::Undefined) {
      return at::empty(
          out_shape,
          t.options().dtype(m_scalar_type),
          t.suggest_memory_format());
    } else {
      return at::empty(out_shape, t.options(), t.suggest_memory_format());
    }
  }

 protected:
  virtual ReturnType get_result_overrideable() {
    HABANA_ASSERT(
        0,
        "out_index is negative, implement get_result_overrideable() in your op.");
    // Call std::terminate here to avoid compilation error due to no return
    // statement. return cannot be here because sometimes the type is Tensor&
    // or a tuple of tensors. This terminate is never reachable though.
    std::terminate();
  }

 private:
} __attribute__((aligned(64)));

} // namespace eager
} // namespace habana
