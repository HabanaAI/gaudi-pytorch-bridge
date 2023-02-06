/******************************************************************************
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

#include <tuple>
#include <utility>

#include "backend/jit_graph_cache.h"
#include "habana_eager/eager_tensor.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/ir.h"
#include "pytorch_helpers/habana_device/HPUStream.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

#include <ATen/EmptyTensor.h>
#include <c10/core/DeviceType.h>

#if IS_PYTORCH_OLDER_THAN(2, 0)
#define C10_AS_INTARRAYREF_SLOW(_X) c10::asIntArrayRefSlow(_X)
#endif

using SmallTensorVector = c10::SmallVector<at::Tensor, 8>;

namespace habana {
namespace eager {

struct OutputSpec {
  c10::ScalarType scalar_type;
  c10::Device device;
  c10::IntArrayRef sizes;
};

// helper function to create JIT graph with single node defined by input symbol
std::shared_ptr<torch::jit::Graph> create_simple_JIT(
    const at::Symbol& symbol,
    const SmallTensorVector& inputs,
    const std::vector<OutputSpec>& outputs,
    const habana_lazy::ir::MetaData& metadata);

template <typename ReturnType>
class EagerOp {
 public:
  explicit EagerOp(
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

  explicit EagerOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      std::vector<std::vector<int64_t>> out_shapes) noexcept
      : m_symbol{at::Symbol::fromQualString(qualstring)},
        m_metadata_indices{},
        m_out_shapes{std::move(out_shapes)},
        m_out_index{} {
    set_inputs(inputs);
  }

  explicit EagerOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      const std::function<std::vector<std::vector<int64_t>>(const at::Stack&)>&
          out_shapes_fn,
      int out_index = 0) noexcept
      : m_symbol{at::Symbol::fromQualString(qualstring)},
        m_metadata_indices{},
        m_out_index{out_index} {
    if (out_shapes_fn) {
      m_out_shapes = out_shapes_fn(inputs);
    }
    set_inputs(inputs);
  }

  explicit EagerOp(
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
    PT_LAZY_DEBUG("Eager Call regular :: ", m_symbol.toQualString());

    // TODO avoid calling get_result
    auto result = get_result();
    auto out_spec =
        OutputSpec{result.scalar_type(), result.device(), result.sizes()};
    auto stack = run({out_spec});
    HABANA_ASSERT(stack.size() == 1); // single output only
    return stack.at(0).toTensor();
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

  void set_scalar_type(const c10::ScalarType scalar_type) {
    m_scalar_type = scalar_type;
  }

  [[nodiscard]] c10::ScalarType get_scalar_type() const {
    return m_scalar_type;
  }

 private:
  torch::jit::Stack run(const std::vector<OutputSpec>& out_spec) {
    SmallTensorVector input_pt_vec, input_backend_pt_vec;
    habana_lazy::ir::MetaData metadata;
    create_inputs(input_pt_vec, metadata);
    convert_inputs_to_backend_tensors(input_pt_vec, input_backend_pt_vec);

    auto graph =
        create_simple_JIT(m_symbol, input_backend_pt_vec, {out_spec}, metadata);

    habana_lazy::exec::HlExec hlexec{};

    torch::jit::Stack stack;
    // stack is used for both inputs to synapse lowering and outputs from
    // synapse lowering, therefore allocate memory which is max of input
    // and output size - out is 1, so size(inputs)
    stack.reserve(input_backend_pt_vec.size());

    for (const auto& in : input_backend_pt_vec) {
      stack.emplace_back(in);
    }

    // Fetch graph from device context
    hlexec.set_graph(graph);

    // TODO enable JIT caching
    static int hash = 0;
    // Set the graph hash
    hlexec.set_hash(hash++);

    // TODO enable JIT caching - what's the diff between hash and graph_key?
    static int graphKey = 0;
    // Set the graph key
    hlexec.set_graph_key(graphKey++);

    // Set the op strs
    hlexec.set_opstrs(m_symbol.toQualString());

    // Launch the execution
    hlexec.Launch(stack, c10::hpu::getCurrentHPUStream(), {}, 0, 0);
    return stack;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor>::value, T>::type
  get_result() {
    PT_LAZY_TRACE;
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

  bool isMetadataCandidate(const at::IValue& input) const {
    return input.isBool() || input.isDevice() || input.isIntList() ||
        input.isDoubleList() || input.isBoolList() || input.isString() ||
        input.isNone() ||
        (input.isList() &&
         !input.toList().elementType()->cast<at::TensorType>());
  }

  void create_inputs(
      SmallTensorVector& input_pt_vec,
      habana_lazy::ir::MetaData& metadata) {
    for (size_t i = 0; i < m_inputs.size(); ++i) {
      const at::IValue& input = m_inputs[i];
      if (m_metadata_indices.count(i)) {
        metadata.set(input, i);
        continue;
      }

      if (input.isScalar() || isMetadataCandidate(input)) {
        metadata.set(input, i);
      } else if (input.isTensor()) {
        const at::Tensor& t = input.toTensor();
        if (t.defined()) {
          HABANA_ASSERT(t.device().type() == c10::DeviceType::HPU)
          input_pt_vec.emplace_back(t);
        } else {
          metadata.set(torch::jit::IValue(), i);
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

 protected:
  std::vector<at::IValue>& get_inputs() {
    return m_inputs;
  }

  void set_inputs(const std::vector<at::IValue>& inputs) {
    auto inputsHpu = inputs;
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
              "Got unexpected tensor as input to HPU Op. Tensor: ",
              tensor.toString());
        }
      }
    }
    m_inputs = inputsHpu;
  }

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
  const at::Symbol m_symbol;
  const std::set<size_t> m_metadata_indices;
  std::vector<std::vector<int64_t>> m_out_shapes;
  const int m_out_index;
  std::vector<at::IValue> m_inputs = {};
  c10::ScalarType m_scalar_type = c10::ScalarType::Undefined;
} __attribute__((aligned(64)));

} // namespace eager
} // namespace habana
