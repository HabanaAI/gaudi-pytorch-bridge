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
#include <c10_ver/core/SymIntArrayRef.h>
#include <tuple>
#include <utility>

#include "backend/habana_device/HPUStream.h"
#include "backend/jit_graph_cache.h"
#include "habana_eager/eager_exec.h"
#include "habana_eager/eager_tensor.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/template_helpers.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

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

  [[nodiscard]] const std::vector<c10::ScalarType>& get_scalar_types() const {
    return m_scalar_types;
  }

  void set_scalar_types(const std::vector<c10::ScalarType> scalar_types) {
    m_scalar_types = scalar_types;
  }

  void SetOutputMeta(
      std::function<habana::OutputMetaDataVector(const at::Stack&)>
          output_meta) {
    m_output_meta = std::move(output_meta);
  }

  void set_eager_op_info(EagerOpMetaData&& eager_op_meta_data) {
    m_eager_op_meta_data = std::move(eager_op_meta_data);
  }

  explicit EagerOpBase(
      const at::Symbol symbol,
      const std::vector<at::IValue>& inputs,
      std::vector<std::vector<int64_t>> out_shapes = {},
      int out_index = 0)
      : m_symbol{symbol},
        m_out_shapes{std::move(out_shapes)},
        m_out_index{out_index},
        m_inputs(inputs) {
    validate_inputs(m_inputs);
  }

 protected:
  torch::jit::Stack run(std::vector<OutputSpec>&& out_spec);

  const at::Symbol m_symbol;
  const std::set<size_t> m_metadata_indices;
  std::vector<std::vector<int64_t>> m_out_shapes;
  const int m_out_index;
  std::vector<at::IValue> m_inputs;
  std::vector<c10::ScalarType> m_scalar_types;
  std::function<habana::OutputMetaDataVector(const at::Stack&)> m_output_meta;
  EagerOpMetaData m_eager_op_meta_data;
  bool m_is_pipeline_supported = false;

 private:
  void validate_inputs(const std::vector<at::IValue>& inputs) {
    for (size_t idx = 0; idx < inputs.size(); ++idx) {
      auto& t = inputs[idx];
      if (!t.isTensor()) {
        continue;
      }

      auto tensor = t.toTensor();
      if (!tensor.defined()) {
        continue;
      }

      if (tensor.device().type() == c10::DeviceType::HPU) {
        continue;
      }

      if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
        continue;
      }

      HABANA_ASSERT(
          0,
          "Got unexpected tensor as input at index ",
          idx,
          " to HPU Op. Tensor: ",
          tensor.toString());
    }
  }
};

template <typename ReturnType>
class EagerOp : public EagerOpBase {
 public:
  explicit EagerOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      std::vector<std::vector<int64_t>> out_shapes = {},
      int out_index = 0)
      : EagerOpBase(
            at::Symbol::fromQualString(qualstring),
            inputs,
            std::move(out_shapes),
            out_index) {}

  explicit EagerOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      std::vector<std::vector<int64_t>> out_shapes)
      : EagerOpBase(
            at::Symbol::fromQualString(qualstring),
            inputs,
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
            out_index) {
    if (out_shapes_fn) {
      m_out_shapes = out_shapes_fn(inputs);
    }
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
    PT_EAGER_DEBUG("Eager Call inplace/out :: ", m_symbol.toQualString());
    m_is_pipeline_supported = true;

    HABANA_ASSERT(
        self.device().type() == at::kHPU,
        "Got a non-HPU tensor, expecting an HPU tensor");

    std::vector<int64_t> out_shape;
    if (m_output_meta) {
      out_shape = m_output_meta(get_inputs())[0].shape;
    } else if (m_out_shapes.empty())
      out_shape = get_inputs().at(m_out_index).toTensor().sizes().vec();
    else {
      out_shape = m_out_shapes[0];
    }

    if (self.sizes() != out_shape) {
      HABANA_ASSERT(
          self.numel() == 0 || (self.numel() == 1 && self.sizes().empty()),
          "Got a non-empty out tensor for out operation. Out shape: ",
          self.sizes(),
          ", out numel: ",
          self.numel());
      THHTensor_resizeNd(
          self.unsafeGetTensorImpl(),
          out_shape.size(),
          out_shape.data(),
          nullptr);
    }

    auto out_spec = OutputSpec{self.scalar_type(), self.device(), self.sizes()};
    auto stack = run({out_spec});
    return self;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, const at::Tensor&>::value, T>::type
  call(const at::Tensor& self) {
    PT_EAGER_DEBUG("Eager Call const inplace :: ", m_symbol.toQualString());
    m_is_pipeline_supported = true;

    HABANA_ASSERT(
        self.device().type() == at::kHPU,
        "Got a non-HPU tensor, expecting an HPU tensor");

    std::vector<int64_t> out_shape;
    if (m_output_meta) {
      out_shape = m_output_meta(get_inputs())[0].shape;
    } else if (m_out_shapes.empty())
      out_shape = get_inputs().at(m_out_index).toTensor().sizes().vec();
    else {
      out_shape = m_out_shapes[0];
    }
    if (self.sizes() != out_shape) {
      THHTensor_resizeNd(
          self.unsafeGetTensorImpl(),
          out_shape.size(),
          out_shape.data(),
          nullptr);
    }

    auto out_spec = OutputSpec{self.scalar_type(), self.device(), self.sizes()};
    auto stack = run({out_spec});
    return self;
  }

  template <typename T = ReturnType>
  typename std::enable_if<is_tuple_of_tensor_ref<T>::value, T>::type call(
      T self) {
    PT_EAGER_DEBUG(
        "Eager Call tuple_of_tensor_ref :: ", m_symbol.toQualString());
    m_is_pipeline_supported = true;

    std::vector<std::vector<int64_t>> out_shapes;
    if (m_output_meta) {
      const auto& meta = m_output_meta(get_inputs());
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          meta.size() == std::tuple_size<T>::value);
      for (const auto& output_meta : meta) {
        out_shapes.emplace_back(output_meta.shape);
      }
    } else {
      out_shapes = m_out_shapes;
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          out_shapes.empty() || out_shapes.size() == std::tuple_size<T>::value);
    }

    auto it_shape = out_shapes.begin();
    habana::for_each_in_tuple(self, [this, &it_shape](const auto& el) {
      HABANA_ASSERT(
          el.device().type() == at::kHPU,
          "Got a non-HPU tensor, expecting an HPU tensor");
    });

    if (!out_shapes.empty()) {
      habana::for_each_in_tuple_with_index(
          self, [this, &out_shapes](const auto& el, size_t index) {
            const auto& out_shape = out_shapes[index];
            if (el.sizes() != out_shape) {
              HABANA_ASSERT(
                  el.numel() == 0 || (el.numel() == 1 && el.sizes().empty()),
                  "Got a non-empty out tensor for out operation. Out shape: ",
                  el.sizes(),
                  ", out numel: ",
                  el.numel());
              THHTensor_resizeNd(
                  el.unsafeGetTensorImpl(),
                  out_shape.size(),
                  out_shape.data(),
                  nullptr);
            }
          });
    }

    std::vector<OutputSpec> out_spec;
    habana::for_each_in_tuple(self, [&out_spec](const auto& el) {
      out_spec.emplace_back(
          OutputSpec{el.scalar_type(), el.device(), el.sizes()});
    });

    auto stack = run(std::move(out_spec));
    return self;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_arithmetic<T>::value, T>::type call() {
    PT_EAGER_DEBUG("Eager Call arithmetic :: ", m_symbol.toQualString());

    auto result = at::empty(
        1,
        get_inputs().at(0).toTensor().options().dtype(
            c10::CppTypeToScalarType<T>::value));
    auto out_spec =
        OutputSpec{result.scalar_type(), result.device(), result.sizes()};
    auto stack = run({out_spec});
    HABANA_ASSERT(stack.size() == 1); // single output only
    auto out = stack.at(0).toTensor();
    return out.item().to<T>();
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_void<T>::value, T>::type call(
      at::TensorList tensors1,
      at::TensorList tensors2) {
    PT_EAGER_DEBUG(
        "Eager call void ( 2x TensorList ) :: ", m_symbol.toQualString());

    for (const auto& tensor : tensors1) {
      HABANA_ASSERT(
          tensor.device().type() == at::kHPU,
          "Got a non-HPU tensor, expecting an HPU tensor");
    }
    for (const auto& tensor : tensors2) {
      HABANA_ASSERT(
          tensor.device().type() == at::kHPU,
          "Got a non-HPU tensor, expecting an HPU tensor");
    }

    std::vector<OutputSpec> out_spec;
    for (auto& el : tensors1) {
      out_spec.emplace_back(
          OutputSpec{el.scalar_type(), el.device(), el.sizes()});
    }

    auto stack = run(std::move(out_spec));
    HABANA_ASSERT(stack.size() == tensors1.size());
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_void<T>::value, T>::type call(
      at::TensorList tensors) {
    PT_EAGER_DEBUG(
        "Eager call void ( 1x TensorList ) :: ", m_symbol.toQualString());

    for (const auto& tensor : tensors) {
      HABANA_ASSERT(
          tensor.device().type() == at::kHPU,
          "Got a non-HPU tensor, expecting an HPU tensor");
    }

    std::vector<OutputSpec> out_spec;
    for (auto& el : tensors) {
      out_spec.emplace_back(
          OutputSpec{el.scalar_type(), el.device(), el.sizes()});
    }

    auto stack = run(std::move(out_spec));
    HABANA_ASSERT(stack.size() == tensors.size());
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_void<T>::value, T>::type call(
      const std::vector<at::Tensor>& tensors) {
    PT_EAGER_DEBUG(
        "Eager call void ( const ref std::vector<at::Tensor> ) :: ",
        m_symbol.toQualString());

    for (const auto& tensor : tensors) {
      HABANA_ASSERT(
          tensor.device().type() == at::kHPU,
          "Got a non-HPU tensor, expecting an HPU tensor");
    }

    std::vector<OutputSpec> out_spec;
    for (auto& el : tensors) {
      out_spec.emplace_back(
          OutputSpec{el.scalar_type(), el.device(), el.sizes()});
    }

    auto stack = run(std::move(out_spec));
    HABANA_ASSERT(stack.size() == tensors.size());
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_void<T>::value, T>::type call(
      const at::Tensor& tensor) {
    PT_EAGER_DEBUG(
        "Eager call void ( const ref at::Tensor ) :: ",
        m_symbol.toQualString());

    HABANA_ASSERT(
        tensor.device().type() == at::kHPU,
        "Got a non-HPU tensor, expecting an HPU tensor");

    auto stack = run(
        {OutputSpec{tensor.scalar_type(), tensor.device(), tensor.sizes()}});
    HABANA_ASSERT(stack.size() == 1); // single output only
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

  template <typename T = ReturnType>
  typename std::enable_if<is_tuple_of_tensors<T>::value, T>::type call() {
    PT_EAGER_DEBUG("Eager Call tuple_of_tensors :: ", m_symbol.toQualString());

    // TODO avoid calling get_result
    auto result = get_result();

    std::vector<OutputSpec> out_spec;
    habana::for_each_in_tuple(result, [&out_spec](const auto& el) {
      out_spec.emplace_back(
          OutputSpec{el.scalar_type(), el.device(), el.sizes()});
    });

    auto stack = run(std::move(out_spec));
    HABANA_ASSERT(stack.size() == std::tuple_size<T>::value);

    ReturnType outputs;
    auto it_stack = stack.begin();
    habana::for_each_in_tuple_with_index(
        outputs, [&stack = stack](auto& el, size_t index) {
          el = stack[index].toTensor();
        });

    return outputs;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, std::vector<at::Tensor>>::value, T>::
      type
      call() {
    PT_EAGER_DEBUG(
        "Eager Call std::vector<at::Tensor> :: ", m_symbol.toQualString());
    // TODO avoid calling get_result
    const auto& tensors = get_result();

    std::vector<OutputSpec> out_spec;
    for (auto& el : tensors) {
      out_spec.emplace_back(
          OutputSpec{el.scalar_type(), el.device(), el.sizes()});
    };
    auto stack = run(std::move(out_spec));
    ReturnType outputs;
    for (auto& el : stack) {
      outputs.push_back(el.toTensor());
    }
    return outputs;
  }

 private:
  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor>::value, T>::type
  get_result() {
    PT_EAGER_TRACE;
    if (m_output_meta) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(m_out_index == 0);
      auto meta = m_output_meta(get_inputs());
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(meta.size() == 1);
      auto output_meta = meta[0];
      return at::empty(
          output_meta.shape, output_meta.dtype, output_meta.mem_format);
    }
    // Get results from derived class when index is negative
    if (m_out_index < 0) {
      return get_result_overrideable();
    }

    const auto& t = get_inputs().at(m_out_index).toTensor();
    const auto& out_shape = m_out_shapes.empty() ? t.sizes() : m_out_shapes[0];
    auto options = t.options();
    if (m_scalar_types.size()) {
      HABANA_ASSERT(m_scalar_types.size() == 1);
      options = options.dtype(m_scalar_types[0]);
    }
    return at::empty(out_shape, options, t.suggest_memory_format());
  }

  template <typename T = ReturnType>
  typename std::enable_if<is_tuple_of_tensors<T>::value, T>::type get_result() {
    PT_EAGER_TRACE;
    if (m_output_meta) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(m_out_index == 0);
      const auto& meta = m_output_meta(get_inputs());
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          std::tuple_size<T>::value == meta.size());
      ReturnType results;
      habana::for_each_in_tuple_with_index(
          results, [&](auto& result, size_t index) {
            auto output_meta = meta[index];
            result = at::empty(
                output_meta.shape, output_meta.dtype, output_meta.mem_format);
          });
      return results;
    }

    // Get results from derived class when index is negative
    if (m_out_index < 0) {
      return get_result_overrideable();
    }
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        std::tuple_size<T>::value == m_out_shapes.size());

    ReturnType results;
    const auto& t = get_inputs().at(m_out_index).toTensor();
    if (m_scalar_types.empty()) {
      habana::for_each_in_tuple_with_index(
          results, [&](auto& result, size_t index) {
            result = at::empty(
                m_out_shapes[index], t.options(), t.suggest_memory_format());
          });
    } else {
      HABANA_ASSERT(m_scalar_types.size() == std::tuple_size<T>::value);
      habana::for_each_in_tuple_with_index(
          results, [&](auto& result, size_t index) {
            result = at::empty(
                m_out_shapes[index],
                t.options().dtype(m_scalar_types[index]),
                t.suggest_memory_format());
          });
    }
    return results;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, std::vector<at::Tensor>>::value, T>::
      type
      get_result() {
    if (m_output_meta) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(m_out_index == 0);
      const auto& meta = m_output_meta(get_inputs());
      std::vector<at::Tensor> results;

      results.reserve(meta.size());
      for (const auto output_meta : meta) {
        results.emplace_back(at::empty(
            output_meta.shape, output_meta.dtype, output_meta.mem_format));
      }
      return results;
    }
    // Get results from derived class when index is negative
    if (m_out_index < 0) {
      return get_result_overrideable();
    }
    return {};
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
