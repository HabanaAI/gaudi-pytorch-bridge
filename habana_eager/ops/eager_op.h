/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#pragma once

#include <ATen/EmptyTensor.h>
#include <c10/core/DeviceType.h>
#include <c10/core/MemoryFormat.h>
#include <c10_ver/core/SymIntArrayRef.h>
#include <tuple>
#include <utility>

#include "backend/habana_device/HPUStream.h"
#include "backend/jit_graph_cache.h"
#include "common/list_of_lists_custom_iterator.h"
#include "habana_eager/eager_exec.h"
#include "habana_eager/eager_tensor.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/template_helpers.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

namespace habana {
namespace eager {

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

  void SetOutputMetaFn(
      std::function<habana::OutputMetaDataVector(const at::Stack&)>
          output_meta_fn) {
    m_output_meta_fn = std::move(output_meta_fn);
  }

  void SetSTMetaFn(std::function<bool(
                       habana_helpers::IShapeList& inputs,
                       habana_helpers::IShapeList& outputs)> fn) {
    m_st_meta_fn = std::move(fn);
  }

  void set_eager_op_info(EagerOpMetaData&& eager_op_meta_data) {
    m_eager_op_meta_data = std::move(eager_op_meta_data);
  }

 private:
  auto process_EagerOpBase_input(
      std::vector<std::vector<int64_t>>&& out_shapes,
      const at::Stack&) {
    return out_shapes;
  }

  auto process_EagerOpBase_input(
      const std::function<std::vector<std::vector<int64_t>>(const at::Stack&)>&
          out_shapes_fn,
      const at::Stack& inputs) {
    return out_shapes_fn ? out_shapes_fn(inputs)
                         : std::vector<std::vector<int64_t>>{};
  }

 public:
  template <
      class T = std::vector<at::IValue>,
      class U = std::vector<std::vector<int64_t>>>
  EagerOpBase(
      const std::string& qualstring,
      T&& inputs,
      U&& out_shapes = {},
      int out_index = 0)
      : m_symbol{at::Symbol::fromQualString(qualstring)},
        m_out_shapes{
            process_EagerOpBase_input(std::forward<U>(out_shapes), inputs)},
        m_out_index{out_index},
        m_inputs(std::forward<T>(inputs)) {
    validate_inputs(m_inputs, qualstring);
  }

 protected:
  void run(OutputSpecsOrTensors&& out_spec_or_tensors);

  at::Symbol m_symbol;
  std::vector<std::vector<int64_t>> m_out_shapes;
  const int m_out_index;
  std::vector<at::IValue> m_inputs;
  std::vector<c10::ScalarType> m_scalar_types;
  std::function<habana::OutputMetaDataVector(const at::Stack&)>
      m_output_meta_fn;
  std::function<bool(
      habana_helpers::IShapeList& inputs,
      habana_helpers::IShapeList& outputs)>
      m_st_meta_fn;
  EagerOpMetaData m_eager_op_meta_data;

  void validate_inputs(
      const std::vector<at::IValue>& inputs,
      const std::string& qualstring);
  static std::mutex m_mutex;
};

template <typename ReturnType>
class EagerOp : public EagerOpBase {
 public:
  using EagerOpBase::EagerOpBase;

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

    HABANA_ASSERT(
        self.device().type() == at::kHPU,
        "Got a non-HPU tensor, expecting an HPU tensor");

    std::vector<int64_t> out_shape;
    if (m_output_meta_fn) {
      out_shape = m_output_meta_fn(get_inputs())[0].shape;
    } else if (m_out_shapes.empty())
      out_shape = get_inputs().at(m_out_index).toTensor().sizes().vec();
    else {
      out_shape = m_out_shapes[0];
    }

    if (self.sizes() != out_shape) {
      if (!(self.numel() == 0 || self.sizes().empty())) {
        PT_EAGER_WARN(
            "Got a non-empty out tensor for out operation. Out shape: ",
            self.sizes());
        TORCH_WARN(
            "An output with one or more elements was resized since it had ",
            "shape ",
            self.sizes(),
            ", which does not match the required ",
            "output shape ",
            c10::ArrayRef<int64_t>(out_shape),
            ". ",
            "This behavior is deprecated, and in a future PyTorch release outputs ",
            "will not be resized unless they have zero elements. You can explicitly ",
            "reuse an out tensor t by resizing it, inplace, to zero elements with ",
            "t.resize_(0).");
      }
      THHTensor_resizeNd(
          self.unsafeGetTensorImpl(),
          out_shape.size(),
          out_shape.data(),
          nullptr);
    }

    auto out_spec =
        OutputSpec{self.scalar_type(), self.device(), self.sizes().vec()};
    run({out_spec});
    return self;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, const at::Tensor&>::value, T>::type
  call(const at::Tensor& self) {
    PT_EAGER_DEBUG("Eager Call const inplace :: ", m_symbol.toQualString());

    HABANA_ASSERT(
        self.device().type() == at::kHPU,
        "Got a non-HPU tensor, expecting an HPU tensor");

    std::vector<int64_t> out_shape;
    if (m_output_meta_fn) {
      out_shape = m_output_meta_fn(get_inputs())[0].shape;
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

    auto out_spec =
        OutputSpec{self.scalar_type(), self.device(), self.sizes().vec()};
    run({out_spec});
    return self;
  }

  template <typename T = ReturnType>
  typename std::enable_if<is_tuple_of_tensor_ref<T>::value, T>::type call(
      T self) {
    PT_EAGER_DEBUG(
        "Eager Call tuple_of_tensor_ref :: ", m_symbol.toQualString());

    std::vector<std::vector<int64_t>> out_shapes;
    if (m_output_meta_fn) {
      const auto& meta = m_output_meta_fn(get_inputs());
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

    habana::for_each_in_tuple(self, [](const auto& el) {
      HABANA_ASSERT(
          el.device().type() == at::kHPU,
          "Got a non-HPU tensor, expecting an HPU tensor");
    });

    if (!out_shapes.empty()) {
      habana::for_each_in_tuple_with_index(
          self, [&out_shapes](const auto& el, size_t index) {
            const auto& out_shape = out_shapes[index];
            if (el.sizes() != out_shape) {
              PT_EAGER_WARN(
                  "Got a non-empty out tensor for out operation. Out shape: ",
                  el.sizes());
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
          OutputSpec{el.scalar_type(), el.device(), el.sizes().vec()});
    });

    run(std::move(out_spec));
    return self;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_arithmetic<T>::value, T>::type call() {
    PT_EAGER_DEBUG("Eager Call arithmetic :: ", m_symbol.toQualString());

    auto result = at::empty(
        1,
        get_inputs().at(0).toTensor().options().dtype(
            c10::CppTypeToScalarType<T>::value),
        at::MemoryFormat::Contiguous);
    run({HbEagerTensorPool::get_backend_tensor(result)});
    return result.item().template to<T>();
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
          OutputSpec{el.scalar_type(), el.device(), el.sizes().vec()});
    }

    run(std::move(out_spec));
  }

  template <typename T = ReturnType, class U>
  typename std::enable_if<std::is_void<T>::value, T>::type call_internal_lists(
      U list,
      const char* label) {
    PT_EAGER_DEBUG(
        "Eager call void ( ", label, " ) :: ", m_symbol.toQualString());

    std::vector<OutputSpec> out_spec;
    size_t tensors_size = 0;
    common::ListOfListsCustomIterator<U> customIt(list);
    if (!customIt.empty()) {
      do {
        auto tensors = customIt.get_next_item();
        tensors_size += tensors.size();
        for (const auto& tensor : tensors) {
          HABANA_ASSERT(
              tensor.device().type() == at::kHPU,
              "Got a non-HPU tensor, expecting an HPU tensor");

          out_spec.emplace_back(OutputSpec{
              tensor.scalar_type(), tensor.device(), tensor.sizes().vec()});
        }
      } while (customIt.has_more_items());
    }

    run(std::move(out_spec));
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_void<T>::value, T>::type call(
      at::TensorList tensors) {
    return call_internal_lists<T>(tensors, "1x TensorList");
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_void<T>::value, T>::type call(
      const std::vector<at::Tensor>& tensors) {
    return call_internal_lists<T>(
        at::TensorList{tensors}, "const ref std::vector<at::Tensor>");
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_void<T>::value, T>::type call(
      const std::vector<at::TensorList>& tensorlists) {
    return call_internal_lists<T>(
        c10::ArrayRef<at::TensorList>{tensorlists},
        "const ref std::vector<at::TensorList>");
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

    run({OutputSpec{
        tensor.scalar_type(), tensor.device(), tensor.sizes().vec()}});
  }

  // For regular variants
  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor>::value, T>::type call() {
    PT_EAGER_DEBUG("Eager Call regular :: ", m_symbol.toQualString());

    auto result = get_result();
    run({HbEagerTensorPool::get_backend_tensor(result)});
    return result;
  }

  template <typename T = ReturnType>
  typename std::enable_if<is_tuple_of_tensors<T>::value, T>::type call() {
    PT_EAGER_DEBUG("Eager Call tuple_of_tensors :: ", m_symbol.toQualString());
    // TODO avoid calling get_result
    auto result = get_result();

    std::vector<at::Tensor> out_tensors;
    habana::for_each_in_tuple(result, [&out_tensors](const auto& el) {
      out_tensors.emplace_back(HbEagerTensorPool::get_backend_tensor(el));
    });

    run(std::move(out_tensors));

    // TODO: SW-159556 metadata processing should be in one place
    if (m_output_meta_fn) {
      const auto& meta = m_output_meta_fn(get_inputs());
      habana::for_each_in_tuple_with_index(
          result, [&](auto& result, size_t index) {
            if (meta[index].undefined)
              result = at::Tensor();
          });
    }

    return result;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, std::vector<at::Tensor>>::value, T>::
      type
      call() {
    PT_EAGER_DEBUG(
        "Eager Call std::vector<at::Tensor> :: ", m_symbol.toQualString());

    auto result = get_result();

    std::vector<at::Tensor> out_tensors;
    for (auto& el : result) {
      out_tensors.emplace_back(HbEagerTensorPool::get_backend_tensor(el));
    };

    run(std::move(out_tensors));
    return result;
  }

 private:
  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor>::value, T>::type
  get_result() {
    PT_EAGER_TRACE;
    if (m_output_meta_fn) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(m_out_index == 0);
      auto meta = m_output_meta_fn(get_inputs());
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(meta.size() == 1);
      auto output_meta = meta[0];
      auto options = at::TensorOptions(at::kHPU).dtype(output_meta.dtype);
      return at::empty(output_meta.shape, options, output_meta.mem_format);
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
    auto mem_format{at::MemoryFormat::Contiguous};

    return at::empty(out_shape, options, mem_format);
  }

  template <typename T = ReturnType>
  typename std::enable_if<is_tuple_of_tensors<T>::value, T>::type get_result() {
    PT_EAGER_TRACE;

    if (m_output_meta_fn) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(m_out_index == 0);
      const auto& meta = m_output_meta_fn(get_inputs());

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          std::tuple_size<T>::value == meta.size());
      ReturnType results;
      auto options = at::TensorOptions(at::kHPU);
      habana::for_each_in_tuple_with_index(
          results, [&](auto& result, size_t index) {
            auto output_meta = meta[index];
            result = at::empty(
                output_meta.shape,
                options.dtype(output_meta.dtype),
                output_meta.mem_format);
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
                m_out_shapes[index], t.options(), at::MemoryFormat::Contiguous);
          });
    } else {
      HABANA_ASSERT(m_scalar_types.size() == std::tuple_size<T>::value);
      habana::for_each_in_tuple_with_index(
          results, [&](auto& result, size_t index) {
            result = at::empty(
                m_out_shapes[index],
                t.options().dtype(m_scalar_types[index]),
                at::MemoryFormat::Contiguous);
          });
    }
    return results;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, std::vector<at::Tensor>>::value, T>::
      type
      get_result() {
    if (m_output_meta_fn) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(m_out_index == 0);
      const auto& meta = m_output_meta_fn(get_inputs());
      auto options = at::TensorOptions(at::kHPU);
      std::vector<at::Tensor> results;

      results.reserve(meta.size());
      for (const auto& output_meta : meta) {
        results.emplace_back(at::empty(
            output_meta.shape,
            options.dtype(output_meta.dtype),
            output_meta.mem_format));
      }
      return results;
    }
    // Get results from derived class when index is negative
    if (m_out_index < 0) {
      return get_result_overrideable();
    }
    const auto results_size = m_out_shapes.size();
    HABANA_ASSERT(results_size != 0);

    std::vector<at::Tensor> results;
    results.reserve(results_size);
    auto options = at::TensorOptions(at::kHPU);
    if (m_scalar_types.empty()) {
      auto dtype = get_inputs().at(m_out_index).toTensor().scalar_type();
      for (const auto& out_shape : m_out_shapes) {
        results.emplace_back(at::empty(
            out_shape, options.dtype(dtype), at::MemoryFormat::Contiguous));
      }
    } else {
      HABANA_ASSERT(m_scalar_types.size() == results_size);
      for (size_t i = 0; i < results_size; ++i) {
        results.emplace_back(at::empty(
            m_out_shapes[i],
            options.dtype(m_scalar_types[i]),
            at::MemoryFormat::Contiguous));
      }
    }
    return results;
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
};

} // namespace eager
} // namespace habana
