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
#include <limits>
#include "backend/jit_graph_cache.h"

namespace habana {
namespace eager {
using MetaDataMap = std::unordered_map<size_t, torch::jit::IValue>;
using SmallTensorVector = c10::SmallVector<at::Tensor, 8>;

struct OutputSpec {
  c10::ScalarType scalar_type;
  c10::Device device;
  std::vector<int64_t> sizes;
};

class OutputSpecsOrTensors {
 public:
  OutputSpecsOrTensors(std::initializer_list<OutputSpec>&& list)
      : m_outputs(std::vector<OutputSpec>(list.begin(), list.end())) {}
  OutputSpecsOrTensors(std::vector<OutputSpec> specs) : m_outputs(specs){};
  OutputSpecsOrTensors(std::initializer_list<at::Tensor>&& list)
      : m_outputs(std::vector<at::Tensor>(list.begin(), list.end())) {}
  OutputSpecsOrTensors(std::vector<at::Tensor> tensors) : m_outputs(tensors){};
  size_t size();
  c10::TensorTypePtr get_tensor_type(size_t indx);
  std::optional<std::vector<at::Tensor>> get_tensors();
  std::vector<std::vector<int64_t>> get_shapes();
  std::variant<std::vector<OutputSpec>, std::vector<at::Tensor>>& get_outputs();

 private:
  std::variant<std::vector<OutputSpec>, std::vector<at::Tensor>> m_outputs;
};

enum eagerOpKind { OutOfPlace = 0, InplaceOut = 1, Inplace = 2, UnknownType };

struct EagerOpMetaData {
  EagerOpMetaData() : op_kind_(UnknownType) {}

  EagerOpMetaData(
      eagerOpKind kind,
      std::string name,
      std::unordered_set<size_t> out_indices)
      : op_kind_(kind), op_name_(name), out_indices_(out_indices) {}

  EagerOpMetaData(
      eagerOpKind kind,
      std::string name,
      size_t num_out_tensors,
      bool skip_lowering = false)
      : op_kind_(kind),
        op_name_(name),
        num_out_tensors_(num_out_tensors),
        skip_lowering_(skip_lowering) {}

  EagerOpMetaData(
      eagerOpKind kind,
      std::string name,
      bool require_h2d,
      bool require_st,
      std::unordered_set<size_t> out_indices)
      : op_kind_(kind),
        op_name_(name),
        require_h2d_(require_h2d),
        require_st_(require_st),
        out_indices_(out_indices) {}

  EagerOpMetaData(
      eagerOpKind kind,
      std::string name,
      bool require_h2d,
      bool require_st,
      size_t num_out_tensors)
      : op_kind_(kind),
        op_name_(name),
        require_h2d_(require_h2d),
        require_st_(require_st),
        num_out_tensors_(num_out_tensors) {}

  std::string to_string() const {
    std::string s = "{ ";
    switch (op_kind_) {
      default:
        s.append("UnknownType }");
        return s;
      case OutOfPlace:
        s.append("OutOfPlace }");
        return s;
      case InplaceOut:
        s.append("InplaceOut (" + std::to_string(num_out_tensors_) + ")}");
        return s;
      case Inplace:
        s.append("Inplace, ");
        break;
    }
    s.append(op_name_);
    if (require_h2d_) {
      s.append(", Require H2D Tensor");
    }
    if (require_st_) {
      s.append(", Require Shape Tensor");
    }
    s.append(", {");
    if (!out_indices_.empty()) {
      std::stringstream ss;
      std::copy(
          out_indices_.begin(),
          out_indices_.end(),
          std::ostream_iterator<int>(ss, " "));
      s.append(ss.str());
    }
    s.append("} }");
    return s;
  }

  eagerOpKind op_kind_;
  std::string op_name_;
  bool require_h2d_ = false;
  bool require_st_ = false;
  std::unordered_set<size_t> out_indices_;
  std::vector<int64_t> new_strided_insert_output_shape_;
  size_t num_out_tensors_ = 0;
  bool skip_lowering_ = false;
};

/**
 * Wrapper over a vector to store input uniqueness info.
 */
class UniqueIdxVec {
 public:
  using element_t = int64_t;
  UniqueIdxVec(size_t num_inputs) : idx_(num_inputs, UNIQUE_ID) {}
  bool is_duplicate(size_t idx) const {
    return idx_[idx] != UNIQUE_ID;
  }
  element_t& operator[](size_t idx) {
    return idx_[idx];
  }
  element_t operator[](size_t idx) const {
    return idx_[idx];
  }
  size_t size() const {
    return idx_.size();
  }
  std::string to_string() const;

 private:
  c10::SmallVector<element_t, 8> idx_;
  static constexpr element_t UNIQUE_ID{std::numeric_limits<element_t>::max()};
};

class EagerExec {
 public:
  EagerExec(
      at::Symbol symbol,
      std::vector<at::IValue>&& inputs,
      OutputSpecsOrTensors&& outputs,
      bool is_pipeline_supported)
      : m_symbol{symbol},
        m_graph_name{symbol.toQualString()},
        m_inputs(std::move(inputs)),
        m_outputs(std::move(outputs)),
        m_is_pipeline_supported(is_pipeline_supported) {}

  void launch();

  void set_eager_op_info(EagerOpMetaData&& eager_op_meta_data);

  bool check_and_skip_lowering();

 private:
  const at::Symbol m_symbol;
  std::string m_graph_name;
  std::vector<at::IValue> m_inputs;
  OutputSpecsOrTensors m_outputs;
  MetaDataMap m_metadata;
  EagerOpMetaData m_eager_op_meta_data;

  std::shared_ptr<torch::jit::Graph> create_eager_graph(
      torch::jit::Stack& stack,
      CValPtrMap& jit_val_map);
  size_t calculate_operator_key(
      const UniqueIdxVec& parent_vec,
      torch::jit::Stack& stack);
  static void update_key_for_tensor(const at::Tensor& t, size_t& optimized_key);
  UniqueIdxVec find_duplicate_in_stack(torch::jit::Stack& stack);
  void prune_duplicate_stack_inputs(
      torch::jit::Stack& stack,
      const UniqueIdxVec& parent_vec);
  void prune_duplicate_graph_inputs(
      const UniqueIdxVec& parent_vec,
      std::shared_ptr<torch::jit::Graph>& graph);
  torch::jit::Stack prepare_input_stack(const torch::jit::Stack& stack);
  void post_process_eager_graph(
      std::shared_ptr<torch::jit::Graph>& graph,
      CValPtrMap& params_jit_val_map);
  bool is_eager_compiler_supported_for_graph(
      std::shared_ptr<torch::jit::Graph>& graph);
  void mark_maybe_grad_view();
  bool m_is_pipeline_supported = true;
};

std::vector<at::IValue> convert_ivalues_to_backend_tensors(
    std::vector<at::IValue>& ivalues,
    std::optional<at::Symbol> symbol = std::nullopt);
} // namespace eager
} // namespace habana
