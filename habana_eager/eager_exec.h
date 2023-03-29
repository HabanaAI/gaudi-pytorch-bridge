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
#include <limits>
#include "backend/jit_graph_cache.h"

namespace habana {
namespace eager {
using MetaDataMap = std::unordered_map<size_t, torch::jit::IValue>;
using SmallTensorVector = c10::SmallVector<at::Tensor, 8>;
struct OutputSpec {
  c10::ScalarType scalar_type;
  c10::Device device;
  c10::IntArrayRef sizes;
};

enum eagerOpKind { OutOfPlace = 0, InplaceOut = 1, Inplace = 2, UnknowType };

struct EagerOpMetaData {
  EagerOpMetaData() {
    op_kind = UnknowType;
    op_name = "";
    out_indices = {};
  }

  EagerOpMetaData(
      eagerOpKind kind,
      std::string name,
      std::vector<int> indices) {
    op_kind = kind;
    op_name = name;
    out_indices = indices;
  }

  std::string to_string() const {
    std::string s = "{ ";
    switch (op_kind) {
      default:
        s.append("UnknowType }");
        return s;
      case OutOfPlace:
        s.append("OutOfPlace }");
        return s;
      case InplaceOut:
        s.append("InplaceOut, ");
        break;
      case Inplace:
        s.append("Inplace, ");
        break;
    }
    s.append(op_name);
    s.append(", {");
    if (!out_indices.empty()) {
      std::stringstream ss;
      std::copy(
          out_indices.begin(),
          out_indices.end(),
          std::ostream_iterator<int>(ss, " "));
      s.append(ss.str());
    }
    s.append("} }");
    return s;
  }

  eagerOpKind op_kind;
  std::string op_name;
  std::vector<int> out_indices;
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
      std::vector<at::Tensor>&& tensor_inputs,
      std::vector<at::IValue>&& inputs,
      std::vector<OutputSpec>&& outputs)
      : m_symbol{symbol},
        m_tensor_inputs(std::move(tensor_inputs)),
        m_inputs(std::move(inputs)),
        m_outputs(std::move(outputs)) {}

  torch::jit::Stack launch();

  void set_eager_op_info(const EagerOpMetaData& eager_op_meta_data) {
    m_eager_op_meta_data = eager_op_meta_data;
  }

 private:
  size_t m_key;
  const at::Symbol m_symbol;
  const std::vector<at::Tensor> m_tensor_inputs;
  const std::vector<at::IValue> m_inputs;
  const std::vector<OutputSpec> m_outputs;
  MetaDataMap m_metadata;
  EagerOpMetaData m_eager_op_meta_data;

  std::shared_ptr<torch::jit::Graph> create_eager_graph();
  size_t calculate_operator_key(const UniqueIdxVec& parent_vec);
  static void update_key_for_tensor(const at::Tensor& t, size_t& optimized_key);
  UniqueIdxVec find_duplicate_in_stack(torch::jit::Stack& stack);
  void prune_duplicate_stack_inputs(
      torch::jit::Stack& stack,
      const UniqueIdxVec& parent_vec);
  void prune_duplicate_graph_inputs(
      const UniqueIdxVec& parent_vec,
      std::shared_ptr<torch::jit::Graph>& graph);
  void post_process_eager_graph(std::shared_ptr<torch::jit::Graph>& graph);
};
std::vector<at::Tensor> convert_inputs_to_backend_tensors(
    std::vector<at::IValue>& inputs);
} // namespace eager
} // namespace habana
