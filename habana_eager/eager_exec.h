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
      const at::Symbol& symbol,
      const SmallTensorVector&
          inputs, // TODO SmallTensorVector is pretty bad for passing
                  // ownership because these are inline we need to copy
      const std::vector<OutputSpec>& outputs,
      habana::eager::MetaDataMap&& metadata)
      : m_symbol{symbol},
        m_inputs{inputs},
        m_outputs{outputs},
        m_metadata{std::move(metadata)} {}
  torch::jit::Stack launch();

 private:
  size_t m_key;
  const at::Symbol& m_symbol;
  SmallTensorVector m_inputs;
  const std::vector<OutputSpec>& m_outputs;
  MetaDataMap m_metadata;

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
  void post_process_eager_graph(
      std::shared_ptr<torch::jit::Graph>& graph,
      const SmallTensorVector& inputs);
};

} // namespace eager
} // namespace habana
