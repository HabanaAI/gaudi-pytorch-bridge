/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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

#include "backend/helpers/tensor_utils.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_graph_hash_disabler.h"
#include "torch/csrc/jit/ir/ir.h"

#define RUNNING_HASH_COMBINE_OPERATOR_STR(m_symbol, m_inputs...)    \
  if (habana_lazy::DisableRunningHashUpdates::IsHashingEnabled()) { \
    auto& graph_hash_builder = GraphHashBuilder::getInstance();     \
    graph_hash_builder.updateRunningHash(m_symbol, m_inputs);       \
  }

#define RUNNING_HASH_COMBINE_OPERATOR(op, inputs...) \
  RUNNING_HASH_COMBINE_OPERATOR_STR(c10::Symbol::fromQualString(#op), inputs)

#define RUNNING_HASH_COMBINE_TENSOR(tensor)                         \
  if (habana_lazy::DisableRunningHashUpdates::IsHashingEnabled()) { \
    PT_LAZY_TRACE;                                                  \
    auto& graph_hash_builder = GraphHashBuilder::getInstance();     \
    graph_hash_builder.addInputTensors(tensor);                     \
  }

#define RUNNING_HASH_COMBINE_OUTPUT_TENSOR(tensor) \
  RUNNING_HASH_COMBINE_TENSOR(tensor)

namespace habana_lazy {

#define HPU_FWD_GRAPH_MAX_NODES_IN_GRAPH (std::numeric_limits<size_t>::max())
#define HPU_FWD_GRAPH_INITIAL_NODES_IN_GRAPH (1000)
#define HPU_FWD_GRAPH_PRODUCER_INDEX (HPU_FWD_GRAPH_MAX_NODES_IN_GRAPH)

using producer_index = size_t;
using producer_result_index = size_t;
using consumer_index = size_t;
using tensor_uid = size_t;
using node_id = size_t;

class OpArrayEntry {
 public:
  void addNode(const c10::Symbol& node_symbol) {
    m_op = node_symbol;
  }

  const c10::Symbol getOp() const {
    return m_op;
  }

  /**
   * We want the hash code to wor-k based on basics like op and its metadata
   * alone without the input connections
   */
  uint64_t getNodeOpHash();

  void populateMetaData(const std::vector<c10::IValue>& input_tensors);

  void updateIndex(size_t node_index) {
    index = node_index;
  }

  size_t& getIndex() {
    return index;
  }

 private:
  bool isMetadataCandidate(const at::IValue& input) const;
  size_t ival_hash(const torch::jit::IValue& v, size_t h = 0);

  // index of this node in op accumulation
  size_t index;

  c10::Symbol m_op;

  std::unordered_map<size_t, torch::jit::IValue> meta_data;
};

class GraphHashBuilder {
 public:
  static GraphHashBuilder& getInstance() {
    if (instance == nullptr) {
      instance = new GraphHashBuilder();
    }
    return *instance;
  }

  void addNode(const c10::Symbol& node_symbol);
  void addInputTensors(const std::vector<c10::IValue>& input_tensors);
  /*
   * combines a Tensor to the running hash
   * This is done to include information about auxilary tensor that is not part.
   * of operator argument list. Do not call this directly, but via
   * RUNNING_HASH_COMBINE_TENSOR macro.
   */
  void addInputTensors(const at::Tensor& tensor);
  /*
   * combines an operator to the running hash
   * This includes input tensors and metatada
   * Do not call this directly, but via RUNNING_HASH_COMBINE_OPERATOR macro.
   */
  void updateRunningHash(
      const c10::Symbol& op_name,
      const std::vector<c10::IValue>& inputs);
  void reset();
  void prepareInputsStackMap(const std::vector<ir::Value>& inputs);

  void prepareInputs(
      const std::vector<uint64_t>& input_map,
      std::vector<ir::Value>& inputs);

  uint64_t getFwdRunningHash();

  const std::vector<OpArrayEntry>& getOpEntries() const {
    return nodes_array;
  }

  std::vector<uint64_t> getInputStackMap() {
    return graph_input_stack_uid_map;
  }

  void validateAccumJitOps(std::shared_ptr<torch::jit::Graph> mp_g);

  int64_t getRunningCntr() {
    return global_cntr++;
  }

  void invalidateDeviceTids(c10::Device& device);
  int64_t combineSyncData(
      const std::vector<HbLazyTensor>& tensors,
      const std::vector<int>& indices);

  void set_graph_input_stack_uids(std::vector<uint64_t>&& uids) {
    graph_input_stack_uids = std::move(uids);
  }

  std::vector<uint64_t>& get_input_stack_uid_map() {
    return graph_input_stack_uid_map;
  }

 private:
  GraphHashBuilder() {
    // TBD: Use absl InlinedVector
    nodes_array.reserve(HPU_FWD_GRAPH_INITIAL_NODES_IN_GRAPH);
    graph_input_tensors.reserve(HPU_FWD_GRAPH_INITIAL_NODES_IN_GRAPH);
    graph_input_stack_uids.reserve(HPU_FWD_GRAPH_INITIAL_NODES_IN_GRAPH);
    graph_input_stack_uid_map.reserve(HPU_FWD_GRAPH_INITIAL_NODES_IN_GRAPH);
  }
  ~GraphHashBuilder() {}
  GraphHashBuilder(const GraphHashBuilder&) = delete;
  GraphHashBuilder& operator=(const GraphHashBuilder&) = delete;

  void hashCombineTensor(const at::Tensor& t, size_t& hash);
  void rememberIfInput(const at::Tensor& tensor);

  OpArrayEntry& getLatestEntry() {
    return nodes_array.back();
  }

  OpArrayEntry& getNodeEntry(size_t idx) {
    return nodes_array.at(idx);
  }

  size_t getCurrentNodeIndex() {
    return nodes_array.size();
  }

  // Forward running hash - gets updated with each op accumulation
  uint64_t fwd_running_hash{0};
  // List of nodes, added in the accumulation order
  std::vector<OpArrayEntry> nodes_array;

  // Stack input info
  std::vector<std::weak_ptr<Data>> graph_input_tensors{};
  std::vector<uint64_t> graph_input_stack_uids{};
  std::vector<uint64_t> graph_input_stack_uid_map{};

  static GraphHashBuilder* instance;

  uint64_t fwd_inputs_running_hash{0};

  int64_t global_cntr{0};
};

} // namespace habana_lazy
