/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once

#include <limits>

#include "habana_helpers/tensor_utils.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "torch/csrc/jit/ir/ir.h"

// [toDO] this should be Independent of ACC thread MACRO
#define RUN_LAZY_GRAPH_OP_HASH(op_name, inputs)                 \
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GRAPH_RUNNING_HASH)) {     \
    PT_LAZY_TRACE;                                              \
    auto& graph_hash_builder = GraphHashBuilder::getInstance(); \
    graph_hash_builder.graph(op_name, inputs);                  \
    graph_hash_builder.updateRunningHash();                     \
  }

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
  void addNode(ir::Node* node_to_add) {
    node = node_to_add;
  }

  void addNode(const c10::Symbol& node_symbol) {
    m_op = node_symbol;
  }

  const c10::Symbol getOp() const {
    return node->op();
  }

  /**
   * We want the hash code to wor-k based on basics like op and its metadata
   * alone without the input connections
   */
  uint64_t getNodeHash();
  uint64_t getNodeOpHash();

  void populateMetaData(const std::vector<c10::IValue>& input_tensors);

  void updateIndex(size_t node_index) {
    index = node_index;
  }

  size_t& getIndex() {
    return index;
  }
  ir::Node* getNode() {
    return node;
  }

 private:
  bool isMetadataCandidate(const at::IValue& input) const {
    return input.isBool() || input.isDevice() || input.isIntList() ||
        input.isDoubleList() || input.isBoolList() || input.isString() ||
        input.isNone() ||
        (input.isList() &&
         !input.toList().elementType()->cast<at::TensorType>());
  }

  size_t ival_hash(const torch::jit::IValue& v, size_t h = 0) {
    if (v.isInt()) {
      return at::hash_combine(h, at::get_hash(habana::mod_exp(v.toInt())));
    } else if (v.isString()) {
      return at::hash_combine(h, at::get_hash(v.toStringView()));
    } else if (v.isBool()) {
      return at::hash_combine(h, at::get_hash(habana::mod_exp(v.toBool())));
    } else if (v.isScalar()) {
      return at::hash_combine(
          h, c10::WeakIValue(v).hash()); // hash() moved to WeakIvalue
    } else {
      if (!v.isNone() && !v.isDevice()) {
        PT_LAZY_WARN(
            "Metadata of type ",
            v.type()->str(),
            " is not hashed. Might get false Lazy IR Cache hits, ",
            "if the value of the constant metadata changes");
      }
    }
    return h;
  }

  /**
   * TBD: Currently, keeping the ir::Node directly.
   * Eventually, we want to get aways from creating the ir::Node and have a
   * simplfied structure here that only need to record the op, metadata
   *
   */
  ir::Node* node;

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

  void updateRunningHash();

  void addNode(ir::Node* node);
  void addNode(const c10::Symbol& node_symbol);
  void addInputTensors(const std::vector<c10::IValue>& input_tensors);
  void addOutPutTensors(const std::vector<at::Tensor>& output_tensors);
  void graph(
      const c10::Symbol& op_name,
      const std::vector<c10::IValue>& inputs,
      const std::vector<at::Tensor>& outputs);
  void reset() {
    nodes_array.clear();

    graph_input_tensors.clear();
    graph_input_stack_uids.clear();
    graph_input_stack_uid_map.clear();

    fwd_running_hash = 0;
    fwd_inputs_running_hash = 0;

    global_cntr = 0;
  }

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

  OpArrayEntry& getLatestEntry() {
    return nodes_array.back();
  }

  OpArrayEntry& getNodeEntry(size_t idx) {
    return nodes_array.at(idx);
  }

  size_t getCurrentNodeIndex() {
    return nodes_array.size();
  }

  int64_t getTensorRunningId(const at::Tensor& tensor);

  // Forward running hash - gets updated with each op accumulation
  uint64_t fwd_running_hash{0};
  // Running intermediate hash values for the op being added
  uint64_t node_hash, input_hash;
  uint64_t input_hash_running_cntr;
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