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
#include <absl/container/flat_hash_set.h>
#include <torch/csrc/jit/ir/ir.h>
#include <climits>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backend/helpers/tensor_utils.h"
#include "habana_helpers/logging_pt.h"
#include "habana_helpers/misc_utils.h"

namespace habana_lazy {
struct Data;

const std::string getHabanaLazyGraphName();

namespace ir {

class Node;
struct Value;
class MetaData;

constexpr size_t inlined_inputs_count = 8;
using DataPtr = std::shared_ptr<Data>;
using NodePtr = std::shared_ptr<Node>;
using NodePtrList = std::vector<NodePtr>;
using InlinedNodePtrList = c10::SmallVector<NodePtr, inlined_inputs_count>;
using ValueList = std::vector<Value>;
using InlinedValueList = c10::SmallVector<Value, inlined_inputs_count>;
using ValuePtr = std::shared_ptr<Value>;
using ValuePtrList = std::vector<ValuePtr>;
using IndexToIvalMap = std::unordered_map<size_t, torch::jit::IValue>;

size_t StdHashCombine(uint64_t a, uint64_t b);

void setCurrentModuleName(const std::string& name);
std::shared_ptr<std::string> getCurrentModuleName();

/**
 * Represents the Use of the Value struct as Output
 *
 * The Use struct keeps track of the Output and its usage
 * Currently not used
 */
struct Use {
  Use() = default;
  Use(Node* node, size_t operand_index, size_t index)
      : mp_node(node), m_operand_index(operand_index), m_index(index) {}

  bool operator<(const Use& rhs) const;

  bool operator==(const Use& other) const;

  size_t operator()(const Use& in) const;

  std::string ToString() const;

  Node* mp_node = nullptr;
  size_t m_operand_index = 0;
  size_t m_index = 0;
};

inline std::ostream& operator<<(std::ostream& stream, const Use& use) {
  stream << use.ToString() << "\n";
  return stream;
}

/**
 * Seperate Output class to avoid circular reference
 * in Value, stores raw Node pointer and index of the output
 */
class Output {
 public:
  Output(const Value& v);

  virtual ~Output() {
    m_node = nullptr;
  }

  Node* GetNode() const {
    return m_node;
  }

  size_t GetIndex() const {
    return m_index;
  }

  bool operator==(const Output& v) const {
    return m_node == v.m_node && m_index == v.m_index;
  }

  bool operator!=(const Output& v) const {
    return !(*this == v);
  }

  operator bool() const {
    return m_node != nullptr;
  }

  std::string ToString() const;

  const c10::optional<c10::Device> get_device() const {
    return device;
  }

  const c10::optional<size_t> get_dims() const {
    return dims;
  }

  const c10::optional<at::ScalarType> get_scalar_type() const {
    return scalar_type;
  }

  const c10::optional<SmallSizeVec> get_sizes() const {
    return sizes;
  }

 protected:
  Node* m_node = nullptr;
  size_t m_index;
  // OutInfo
  c10::optional<c10::Device> device;
  c10::optional<size_t> dims;
  c10::optional<SmallSizeVec> sizes;
  c10::optional<at::ScalarType> scalar_type;
  uint64_t unique_id;
};
using OutputList = std::vector<Output>;

/*
 * Class to store the Meta data for an operator
 * Data stored as IValue for now. Supported type
 * of MetaData are similar to IValue supported types
 */

class MetaData {
 public:
  using iterator = IndexToIvalMap::iterator;
  using const_iterator = IndexToIvalMap::const_iterator;

  size_t size() const {
    return m_data.size();
  }

  const torch::jit::IValue& get(size_t index) const {
    HABANA_ASSERT(m_data.count(index));
    return m_data.at(index);
  }

  bool set(const torch::jit::IValue& value, size_t index) {
    return m_data.insert({index, value}).second;
  }

  iterator begin() {
    return m_data.begin();
  }

  iterator end() {
    return m_data.end();
  }

  const_iterator begin() const {
    return m_data.begin();
  }

  const_iterator end() const {
    return m_data.end();
  }

  const_iterator cbegin() const {
    return m_data.cbegin();
  }

  const_iterator cend() const {
    return m_data.cend();
  }

  bool count(size_t key) const {
    return m_data.count(key);
  }

  size_t get_hash() {
    size_t hash = 0;
    for (auto& m : m_data) {
      hash = at::hash_combine(m.first, hash);
      if (m.second.isList()) {
        for (auto& v : m.second.toListRef()) {
          hash = ival_hash(v, hash);
        }
      } else {
        hash = ival_hash(m.second, hash);
      }
    }
    return hash;
  }

  void enableToString() {
    m_enable_to_string = true;
  }

  std::string ToString() const {
    if (!m_enable_to_string) {
      return {};
    }

    std::stringstream ss;
    unsigned i = 0;
    for (const auto& m : m_data) {
      ss << '@' << m.first << '=' << m.second;
      if (i++ != m_data.size() - 1) {
        ss << ", ";
      }
    }
    return ss.str();
  }

  std::string ToStringIrGraph() const {
    if (m_data.size() == 0) {
      return {};
    }

    std::stringstream ss;
    unsigned i = 0;
    for (const auto& m : m_data) {
      ss << "@ " << m.first << '=' << m.second;
      if (i++ != m_data.size() - 1) {
        ss << ", ";
      }
    }
    return ss.str();
  }

  static size_t ival_hash(const torch::jit::IValue& v, size_t h = 0) {
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

 protected:
  /* This meta data store mapping of index of jit input
   * to the IValue
   */
  IndexToIvalMap m_data;

 private:
  bool m_enable_to_string = false;
};

struct Tracker {
  static std::size_t scope_use_count;
  static std::string previous_scope;
};

/**
 * Intermediate struct that connects nodes/operators in Graph
 *
 * The Value struct is an interface for handling different aten
 * types (tensor, scalar, int, double, bool)
 */
struct Value final {
  Value() : unique_id(unique_id_count++) {}
  Value(DataPtr data_ptr, size_t index)
      : unique_id(unique_id_count++), m_data_ptr(data_ptr) {
    m_index = index;
  }

  Value(DataPtr data_ptr)
      : unique_id(unique_id_count++), m_data_ptr(data_ptr) {}

  Value(NodePtr node, size_t index = 0) : unique_id(unique_id_count++) {
    SetNode(node, c10::DeviceType::HPU, {}, {});
    m_index = index;
  }

  Value(const Value& other) = default;
  Value(Value&& other) = default;
  Value& operator=(const Value&) = default;
  Value& operator=(Value&&) = default;

  /* Assigns Node and metadata to Value instance. You should prefer using
   * IrSetNode api instead of calling this directly
   */
  void SetNode(
      NodePtr node,
      const c10::Device& device,
      const SmallSizeVec& dims,
      const c10::optional<at::ScalarType> scalar_type,
      size_t index = 0);

  size_t GetIndex() const {
    return m_index;
  }

  uint64_t get_unique_id() const {
    return unique_id;
  }

  bool operator==(const Value& v) const {
    return mp_node.get() == v.mp_node.get() && m_index == v.m_index;
  }

  bool operator!=(const Value& v) const {
    return !(*this == v);
  }

  operator bool() const {
    return mp_node.get() != nullptr;
  }

  std::string ToString() const;

  std::string ToStringIrGraph() const;

  bool IsHpuInputNode() const;

  bool IsInplaceOnInput() const;

  bool IsInplace() const;
  bool IsAllReduce() const;

  int64_t GetHbLazyTensorUniqueId() const;

  bool DataPtrValid() const;

  bool DataPtrValidAndNotExpired() const;

  const c10::optional<c10::Device> get_device() const {
    return device;
  }

  const c10::optional<size_t> get_dims() const {
    return dims;
  }

  const c10::optional<at::ScalarType> get_scalar_type() const {
    return scalar_type;
  }

  const c10::optional<SmallSizeVec> get_sizes() const {
    return sizes;
  }

  /* Unique id for Value */
  uint64_t unique_id;
  /* The payload field holds the values */
  std::weak_ptr<Data> m_data_ptr;
  /* Value is output of this node */
  NodePtr mp_node = nullptr;
  /**
   * Static global variable used to generate the unique_id for
   * each Value created
   */
  static std::atomic_uint64_t unique_id_count;
  // This keeps track of the version of the data this IR points to
  // helps us track view scenarios where we have RAW or WAR kind of ops on
  // different sections of the same tensor
  uint64_t version_;

 protected:
  // OutInfo
  c10::optional<c10::Device> device;
  c10::optional<size_t> dims;
  c10::optional<SmallSizeVec> sizes;
  c10::optional<at::ScalarType> scalar_type;
  /* The m_index field points to the output index from the node*/
  size_t m_index = 0;
};
/**
 * Node in the IR Graph
 *
 * A Node in an IR Graphs represents an aten operator.
 * Inputs represents connections into this Node (or Operator).
 * Inputs to the Node are in order (as per aten operator schema)
 * num_outputs represent number of outputs generated from this
 * Node.
 *
 */
class Node {
 public:
  Node() = delete;
  Node(c10::Symbol op, bool _is_input = false);

  const c10::Symbol op() const {
    return m_op;
  }

  std::string GetName() const {
    std::stringstream ss;
    ss << "n" << m_id;
    return (!m_scope || m_scope->empty())
        ? m_op.toQualString()
        : *m_scope + "/" + m_op.toQualString();
  }

  void OverrideScope(std::string opname) {
    std::string token = "bmm";
    std::string present_scope = *m_scope;
    if ((present_scope.find(token) == std::string::npos) &&
        (opname.find(token) != std::string::npos)) {
      PT_BRIDGE_DEBUG("previous_scope: ", Tracker::previous_scope);
      PT_BRIDGE_DEBUG("present_scope: ", present_scope);
      if (Tracker::previous_scope != present_scope) {
        Tracker::scope_use_count = 1;
      } else {
        Tracker::scope_use_count++;
      }
      std::string overridden_scope = present_scope + "/" + token;
      if (Tracker::scope_use_count > 1) {
        overridden_scope += std::to_string(Tracker::scope_use_count);
      }
      PT_BRIDGE_DEBUG("overridden_scope: ", overridden_scope);
      m_scope = std::make_shared<std::string>(overridden_scope);
      Tracker::previous_scope = present_scope;
    }
  }

  std::shared_ptr<std::string> GetScope() const {
    return m_scope;
  }

  void SetModuleName(std::string name) {
    module_name = name;
  }

  std::string GetModuleName() {
    return module_name;
  }

  virtual std::string ToString() const;
  virtual std::string ToStringIrGraph() const;

  void AddInput(const Value& value);

  void ReplaceInput(
      const Value& value,
      size_t operand_index,
      const at::Tensor& tensor);

  absl::flat_hash_set<Use, Use>& GetUses() {
    return m_uses;
  }

  const InlinedValueList& GetInputs() const {
    return m_inputs;
  }

  const Output GetOutput(size_t index) const {
    TORCH_CHECK(index < GetNumOutputs(), "Node::GetOutputs index out of range");
    return m_outputs[index];
  }

  virtual ~Node();

  static NodePtr Create(c10::Symbol oper, const InlinedValueList& inputs);

  size_t GetNumOutputs() const {
    return m_outputs.size();
  }

  const MetaData& GetMetaData() const {
    return m_meta_data;
  }

  void SetMetaData(MetaData metadata) {
    m_meta_data = std::move(metadata);
    m_meta_data.enableToString();
  }

  void AddInputPtTensors(std::vector<at::Tensor>& input_pt_vec);

  friend struct Value;

  size_t get_hash();
  size_t get_hash_without_connections();

  bool is_input() const {
    return m_is_input;
  }

  bool is_control_edge() const {
    return m_is_control_edge;
  }
  void set_as_control_edge() {
    m_is_control_edge = true;
  }

  void set_as_output_tensor_list() {
    m_is_output_tensor_list = true;
  }

  bool is_output_tensor_list() const {
    return m_is_output_tensor_list;
  }

  const std::vector<bool>& get_broadcast_details() const {
    return m_bcast_details;
  }

  size_t get_post_order_pos() const {
    return post_order_pos;
  }

  void set_post_order_pos(size_t pos) {
    post_order_pos = pos;
  }

  uint64_t get_id() const {
    return m_id;
  }

  void set_broadcast_details(std::vector<bool>&& bcast_details) {
    m_bcast_details = std::move(bcast_details);
  }

  bool getDeterministic() const {
    return deterministic;
  }

 protected:
  c10::Symbol m_op;
  bool m_is_input = false;
  bool m_is_control_edge = false;
  bool m_is_output_tensor_list = false;
  std::vector<bool> m_bcast_details;
  InlinedValueList m_inputs;
  OutputList m_outputs;
  absl::flat_hash_set<Use, Use> m_uses;
  InlinedNodePtrList m_uses_reverse_nodes;
  MetaData m_meta_data;
  size_t m_node_hash = 0;
  size_t m_node_hash_without_connection = 0;
  size_t post_order_pos = ULLONG_MAX;
  c10::SmallVector<at::Tensor, 8> m_input_pt_tensors;
  std::shared_ptr<std::string> m_scope;
  uint64_t m_id;
  bool deterministic = 0;
  std::unordered_map<uint32_t, uint32_t> m_pt_vec_to_input_ival;
  std::string module_name = std::string();
};

inline std::ostream& operator<<(std::ostream& stream, const Node& node) {
  stream << node.ToString() << "\n";
  return stream;
}

inline std::ostream& operator<<(std::ostream& stream, const Value& value) {
  stream << value.ToString() << "\n";
  return stream;
}

// Hash functor for Output
struct OutputHash {
 public:
  size_t operator()(const Output& v) const {
    return StdHashCombine(
        reinterpret_cast<uintptr_t>(v.GetNode()), v.GetIndex());
  }
};

// Equal functor for Output
struct OutputEqual {
 public:
  bool operator()(const Output& v1, const Output& v2) const {
    return v1 == v2;
  }
};

// Hash functor for Value
struct ValueHash {
 public:
  size_t operator()(const Value& v) const {
    return StdHashCombine(
        reinterpret_cast<uintptr_t>(v.mp_node.get()), v.GetIndex());
  }
};

// Equal functor for Value
struct ValueEqual {
 public:
  bool operator()(const Value& v1, const Value& v2) const {
    return v1 == v2;
  }
};

} // namespace ir
} // namespace habana_lazy

CREATE_OSTREAM_FORMATTER(habana_lazy::ir::Node);
