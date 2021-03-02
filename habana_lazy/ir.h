/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#pragma once
#include <torch/csrc/jit/ir/ir.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "habana_helpers/logging.h"

namespace habana_lazy {
struct Data;

namespace ir {

class Node;
struct Value;
class MetaData;

using DataPtr = std::shared_ptr<Data>;
using NodePtr = std::shared_ptr<Node>;
using NodePtrList = std::vector<NodePtr>;
using ValueList = std::vector<Value>;
using ValuePtr = std::shared_ptr<Value>;
using ValuePtrList = std::vector<ValuePtr>;
using IndexToIvalMap = std::map<size_t, torch::jit::IValue>;

size_t StdHashCombine(uint64_t a, uint64_t b);

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

  const std::string& ToString() const {
    return m_name;
  }

 protected:
  Node* m_node = nullptr;
  size_t m_index;
  const std::string m_name;
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

  bool set(torch::jit::IValue value, size_t index) {
    return m_data.insert({index, value}).second;
  }

  iterator begin() {
    return m_data.begin();
  }

  iterator end() {
    return m_data.end();
  }

  const_iterator cbegin() const {
    return m_data.begin();
  }

  const_iterator cend() const {
    return m_data.end();
  }

  bool count(size_t key) const {
    return m_data.count(key);
  }

  size_t get_hash() {
    size_t hash = 0;
    for (auto& m : m_data) {
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

 protected:
  /* This meta data store mapping of index of jit input
   * to the IValue
   */
  IndexToIvalMap m_data;

  size_t ival_hash(const torch::jit::IValue& v, size_t h = 0) {
    if (v.isScalar()) {
      return at::hash_combine(
          c10::WeakIValue(v).hash(), h); // hash() moved to WeakIvalue
    } else if (v.isBool()) {
      size_t bv = v.toBool();
      return at::hash_combine(bv, h);
    }
    return 0;
  }
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
  Node(c10::Symbol op, bool _is_input = false)
      : m_op(op), m_is_input(_is_input) {}

  const c10::Symbol op() const {
    return m_op;
  }

  virtual std::string ToString() const;

  void AddInput(const Value& value);

  const ValueList GetInputs() const {
    return m_inputs;
  }

  const Output GetOutput(size_t index) const {
    TORCH_CHECK(index < GetNumOutputs(), "Node::GetOutputs index out of range");
    return m_outputs[index];
  }

  virtual ~Node() {
    m_inputs.clear();
    m_outputs.clear();
  }

  static NodePtr Create(c10::Symbol oper, ValueList inputs);

  size_t GetNumOutputs() const {
    return m_outputs.size();
  }

  const MetaData& GetMetaData() const {
    return m_meta_data;
  }

  void AddInputPtTensors(std::vector<at::Tensor>& input_pt_vec);

  friend struct Value;

  size_t get_hash();

  bool is_input() const {
    return m_is_input;
  }

 protected:
  c10::Symbol m_op;
  bool m_is_input = false;
  ValueList m_inputs;
  OutputList m_outputs;
  std::set<Use> m_uses;
  MetaData m_meta_data;
  size_t m_node_hash = 0;
  std::vector<at::Tensor> m_input_pt_tensors;
};

inline std::ostream& operator<<(std::ostream& stream, const Node& node) {
  stream << node.ToString() << "\n";
  return stream;
}

/**
 * Intermediate struct that connects nodes/operators in Graph
 *
 * The Value struct is an interface for handling different aten
 * types (tensor, scalar, int, double, bool)
 */
struct Value {
  Value() : unique_id(unique_id_count++) {}
  Value(DataPtr data_ptr, size_t index)
      : unique_id(unique_id_count++), m_data_ptr(data_ptr) {
    m_index = index;
  }

  Value(DataPtr data_ptr)
      : unique_id(unique_id_count++), m_data_ptr(data_ptr) {}

  Value(NodePtr node, size_t index = 0) : unique_id(unique_id_count++) {
    SetNode(node);
    m_index = index;
  }

  void SetNode(NodePtr node) {
    mp_node = node;
    mp_node->m_outputs.emplace_back(Output(*this));
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

  bool IsHpuInputNode() const;

  virtual ~Value();

  /* Unique id for Value */
  uint64_t unique_id;
  /* The payload field holds the values */
  std::weak_ptr<Data> m_data_ptr;
  /* The m_index field points to the output index from the node*/
  size_t m_index = 0;
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
};

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
        reinterpret_cast<uintptr_t>(v.mp_node.get()), v.m_index);
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
