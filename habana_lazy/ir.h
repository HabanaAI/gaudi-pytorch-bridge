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
#include <climits>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "habana_helpers/logging.h"

namespace habana_lazy {
struct Data;

namespace ir {

inline int64_t mod_exp(int64_t y, int64_t x = 997) {
  const int64_t p{1000000007};
  int64_t z = 1;
  y = y % p;
  if (y == 0) {
    return 0;
  }

  while (y > 0) {
    if (y & 1) {
      z = (z * x) % p;
    }

    y >>= 1;
    x = (x * x) % p;
  }
  return z;
}

inline int64_t mod_exp(bool w, int64_t x = 997) {
  int64_t y = (w ? 97 : 43);
  return (mod_exp(y, x));
}

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

  const c10::optional<c10::Device> get_device() const {
    return device;
  }

  const c10::optional<size_t> get_dims() const {
    return dims;
  }

  const c10::optional<at::ScalarType> get_scalar_type() const {
    return scalar_type;
  }

 protected:
  Node* m_node = nullptr;
  size_t m_index;
  const std::string m_name;
  // OutInfo
  c10::optional<c10::Device> device;
  c10::optional<size_t> dims;
  c10::optional<at::ScalarType> scalar_type;
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

 protected:
  /* This meta data store mapping of index of jit input
   * to the IValue
   */
  IndexToIvalMap m_data;

  size_t ival_hash(const torch::jit::IValue& v, size_t h = 0) {
    if (v.isInt()) {
      return at::hash_combine(h, at::get_hash(mod_exp(v.toInt())));
    } else if (v.isString()) {
      return at::hash_combine(h, at::get_hash(v.toString()));
    } else if (v.isBool()) {
      return at::hash_combine(h, at::get_hash(mod_exp(v.toBool())));
    } else if (v.isScalar()) {
      return at::hash_combine(
          h, c10::WeakIValue(v).hash()); // hash() moved to WeakIvalue
    } else {
      PT_LAZY_WARN(
          "Metadata of type ",
          v.type()->str(),
          " is not hashed. Might get false Lazy IR Cache hits, ",
          "if the value of the constant metadata changes");
    }
    return h;
  }

 private:
  bool m_enable_to_string = false;
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
      : m_op(op), m_is_input(_is_input), m_is_control_edge(false) {}

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

  static NodePtr Create(c10::Symbol oper, const ValueList& inputs);

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

  size_t get_post_order_pos() {
    return post_order_pos;
  }

  void set_post_order_pos(size_t pos) {
    post_order_pos = pos;
  }

 protected:
  c10::Symbol m_op;
  bool m_is_input = false;
  bool m_is_control_edge = false;
  bool m_is_output_tensor_list = false;
  ValueList m_inputs;
  OutputList m_outputs;
  std::set<Use> m_uses;
  MetaData m_meta_data;
  size_t m_node_hash = 0;
  size_t post_order_pos = ULLONG_MAX;
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
    SetNode(node, c10::DeviceType::HPU, {}, {});
    m_index = index;
  }

  void SetNode(
      NodePtr node,
      const c10::Device& device,
      const std::vector<int64_t>& dims,
      const c10::optional<at::ScalarType> scalar_type,
      size_t index = 0) {
    if (m_index == 0) {
      // m_index has been set directly, don't reset to 0
      // TODO: make m_index private.
      m_index = index;
    }
    this->device = c10::make_optional(device);
    this->dims = c10::make_optional(dims.size());
    this->scalar_type = scalar_type;
    mp_node = std::move(node);
    mp_node->m_outputs.emplace_back(Output(*this));
  }

  void SetNodeForShallowCopy(NodePtr node, size_t index = 0) {
    if (m_index == 0) {
      // m_index has been set directly, don't reset to 0
      // TODO: make m_index private.
      m_index = index;
    }
    mp_node = std::move(node);
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

 protected:
  // OutInfo
  c10::optional<c10::Device> device;
  c10::optional<size_t> dims;
  c10::optional<at::ScalarType> scalar_type;
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
