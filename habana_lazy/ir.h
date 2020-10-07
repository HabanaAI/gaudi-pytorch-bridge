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

namespace habana_lazy {

class Node;
struct Value;
struct Data;

using DataPtr = std::shared_ptr<Data>;
using NodePtr = std::shared_ptr<Node>;
using NodePtrList = std::vector<NodePtr>;
using ValueList = std::vector<Value>;
using ValuePtr = std::shared_ptr<Value>;
using ValuePtrList = std::vector<ValuePtr>;

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
  stream << use.ToString();
  return stream;
}

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
  Node(c10::Symbol op, size_t num_outputs)
      : m_op(op), m_num_outputs(num_outputs) {}

  const c10::Symbol op() const {
    return m_op;
  }

  std::string ToString() const;

  void AddInput(const Value& value);

  const ValueList GetInputs() const {
    return m_inputs;
  }

  virtual ~Node() {}

  static NodePtr Create(c10::Symbol oper, ValueList inputs, size_t num_outputs);

  size_t get_num_outputs() const {
    return m_num_outputs;
  }

 private:
  c10::Symbol m_op;
  size_t m_num_outputs = 1;
  ValueList m_inputs;
  std::set<Use> m_uses;
};

inline std::ostream& operator<<(std::ostream& stream, const Node& node) {
  stream << node.ToString();
  return stream;
}

/**
 * Tag Enum class
 *
 * This Enum class represnt if Value is a
 * Tensor, Int, Double, Bool, etc
 * Need to extend for Tensor List, Int List, etc
 */
enum class Tag : uint32_t {
  None = 0,
  Tensor = 1,
  Int = 2,
  Double = 3,
  Bool = 4,
};

/**
 * Intermediate struct that connects nodes/operators in Graph
 *
 * The Value struct is an interface for handling different aten
 * types (tensor, scalar, int, double, bool)
 */
struct Value {
  Value() {}
  Value(DataPtr data_ptr, size_t index)
      : m_data(data_ptr), m_tag(Tag::Tensor), m_index(index) {}

  Value(int val, size_t index) : m_data(val), m_tag(Tag::Int), m_index(index) {}

  Value(double val, size_t index)
      : m_data(val), m_tag(Tag::Double), m_index(index) {}

  Value(bool val, size_t index)
      : m_data(val), m_tag(Tag::Bool), m_index(index) {}

  Value(c10::Scalar val, size_t index);

  Value(DataPtr data_ptr) : m_data(data_ptr), m_tag(Tag::Tensor) {}

  void SetNode(NodePtr node) {
    mp_node = std::move(node);
  }

  operator bool() const {
    return mp_node.get() != nullptr;
  }

  std::string ToString() const;

  virtual ~Value() {}

  bool isTensor() {
    return Tag::Tensor == m_tag;
  }
  bool isInt() {
    return Tag::Int == m_tag;
  }
  bool isDouble() {
    return Tag::Double == m_tag;
  }
  bool isBool() {
    return Tag::Bool == m_tag;
  }
  bool isScalar() {
    return isInt() || isDouble();
  }

  struct Payload {
    std::weak_ptr<Data> m_data_ptr;
    double d = 0;
    int i = 0;
    bool b = false;

    Payload(DataPtr data_ptr) {
      m_data_ptr = data_ptr;
    }
    Payload(double val) {
      d = val;
    }
    Payload(int val) {
      i = val;
    }
    Payload(bool val) {
      b = val;
    }
    Payload() {}
  };

  /* The payload field holds the values */
  Payload m_data;
  Tag m_tag = Tag::None;
  /* The m_index field points to the output index from the node*/
  size_t m_index = 0;
  /* Value is output of this node */
  NodePtr mp_node = nullptr;
};

inline std::ostream& operator<<(std::ostream& stream, const Value& value) {
  stream << value.ToString();
  return stream;
}

} // namespace habana_lazy
