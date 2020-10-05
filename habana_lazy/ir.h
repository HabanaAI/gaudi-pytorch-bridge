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

class HbLazyTensor;
class Node;
struct Value;

using NodePtr = std::shared_ptr<Node>;
using NodePtrList = std::vector<NodePtr>;
using ValuePtr = std::shared_ptr<Value>;
using ValuePtrList = std::vector<ValuePtr>;
using HbLazyTensorPtr = std::shared_ptr<HbLazyTensor>;
using HbLazyTensorPtrList = std::vector<HbLazyTensorPtr>;

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

class Node {
 public:
  Node() = delete;
  Node(c10::Symbol op) : m_op_(op) {}

  const c10::Symbol op() const {
    return m_op_;
  }

  std::string ToString() const;

  void AddInput(const ValuePtr& value);
  void AddOutput(const ValuePtr& value);

  ValuePtr GetOutput(size_t index) const;
  const ValuePtrList GetInputs() const;

  virtual ~Node() {}

  static NodePtr Create(c10::Symbol oper, HbLazyTensorPtrList inputs);

  size_t num_outputs() const {
    return m_outputs_.size();
  }

 private:
  c10::Symbol m_op_;
  // size_t m_num_outputs_ = 1;
  ValuePtrList m_inputs_;
  ValuePtrList m_outputs_;
  std::set<Use> m_uses_;
};

inline std::ostream& operator<<(std::ostream& stream, const Node& node) {
  stream << node.ToString();
  return stream;
}

struct Value {
  Value() {}
  Value(HbLazyTensorPtr tensor, NodePtr node, size_t index)
      : m_hltensor(tensor), mp_node(std::move(node)), m_index(index) {}

  Value(HbLazyTensorPtr tensor) : m_hltensor(tensor) {}

  operator bool() const {
    return mp_node.get() != nullptr;
  }

  std::string ToString() const;

  virtual ~Value() {}

  std::weak_ptr<HbLazyTensor> m_hltensor;
  NodePtr mp_node = nullptr;
  size_t m_index = 0;
};

inline std::ostream& operator<<(std::ostream& stream, const Value& value) {
  stream << value.ToString();
  return stream;
}

} // namespace habana_lazy
