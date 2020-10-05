/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "ir.h"

namespace habana_lazy {

bool Use::operator<(const Use& rhs) const {
  if (mp_node->op() != rhs.mp_node->op()) {
    return mp_node->op() < rhs.mp_node->op();
  }
  if (m_operand_index != rhs.m_operand_index) {
    return m_operand_index < rhs.m_operand_index;
  }
  return m_index < rhs.m_index;
}

std::string Use::ToString() const {
  std::stringstream ss;
  ss << mp_node->ToString() << ", operand_index=" << m_operand_index
     << ", index=" << m_index;
  return ss.str();
}

std::string Node::ToString() const {
  std::stringstream ss;
  ss << "Op: " << m_op_.toQualString() << ", Inputs: {";
  for (auto& v : m_inputs_) {
    ss << v->ToString() << " ";
  }
  ss << "}, Outputs: {";
  for (auto& v : m_outputs_) {
    ss << v->ToString() << " ";
  }
  ss << "}\n";
  return ss.str();
}

void Node::AddInput(const ValuePtr& value) {
  m_inputs_.emplace_back(value);
}

void Node::AddOutput(const ValuePtr& value) {
  m_outputs_.emplace_back(value);
}

ValuePtr Node::GetOutput(size_t index) const {
  assert(index < num_outputs());
  return m_outputs_.at(index);
}

const ValuePtrList Node::GetInputs() const {
  return m_inputs_;
}

std::string Value::ToString() const {
  std::stringstream ss;
  ss << "tensorname: "
     << "hltensor->name()?\n";
  return ss.str();
}

NodePtr Node::Create(c10::Symbol oper, HbLazyTensorPtrList inputs) {
  NodePtr node = std::make_shared<Node>(oper);
  for (size_t i = 0; i < inputs.size(); ++i) {
    ValuePtr v = std::make_shared<Value>(inputs[i], node, i);
    node->AddInput(v);
  }
  return node;
}

} // namespace habana_lazy