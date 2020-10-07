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
#include "habana_helpers/logging.h"

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
  ss << "Op: " << m_op.toQualString() << ", Inputs: {";
  for (auto& v : m_inputs) {
    ss << v.ToString() << " ";
  }
  ss << "}\n";
  return ss.str();
}

void Node::AddInput(const Value& value) {
  m_inputs.emplace_back(value);
}

Value::Value(c10::Scalar val, size_t index) {
  HABANA_ASSERT(val.isFloatingPoint() || val.isIntegral(false));
  if (val.isFloatingPoint()) {
    m_data.d = val.toDouble();
    m_tag = Tag::Double;
  } else if (val.isIntegral(false)) {
    /* includebool is false */
    m_data.i = val.toInt();
    m_tag = Tag::Int;
  }
}

std::string Value::ToString() const {
  std::stringstream ss;
  ss << "tensorname: "
     << "hltensor->name()?\n";
  return ss.str();
}

NodePtr Node::Create(c10::Symbol oper, ValueList inputs, size_t num_outputs) {
  NodePtr node = std::make_shared<Node>(oper, num_outputs);
  for (auto& i : inputs) {
    node->AddInput(i);
  }
  return node;
}

} // namespace habana_lazy