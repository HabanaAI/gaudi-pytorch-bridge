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
#include "hblazy/csrc/lazy_executor.h"

namespace habana_lazy {
namespace ir {

size_t StdHashCombine(uint64_t a, uint64_t b) {
  return a ^
      (b * 0x27d4eb2f165667c5 + 0x9e3779b97f4a7c15 + (a << 6) + (a >> 2));
}
/*
 * Initilaize static data from Value Class
 */
std::atomic_uint64_t Value::unique_id_count(0);

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

std::string Value::ToString() const {
  std::stringstream ss;
  ss << "id:" << unique_id;
  return ss.str();
}

NodePtr Node::Create(c10::Symbol oper, ValueList inputs) {
  NodePtr node = std::make_shared<Node>(oper);
  for (auto& i : inputs) {
    node->AddInput(i);
  }
  return node;
}

Value::~Value() {
  std::shared_ptr<Data> data_ptr = m_data_ptr.lock();
  if (data_ptr) {
    auto tensor = data_ptr->tensor_data;
    if (tensor) {
      auto tensor_val = tensor.value();
      auto context = habana_lazy_executor.getDeviceExecutionContext(
          tensor_val.device().index());
      context->removeRetainedTensor(tensor_val);
    }
  }
}

} // namespace ir
} // namespace habana_lazy
