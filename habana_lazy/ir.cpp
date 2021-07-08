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
#include "lazy_executor.h"

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
  ss << m_op.toQualString() << "{";
  for (auto& v : m_inputs) {
    ss << v.ToString() << " ";
  }
  ss << "}\n";
  ss << m_meta_data.ToString();
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

void Node::AddInputPtTensors(std::vector<at::Tensor>& input_pt_vec) {
  // This code assumes that the input tensors are in the same order
  // as the node inputs
  size_t input_pt_idx = 0;
  for (const auto& inp : m_inputs) {
    // If the input value points to a hpu::input node,
    // keep the input_pt_tensor in this node.
    // The reason is to keep the input_pt_tensor alive as long
    // as this node is not yet evaluated
    if (inp.IsHpuInputNode()) {
      HABANA_ASSERT(input_pt_idx < input_pt_vec.size());
      m_input_pt_tensors.emplace_back(input_pt_vec[input_pt_idx]);
    } else if (
        c10::Symbol::fromQualString("prim::constant") == inp.mp_node->op()) {
      // Skip this input index
      continue;
    }
    input_pt_idx++;
  }
}

NodePtr Node::Create(c10::Symbol oper, const ValueList& inputs) {
  NodePtr node = std::make_shared<Node>(oper);
  for (auto& i : inputs) {
    node->AddInput(i);
  }
  return node;
}

size_t Node::get_hash() {
  if (0 == m_node_hash) {
    m_node_hash = static_cast<uint32_t>(m_op);
    for (size_t i = 0; i < m_inputs.size(); ++i) {
      m_node_hash = at::hash_combine(m_node_hash, i);
      if (m_inputs[i]) {
        m_node_hash =
            at::hash_combine(m_node_hash, m_inputs[i].mp_node->get_hash());
      }
    }
    m_node_hash = at::hash_combine(m_node_hash, m_meta_data.get_hash());
  }
  return m_node_hash;
}

bool Value::IsHpuInputNode() const {
  // Does it point to an Input node (hpu::input)?
  return mp_node && mp_node->is_input();
}

bool Value::DataPtrValid() const {
  // Check the owner_before for an empty weak pointer.
  // As per https://en.cppreference.com/w/cpp/memory/weak_ptr/owner_before,
  // "The order is such that two smart pointers compare equivalent only if
  // they are both empty or if they both own the same object"
  // If the weak_ptr is uninitialized, expired() call still returns true as
  // the use_count() is 0 and we can't differentiate an uninitialized tensor
  // against an initialized and expired tensor.
  // The owner_before with an empty weak_ptr is going to return false if the
  // m_data_ptr is uninitialized.
  return m_data_ptr.owner_before(std::weak_ptr<Data>{}) ||
      std::weak_ptr<Data>{}.owner_before(m_data_ptr);
}

bool Value::DataPtrValidAndNotExpired() const {
  return DataPtrValid() && !m_data_ptr.expired();
}

Value::~Value() {}

Output::Output(const Value& v)
    : m_node(v.mp_node.get()), m_index(v.m_index), m_name(v.ToString()) {
  if (v.DataPtrValidAndNotExpired()) {
    std::shared_ptr<Data> d = v.m_data_ptr.lock();
    device = d->device;
    dims = d->sizes.size();
    scalar_type = d->logical_element_type;
  }
}
} // namespace ir
} // namespace habana_lazy
