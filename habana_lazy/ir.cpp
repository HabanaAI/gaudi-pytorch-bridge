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
#include <absl/strings/str_format.h>
#include "habana_helpers/logging.h"
#include "lazy_executor.h"

namespace habana_lazy {
namespace ir {

size_t StdHashCombine(uint64_t a, uint64_t b) {
  return a ^
      (b * 0x27d4eb2f165667c5 + 0x9e3779b97f4a7c15 + (a << 6) + (a >> 2));
}

// This thread local variable will serve as state to save the current namespace.
// Graph built in the current thread will set it using htcore.set_module_name.
// It will be used to name next nodes created in this graph.
static thread_local std::string currentModule;

void setCurrentModuleName(const std::string& name) {
  currentModule = name;
}

const std::string& getCurrentModuleName() {
  return currentModule;
}

/*
 * Initilaize static data from Value Class
 */
std::atomic_uint64_t Value::unique_id_count(0);

bool Use::operator<(const Use& rhs) const {
  if (mp_node != rhs.mp_node) {
    return mp_node < rhs.mp_node;
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
  if (GET_ENV_FLAG(PT_HPU_AVOID_RE_EXECUTE_GRAPHS)) {
    if (value.mp_node) {
      value.mp_node->m_uses.insert({this, m_inputs.size(), value.GetIndex()});
      m_uses_reverse_nodes.push_back(value.mp_node);
    }
  }
  m_inputs.emplace_back(value);
}

void Node::ReplaceInput(
    const Value& value,
    size_t operand_index,
    const at::Tensor& tensor) {
  HABANA_ASSERT(operand_index < m_inputs.size());
  if (m_inputs[operand_index].DataPtrValidAndNotExpired()) {
    m_inputs[operand_index] = value;
    m_input_pt_tensors.emplace_back(tensor);
  }
}

Node::~Node() {
  // auto hash1 = this->get_hash();
  for (auto node_ptr : m_uses_reverse_nodes) {
    auto node = node_ptr.get();
    if (node) {
      auto& uses = node->GetUses();
      /* Note :
        Ideally we dont need to clear all uses. But if its cleared individually,
        i could see use.mp_node is invalid as it was freed as part of
        postorder/SetNode functions. This 2 cases it will be freed and
        use.mp_node will be dangling and use.mp_node->get_hash() will create
        segfault. This scenario happens while running UT cases all together.
        Probably because the ut teardown is not proper.
        Individually testcases will run without any issues.
      */
      uses.clear();
      // for (ir::Use use : uses) {
      //   auto hash2 = use.mp_node->get_hash();
      //   if (hash2 == hash1) {
      //     uses.erase(use);
      //   }
      // }
    }
  }
  m_uses_reverse_nodes.clear();
  m_uses.clear();
  m_inputs.clear();
  m_outputs.clear();
}

void Value::SetNode(
    NodePtr node,
    const c10::Device& device,
    const std::vector<int64_t>& dims,
    const c10::optional<at::ScalarType> scalar_type,
    size_t index) {
  if (m_index == 0) {
    // m_index has been set directly, don't reset to 0
    m_index = index;
  }
  this->device = c10::make_optional(device);
  this->dims = c10::make_optional(dims.size());
  this->scalar_type = scalar_type;
  mp_node = std::move(node);

  if (GET_ENV_FLAG(PT_HPU_ENABLE_DEBUG_NAMES)) {
    this->m_name = absl::StrFormat(
        "t%d_%s_%d", unique_id, mp_node->GetName().c_str(), m_index);
  }

  mp_node->m_outputs.emplace_back(Output(*this));
}

std::string Value::ToString() const {
  if (m_name.empty()) {
    std::stringstream ss;
    ss << "id:" << unique_id;
    return ss.str();
  }
  return m_name;
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
  if (GET_ENV_FLAG(PT_HPU_ENABLE_DEBUG_NAMES)) {
    static std::atomic<uint64_t> id(0);
    node->SetName(absl::StrFormat(
        "n%d_%s/%s", id++, getCurrentModuleName(), node->op().toQualString()));
  }
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
        m_node_hash = at::hash_combine(m_node_hash, m_inputs[i].GetIndex());
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
    : m_node(v.mp_node.get()), m_index(v.GetIndex()), m_name(v.ToString()) {
  device = v.get_device();
  dims = v.get_dims();
  scalar_type = v.get_scalar_type();
}
} // namespace ir
} // namespace habana_lazy
