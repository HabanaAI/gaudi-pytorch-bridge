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

#include "ir.h"
#include <absl/strings/str_format.h>
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/runtime_config.h"
#include "habana_helpers/logging.h"
#include "lazy_executor.h"

namespace habana_lazy {

const std::string getHabanaLazyGraphName() {
  return "HabanaFusedOpLazy";
}

namespace ir {

size_t StdHashCombine(uint64_t a, uint64_t b) {
  return a ^
      (b * 0x27d4eb2f165667c5 + 0x9e3779b97f4a7c15 + (a << 6) + (a >> 2));
}

// This thread local variable will serve as state to save the current namespace.
// Graph built in the current thread will set it using htcore.set_module_name.
// It will be used to name next nodes created in this graph.
static thread_local std::shared_ptr<std::string> currentModule =
    std::make_shared<std::string>();

void setCurrentModuleName(const std::string& name) {
  currentModule = std::make_shared<std::string>(name);
}

std::shared_ptr<std::string> getCurrentModuleName() {
  return currentModule;
}

/*
 * Initilaize static members from Tracker Class
 */
std::size_t Tracker::scope_use_count(0);
std::string Tracker::previous_scope = std::string();

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

bool Use::operator==(const Use& other) const {
  return mp_node == other.mp_node && m_index == other.m_index &&
      m_operand_index == other.m_operand_index;
};

size_t Use::operator()(const Use& in) const {
  size_t hash = in.m_index;
  hash = at::hash_combine(hash, in.m_operand_index);
  hash = at::hash_combine(hash, reinterpret_cast<size_t>(in.mp_node));
  return hash;
};

Node::Node(c10::Symbol op, bool _is_input)
    : m_op(op),
      m_is_input(_is_input),
      m_is_control_edge(false),
      deterministic(
          habana::HPUGlobalConfig::get().getDeterministic() ||
          at::globalContext().deterministicAlgorithms()) {
  /*Need to set this node if the deterministic mode is ON*/
  SetModuleName(*(habana_lazy::ir::getCurrentModuleName()));
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_DEBUG_NAMES)) {
    static std::atomic<uint64_t> next_id = 0;
    m_id = next_id++;
    m_scope = getCurrentModuleName();
    if (habana_helpers::IsInferenceMode()) {
      if (m_scope && !m_scope->empty()) {
        OverrideScope(m_op.toQualString());
      }
    }
  }
}

std::string Node::ToString() const {
  std::stringstream ss;
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_DEBUG_NAMES)) {
    ss << "n" << m_id << "_";
  }
  ss << m_op.toQualString() << "{";
  for (auto& v : m_inputs) {
    ss << v.ToString() << " ";
  }
  ss << "}\n";
  ss << m_meta_data.ToString();
  if (m_scope && !m_scope->empty()) {
    ss << ", scope=" << *m_scope;
  }
  return ss.str();
}

std::string Node::ToStringIrGraph() const {
  std::stringstream ss;
  ss << m_op.toQualString() << "{";
  for (auto& v : m_inputs) {
    ss << v.ToStringIrGraph() << " ";
  }
  ss << "}\n";
  ss << m_meta_data.ToStringIrGraph();
  return ss.str();
}

void Node::AddInput(const Value& value) {
  if (GET_ENV_FLAG_NEW(PT_HPU_AVOID_RE_EXECUTE_GRAPHS)) {
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
  if (GET_ENV_FLAG_NEW(PT_HPU_AVOID_RE_EXECUTE_GRAPHS)) {
    for (auto node_ptr : m_uses_reverse_nodes) {
      auto node = node_ptr.get();
      if (node) {
        auto& uses = node->GetUses();
        /* Note :
          Ideally we dont need to clear all uses. But if its cleared
          individually, i could see use.mp_node is invalid as it was freed as
          part of postorder/SetNode functions. This 2 cases it will be freed and
          use.mp_node will be dangling and use.mp_node->get_hash() will create
          segfault. This scenario happens while running UT cases all together.
          Probably because the ut teardown is not proper.
          Individually testcases will run without any issues.
        */
        // for (ir::Use use : uses) {
        //   auto hash2 = use.mp_node->get_hash();
        //   if (hash2 == hash1) {
        //     uses.erase(use);
        //   }
        // }
        uses.clear();
      }
    }
  }
  for (const auto& k : m_pt_vec_to_input_ival) {
    auto& tensor = m_input_pt_tensors.at(k.first);
    auto inp = m_inputs.at(k.second);
    if (!inp.IsHpuInputNode() && inp.mp_node &&
        (tensor.getIntrusivePtr().use_count() > 1 ||
         inp.mp_node->GetUses().size() > 1)) {
      PT_LAZY_DEBUG(
          "Node ",
          GetName(),
          " input is ",
          inp.mp_node->GetName(),
          " Uses ",
          inp.mp_node->GetUses().size(),
          " Tensor use_count ",
          tensor.getIntrusivePtr().use_count());

      HABANA_ASSERT(
          inp.mp_node->m_inputs[0].IsHpuInputNode(),
          "Expected mp_node to be inplace, but that's not the case.");
      inp.mp_node->m_input_pt_tensors.emplace_back(tensor);
    }
  }
}

void Value::SetNode(
    NodePtr node,
    const c10::Device& device,
    const SmallSizeVec& dims,
    const c10::optional<at::ScalarType> scalar_type,
    size_t index) {
  if (m_index == 0) {
    // m_index has been set directly, don't reset to 0
    m_index = index;
  }
  this->device = c10::make_optional(device);
  this->dims = c10::make_optional(dims.size());
  this->sizes = c10::make_optional(dims);
  this->scalar_type = scalar_type;
  mp_node = std::move(node);

  mp_node->m_outputs.emplace_back(Output(*this));

  HbContext* devctx = habana_lazy::HbContextArena::Get()->GetHbContext(device);

  auto shared_ptr = m_data_ptr.lock();
  // Collect data_ptr corresponding to all non input lazy tensors (graph
  // outputs)
  // Skip nodes without valid data ptr, e.g. meta-data nodes
  if (shared_ptr == nullptr) {
    return;
  }
  if (!mp_node->is_input()) {
    {
      std::lock_guard<std::recursive_mutex> lock(
          habana_lazy::HbContextArena::Get()->GetMutex());
      devctx->insert(shared_ptr->unique_id, m_data_ptr);
    }
    // Set execution status again to Registered because in case of .out op
    // variants, same tensor may have been considered as input earlier and
    // marked with Execution Complete
    shared_ptr->execution_status = kREGISTERED;
  } else {
    shared_ptr->execution_status = kEXECUTION_COMPLETE;
  }
}

std::string Value::ToString() const {
  std::stringstream ss;
  ss << "id_" << unique_id;
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_DEBUG_NAMES) && mp_node) {
    auto name{mp_node->GetName()};
    std::replace(name.begin(), name.end(), ':', '_');
    ss << "_" << name;
  }
  if (DataPtrValidAndNotExpired()) {
    std::shared_ptr<Data> d = m_data_ptr.lock();
    ss << " Uniqueid: " << d->unique_id;
  }
  return ss.str();
}

std::string Value::ToStringIrGraph() const {
  std::stringstream ss;
  ss << ToString();
  if (DataPtrValidAndNotExpired()) {
    std::shared_ptr<Data> d = m_data_ptr.lock();
    ss << " id:" << unique_id;
    ss << " dims: " << d->sizes;
    ss << " rank: " << d->sizes.size();
  }
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
        c10::Symbol::fromQualString("prim::constant") == inp.mp_node->op() ||
        c10::Symbol::fromQualString("prim::ListConstruct") ==
            inp.mp_node->op()) {
      // Skip this input index for constant and list construct
      // For list construct AddInputPtTensors() is handled separately
      continue;
    }
    input_pt_idx++;
  }

  auto input_idx = 0;
  auto pt_idx = m_input_pt_tensors.size();
  for (const auto& inp : m_inputs) {
    if (inp.IsInplaceOnInput()) {
      PT_LAZY_DEBUG(
          "Node ", GetName(), " candidate Inplace Op ", inp.mp_node->GetName());
      m_input_pt_tensors.emplace_back(inp.mp_node->m_input_pt_tensors[0]);
      inp.mp_node->m_input_pt_tensors.clear();
      m_pt_vec_to_input_ival[pt_idx++] = input_idx;
    }
    input_idx++;
  }
}

NodePtr Node::Create(c10::Symbol oper, const InlinedValueList& inputs) {
  NodePtr node = std::make_shared<Node>(oper);
  for (auto& i : inputs) {
    node->AddInput(i);
  }
  return node;
}

size_t Node::get_hash_without_connections() {
  if (0 == m_node_hash_without_connection) {
    // Op id
    m_node_hash_without_connection = static_cast<uint32_t>(m_op);
    // Op metadata
    m_node_hash_without_connection = at::hash_combine(
        m_node_hash_without_connection, m_meta_data.get_hash());
    // Op deterministic flag

    m_node_hash_without_connection =
        at::hash_combine(m_node_hash_without_connection, deterministic);
  }
  return m_node_hash_without_connection;
}

size_t Node::get_hash() {
  if (0 == m_node_hash) {
    m_node_hash = static_cast<uint32_t>(m_op);
    for (size_t i = 0; i < m_inputs.size(); ++i) {
      m_node_hash = at::hash_combine(m_node_hash, i);
      m_node_hash = at::hash_combine(m_node_hash, m_inputs.at(i).GetIndex());
      m_node_hash =
          at::hash_combine(m_node_hash, m_inputs.at(i).mp_node->get_hash());
    }
    m_node_hash = at::hash_combine(m_node_hash, m_meta_data.get_hash());

    m_node_hash = at::hash_combine(m_node_hash, deterministic);
  }
  return m_node_hash;
}

bool Value::IsHpuInputNode() const {
  // Does it point to an Input node (hpu::input)?
  return mp_node && mp_node->is_input();
}

bool Value::IsInplaceOnInput() const {
  if (mp_node && !mp_node->is_control_edge()) {
    std::string node_name = (std::string)mp_node->op().toQualString();
    auto len = node_name.length();
    if (len && node_name.back() == '_') {
      auto input_mp_node = mp_node->m_inputs[0].mp_node;
      if (input_mp_node && input_mp_node->is_input() &&
          mp_node->m_input_pt_tensors.size() == 1) {
        return true;
      }
    }
  }
  return false;
}

bool Value::IsInplace() const {
  if (mp_node && !mp_node->is_control_edge()) {
    std::string node_name = (std::string)mp_node->op().toQualString();
    auto len = node_name.length();
    if (len && node_name.back() == '_') {
      return true;
    }
  }
  return false;
}
bool Value::IsAllReduce() const {
  if (mp_node && !mp_node->is_control_edge()) {
    std::string node_name = (std::string)mp_node->op().toQualString();
    if (strcmp(node_name.c_str(), "hccl::allreduce_") == 0) {
      return true;
    }
  }
  return false;
}

int64_t Value::GetHbLazyTensorUniqueId() const {
  std::shared_ptr<habana_lazy::Data> d = m_data_ptr.lock();
  return habana_lazy::HbLazyTensor(std::move(d)).getTensorUniqueId();
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

Output::Output(const Value& v)
    : m_node(v.mp_node.get()), m_index(v.GetIndex()) {
  device = v.get_device();
  dims = v.get_dims();
  sizes = v.get_sizes();
  scalar_type = v.get_scalar_type();
  unique_id = v.get_unique_id();
}

std::string Output::ToString() const {
  std::stringstream ss;
  ss << "id_" << unique_id;
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_DEBUG_NAMES)) {
    auto name{m_node->GetName()};
    std::replace(name.begin(), name.end(), ':', '_');
    ss << "_" << name;
  }
  return ss.str();
}

} // namespace ir
} // namespace habana_lazy
