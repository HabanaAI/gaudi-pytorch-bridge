/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "hpu_lazy_cache.h"

#include <sstream>

#include <torch/csrc/api/include/torch/jit.h>

#include "habana_lazy/aten_lazy_bridge.h"

namespace habana_lazy {

void ComputeGraphHashCode(
    const std::shared_ptr<torch::jit::Graph>& irgraph,
    const std::string& id,
    at::ArrayRef<torch::jit::IValue> input_refs,
    std::string& op_strs,
    size_t& graphHashCode) {
  std::hash<std::string> str_hash;
  op_strs.append((id.empty() ? std::string("UNNAMED") : id) + "::\n");
  std::unordered_map<torch::jit::Node*, size_t> node_idx_map;
  std::unordered_map<size_t, std::string> idx_const_map;
  size_t idx{0};
  for (auto node : irgraph->nodes()) {
    if (node->kind() != torch::jit::prim::Constant) {
      std::string s(node->kind().toQualString());
      s.append("(");
      bool is_start{true};
      for (auto value_in : node->inputs()) {
        auto in_node = value_in->node();
        std::size_t output_index = 0;
        if (in_node) {
          for (output_index = 0; output_index < in_node->outputs().size();
               ++output_index) {
            if (in_node->output(output_index) == value_in) {
              break;
            }
          }
        }
        if (!is_start) {
          s.append(",");
        }
        is_start = false;
        s.append(std::to_string(output_index));
        s.append("_");
        s.append(value_in->node()->kind().toQualString());
      }
      s.append(")");
      // Adding delemeters for better readability
      op_strs.append(s + "\n");
    } else {
      std::ostringstream oss;
      oss << *node;
      std::string cstr = oss.str();
      size_t pos = cstr.find(':');
      if (pos != std::string::npos && pos < cstr.size() - 1)
        cstr = cstr.substr(pos + 1);
      op_strs.append(cstr);
      idx_const_map.emplace(idx, cstr);
    }
    node_idx_map.emplace(node, idx);
    idx++;
  }
  graphHashCode = str_hash(op_strs);

  size_t connection_hash{0};
  // Adding input hash
  for (size_t i = 0; i < irgraph->inputs().size(); ++i) {
    auto value_in = irgraph->inputs().at(i);
    size_t input_connection_hash = i;
    for (auto& use : value_in->uses()) {
      auto node = use.user;
      HABANA_ASSERT(node);
      input_connection_hash =
          at::hash_combine(input_connection_hash, node_idx_map[node]);
    }
    connection_hash = at::hash_combine(connection_hash, input_connection_hash);
  }
  // Adding output hash
  for (size_t i = 0; i < irgraph->outputs().size(); ++i) {
    auto value_out = irgraph->outputs().at(i);
    size_t output_connection_hash = i;
    auto node = value_out->node();
    HABANA_ASSERT(node);
    output_connection_hash =
        at::hash_combine(output_connection_hash, node_idx_map[node]);
    connection_hash = at::hash_combine(connection_hash, output_connection_hash);
  }

  // Adding node connection hash
  size_t node_connection_hash{0};
  for (auto node : irgraph->nodes()) {
    if (node->kind() != torch::jit::prim::Constant) {
      for (auto value_in : node->inputs()) {
        auto in_node = value_in->node();
        if (in_node) {
          if (in_node->kind() != torch::jit::prim::Constant) {
            node_connection_hash =
                at::hash_combine(node_connection_hash, node_idx_map[in_node]);
          } else {
            auto idx = node_idx_map[in_node];
            node_connection_hash = at::hash_combine(
                node_connection_hash, str_hash(idx_const_map.at(idx)));
          }
        }
      }
    }
  }
  connection_hash = at::hash_combine(connection_hash, node_connection_hash);
  graphHashCode = at::hash_combine(graphHashCode, connection_hash);

  // Handle the dims also
  size_t typedims_hash{0};
  for (auto& input : input_refs) {
    if (input.isTensor()) {
      auto pt_tensor = input.toTensor();
      typedims_hash =
          at::hash_combine(typedims_hash, habana::mod_exp(pt_tensor.dim()));
      auto pt_type = pt_tensor.scalar_type();
      int64_t pt_type_int{
          static_cast<std::underlying_type<c10::ScalarType>::type>(pt_type)};
      typedims_hash =
          at::hash_combine(typedims_hash, habana::mod_exp(pt_type_int));
    }
  }
  graphHashCode = at::hash_combine(graphHashCode, typedims_hash);
}

std::unordered_map<size_t, std::shared_ptr<torch::jit::Graph>>
    LazyArgumentSpec::m_compiled_graph;

LazyArgumentSpec::LazyArgumentSpec(
    bool with_grad,
    const at::ArrayRef<torch::jit::IValue>& input_refs,
    size_t post_order_nodes_hash,
    const ir::ValueList& inputs,
    const ir::ValueNodeListMap& value_input_nodes_map,
    const ir::ValueList& outputs,
    const std::vector<size_t>& parent_vec) {
  PT_LAZY_TRACE;
  // Create the ArgumentSpec from nodes and inputs
  // ArgumentSpec hash is created based on the inputs
  GetArgSpecKey(with_grad, input_refs, inputs, value_input_nodes_map, outputs);

  m_post_order_nodes_hash = post_order_nodes_hash;
  HABANA_ASSERT(m_post_order_nodes_hash > 0);

  // Create final hash_code by combining the ArgumentSpec
  // and post order nodes hash
  m_hash_code = at::hash_combine(m_hash_code, m_post_order_nodes_hash);

  // Include num_outputs also part of the hash code
  auto num_outputs = outputs.size();
  m_hash_code = at::hash_combine(m_hash_code, num_outputs);

  // Include the initial duplicate information within the hash code
  for (const auto& a : parent_vec) {
    m_hash_code = at::hash_combine(m_hash_code, a);
  }
}

torch::jit::Stack LazyArgumentSpec::CreateStack(
    const at::ArrayRef<torch::jit::IValue>& list) {
  // Create a torch::jit::Stack from the IValues
  return torch::jit::Stack(
      std::make_move_iterator(list.begin()),
      std::make_move_iterator(list.end()));
}

size_t LazyArgumentSpec::GetInputHash(
    const ir::ValueList& inputs,
    const ir::ValueNodeListMap& value_input_nodes_map) {
  PT_LAZY_TRACE;
  size_t hash_val = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    size_t input_connection_hash = i;
    HABANA_ASSERT(value_input_nodes_map.count(inputs[i]) > 0);
    auto nodes = value_input_nodes_map.at(inputs[i]);
    for (auto& node : nodes) {
      HABANA_ASSERT(node);
      input_connection_hash =
          at::hash_combine(input_connection_hash, node->get_hash());
    }
    hash_val = at::hash_combine(hash_val, input_connection_hash);
  }
  return hash_val;
}

size_t LazyArgumentSpec::GetOutputHash(const ir::ValueList& outputs) {
  PT_LAZY_TRACE;
  size_t hash_val = 0;
  for (size_t i = 0; i < outputs.size(); ++i) {
    size_t output_connection_hash = i;
    if (outputs[i]) {
      auto node = outputs[i].mp_node;
      HABANA_ASSERT(node);
      output_connection_hash =
          at::hash_combine(output_connection_hash, node->get_post_order_pos());
      output_connection_hash =
          at::hash_combine(output_connection_hash, node->get_hash());
      hash_val = at::hash_combine(hash_val, output_connection_hash);
    }
  }
  return hash_val;
}

void LazyArgumentSpec::GetArgSpecKey(
    bool with_grad,
    const at::ArrayRef<torch::jit::IValue>& input_refs,
    const ir::ValueList& inputs,
    const ir::ValueNodeListMap& value_input_nodes_map,
    const ir::ValueList& outputs) {
  // ArgumentSpecCreator requires a JIT graph to be
  // passed, where the JIT graph inputs are the only
  // content used.
  PT_LAZY_TRACE;
  m_hash_code = at::hash_combine(
      m_hash_code, GetInputHash(inputs, value_input_nodes_map));
  m_hash_code = at::hash_combine(m_hash_code, GetOutputHash(outputs));

  auto num_inputs = input_refs.size();

  uint64_t input_hash{};

  torch::jit::ArgumentSpec as(num_inputs, 0);
  for (auto& input : input_refs) {
    as.addTensor(input, with_grad);
  }
  input_hash = as.hashCode();
  m_hash_code = at::hash_combine(m_hash_code, input_hash);

  // Incorporate the memory format of the inputs within hash
  int64_t mf_hash_code{};
  for (auto const& input_ival : input_refs) {
    if (input_ival.isTensor()) {
      auto in_tensor = input_ival.toTensor();
      auto m = in_tensor.suggest_memory_format();
      int64_t m_int =
          static_cast<std::underlying_type<c10::MemoryFormat>::type>(m);
      mf_hash_code =
          at::hash_combine(mf_hash_code, at::get_hash(habana::mod_exp(m_int)));
      if (in_tensor.has_storage()) {
        auto hb_tensor = GetHbInternalTensorImpl(in_tensor);
        if (hb_tensor) {
          auto m_lazy = hb_tensor->GetTensorLayout();
          int64_t m_lazy_int = static_cast<
              std::underlying_type<habana_lazy::LayoutFormat>::type>(m_lazy);
          mf_hash_code = at::hash_combine(
              mf_hash_code, at::get_hash(habana::mod_exp(m_lazy_int)));
        }
      }
    }
  }
  m_hash_code = at::hash_combine(m_hash_code, mf_hash_code);
}

OptimizedJITGraphAndMetaData::OptimizedJITGraphAndMetaData() {}

OptimizedJITGraphAndMetaData::OptimizedJITGraphAndMetaData(
    const std::shared_ptr<torch::jit::Graph> JitGraphToLowering,
    const at::ArrayRef<torch::jit::IValue>& input_refs)
    : jit_graph_to_lowering(JitGraphToLowering) {
  // Compute the graph hash
  ComputeGraphHashCode(JitGraphToLowering, "", input_refs, opstrs, graphKey);
}

std::string& OptimizedJITGraphAndMetaData::GetOpName() {
  return op_name;
}

void OptimizedJITGraphAndMetaData::SetOpName(std::string name) {
  op_name = name;
}

size_t OptimizedJITGraphAndMetaData::GetGraphIndex() {
  return graph_index;
}

void OptimizedJITGraphAndMetaData::SetGraphIndex(size_t index) {
  graph_index = index;
}

bool OptimizedJITGraphAndMetaData::GetDbgFlag() {
  return dbg;
}

void OptimizedJITGraphAndMetaData::SetDbgFlag(bool flag) {
  dbg = flag;
}

bool OptimizedJITGraphAndMetaData::GetOptimizedLazyEagerFlag() {
  return isOptimizedLazyEager;
}

void OptimizedJITGraphAndMetaData::SetOptimizedLazyEagerFlag(bool flag) {
  isOptimizedLazyEager = flag;
}

void OptimizedJITGraphAndMetaData::set_jit_cached_graph_info_available_flag(
    bool flag) {
  isJITCachedGraphInfoAvailable = flag;
}

bool OptimizedJITGraphAndMetaData::get_jit_cached_graph_info_available_flag() {
  return isJITCachedGraphInfoAvailable;
}

void OptimizedJITGraphAndMetaData::set_outputs_metadata(
    habana::OutputMetaDataVector meta_data) {
  outputs_metadata.emplace_back(meta_data);
}

habana::OutputMetaDataVector& OptimizedJITGraphAndMetaData::
    get_outputs_metadata(size_t index) {
  HABANA_ASSERT(index < outputs_metadata.size());
  return outputs_metadata[index];
}

void OptimizedJITGraphAndMetaData::clear_cached_graph_info() {
  outputs_metadata.clear();
  prim_nodes_ivals.clear();
  new_positions.clear();
  is_in_graph_outputs.clear();
  is_control_edge_processing_required = false;
}

void OptimizedJITGraphAndMetaData::set_prim_nodes_ival(IValPtrShared ival) {
  prim_nodes_ivals.emplace_back(ival);
}

IValPtrShared OptimizedJITGraphAndMetaData::get_prim_nodes_ival(size_t index) {
  HABANA_ASSERT(index < prim_nodes_ivals.size());
  return prim_nodes_ivals[index];
}

void OptimizedJITGraphAndMetaData::set_new_pos(std::vector<int64_t> pos) {
  new_positions.emplace_back(pos);
}

std::vector<int64_t>& OptimizedJITGraphAndMetaData::get_new_pos(size_t index) {
  HABANA_ASSERT(index < new_positions.size());
  return new_positions[index];
}

void OptimizedJITGraphAndMetaData::set_is_in_graph_outputs(
    bool is_graph_output) {
  is_in_graph_outputs.emplace_back(is_graph_output);
}

bool OptimizedJITGraphAndMetaData::get_is_in_graph_outputs(size_t index) {
  HABANA_ASSERT(index < is_in_graph_outputs.size());
  return is_in_graph_outputs[index];
}

void OptimizedJITGraphAndMetaData::set_is_control_edge_processing_required(
    bool is_c_edge_required) {
  is_control_edge_processing_required = is_c_edge_required;
}

bool OptimizedJITGraphAndMetaData::get_is_control_edge_processing_required() {
  return is_control_edge_processing_required;
}

// LazyGraphCache Functions
//==========================
LazyGraphCache::LazyGraphCache() : m_mutex{} {}

std::shared_ptr<OptimizedJITGraphAndMetaData> LazyGraphCache::
    GetOptimizedJITGraphAndMetaData(size_t key) {
  std::unique_lock<std::mutex> lck(m_mutex);
  auto iter = m_cache_map.find(key);
  if (iter != m_cache_map.end()) {
    // We found the graph in cache
    // return the optimized graph from the cache
    return iter->second;
  }
  return nullptr;
}

void LazyGraphCache::Add(
    size_t key,
    std::shared_ptr<OptimizedJITGraphAndMetaData> val) {
  TORCH_CHECK(!IsCached(key), "This key is already cached!");

  std::unique_lock<std::mutex> lck(m_mutex);
  m_cache_map.emplace(key, val);
}

void LazyGraphCache::RemoveGraph(size_t key) {
  std::unique_lock<std::mutex> lck(m_mutex);
  auto iter = m_cache_map.find(key);
  if (iter != m_cache_map.end()) {
    m_cache_map.erase(iter);
  }
}

bool LazyGraphCache::IsCached(size_t key) {
  std::unique_lock<std::mutex> lck(m_mutex);
  auto iter = m_cache_map.find(key);
  if (!m_cache_map.empty() && iter != m_cache_map.end()) {
    return true;
  }
  return false;
}

bool LazyGraphCache::Empty() {
  return (m_cache_map.size() == 0);
}

void LazyGraphCache::Clear() {
  m_cache_map.clear();
}

LazyGraphCache::~LazyGraphCache() {
  Clear();
}

// OptimizedLazyGraphCache Functions
//==========================
OptimizedLazyGraphCache::OptimizedLazyGraphCache() : m_mutex{} {}

std::shared_ptr<OptimizedJITGraphAndMetaData> OptimizedLazyGraphCache::
    GetOptimizedJITGraphAndMetaData(size_t key) {
  std::unique_lock<std::mutex> lck(m_mutex);
  auto iter = m_cache_map.find(key);
  if (iter != m_cache_map.end()) {
    // We found the graph in cache
    // return the optimized graph from the cache
    return iter->second;
  }
  return nullptr;
}

void OptimizedLazyGraphCache::Add(
    size_t key,
    std::shared_ptr<OptimizedJITGraphAndMetaData> val) {
  TORCH_CHECK(!IsCached(key), "This key is already cached!");

  std::unique_lock<std::mutex> lck(m_mutex);
  m_cache_map.emplace(key, val);
}

void OptimizedLazyGraphCache::RemoveGraph(size_t key) {
  std::unique_lock<std::mutex> lck(m_mutex);
  auto iter = m_cache_map.find(key);
  if (iter != m_cache_map.end()) {
    m_cache_map.erase(iter);
  }
}

bool OptimizedLazyGraphCache::IsCached(size_t key) {
  std::unique_lock<std::mutex> lck(m_mutex);
  auto iter = m_cache_map.find(key);
  if (!m_cache_map.empty() && iter != m_cache_map.end()) {
    return true;
  }
  return false;
}

bool OptimizedLazyGraphCache::Empty() {
  return (m_cache_map.size() == 0);
}

void OptimizedLazyGraphCache::Clear() {
  m_cache_map.clear();
}

OptimizedLazyGraphCache::~OptimizedLazyGraphCache() {
  Clear();
}

} // namespace habana_lazy
