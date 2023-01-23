/******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include "backend/jit_graph_cache.h"
#include <sstream>
// WeightIdentificationPass
#include "habana_lazy/passes/pass_utils.cpp"

#include <torch/csrc/api/include/torch/jit.h>

namespace habana {

size_t GetWeightHash(
    const at::ArrayRef<torch::jit::IValue>& input_refs,
    const std::shared_ptr<torch::jit::Graph>& irgraph) {
  habana_lazy::WeightIdentificationPass w_pass;
  size_t hash_code = 0;
  w_pass.markWeightTensors(
      const_cast<std::shared_ptr<torch::jit::Graph>&>(irgraph));
  auto weights = w_pass.getWeightTensors();
  HABANA_ASSERT(input_refs.size() == irgraph->inputs().size());

  for (size_t i = 0; i < input_refs.size(); ++i) {
    auto value_input = irgraph->inputs().at(i);
    if (weights.count(value_input)) {
      hash_code =
          at::hash_combine(hash_code, habana::mod_exp(static_cast<int64_t>(i)));
      HABANA_ASSERT(input_refs[i].isTensor());
      auto& tensor = input_refs[i].toTensor();
      for (size_t shape : tensor.sizes()) {
        hash_code = at::hash_combine(hash_code, shape);
      }
    }
  }

  return hash_code;
}

void ComputeGraphHashCode(
    const std::shared_ptr<torch::jit::Graph>& irgraph,
    const std::string& id,
    at::ArrayRef<torch::jit::IValue> input_refs,
    std::string& op_strs,
    size_t& graphHashCode,
    uint64_t unique_graph_cntr,
    std::vector<bool> node_bcast_details) {
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
  graphHashCode = at::hash_combine(graphHashCode, unique_graph_cntr);

  if (!node_bcast_details.empty()) {
    std::hash<std::vector<bool>> hash_bcast;
    graphHashCode =
        at::hash_combine(graphHashCode, hash_bcast(node_bcast_details));
  }
  if (habana_helpers::GetRefineDynamicShapeStatus()) {
    graphHashCode =
        at::hash_combine(graphHashCode, GetWeightHash(input_refs, irgraph));
  }
}

size_t ComputePermutationHashCode(at::ArrayRef<torch::jit::IValue> input_refs) {
  size_t perm_hash_code = 0;
  uint32_t cnt = 0;
  for (auto& input : input_refs) {
    if (input.isTensor()) {
      auto tensor = input.toTensor();
      auto impl = habana_lazy::GetHbInternalTensorImpl(tensor);
      if (impl) {
        for (auto item : impl->GetMemoryPermutation()) {
          perm_hash_code = at::hash_combine(perm_hash_code, cnt);
          perm_hash_code = at::hash_combine(perm_hash_code, item);
        }
      } else {
        PT_BRIDGE_DEBUG(
            "Could not update cache key with tensor's permutation because the BE tensor has no internal impl");
      }
    }
    cnt++;
  }
  return perm_hash_code;
}

OptimizedJITGraphAndMetaData::OptimizedJITGraphAndMetaData() {}

OptimizedJITGraphAndMetaData::OptimizedJITGraphAndMetaData(
    const std::shared_ptr<torch::jit::Graph> JitGraphToLowering,
    const at::ArrayRef<torch::jit::IValue>& input_refs,
    uint64_t ug_cntr,
    std::vector<bool> bcast_details)
    : jit_graph_to_lowering(JitGraphToLowering),
      unique_graph_cntr(ug_cntr),
      node_bcast_details(bcast_details) {
  // Compute the graph hash
  ComputeGraphHashCode(JitGraphToLowering, input_refs);
}

void OptimizedJITGraphAndMetaData::ComputeGraphHashCode(
    const std::shared_ptr<torch::jit::Graph> JitGraphToLowering,
    const at::ArrayRef<torch::jit::IValue>& input_refs) {
  set_cached_graph_key(0);
  set_cached_opstrs(std::string());
  habana::ComputeGraphHashCode(
      JitGraphToLowering,
      "",
      input_refs,
      opstrs,
      graphKey,
      unique_graph_cntr,
      node_bcast_details);
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

std::shared_ptr<habana::OptimizedJITGraphAndMetaData> LazyGraphCache::
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
    std::shared_ptr<habana::OptimizedJITGraphAndMetaData> val) {
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

std::shared_ptr<habana::OptimizedJITGraphAndMetaData> OptimizedLazyGraphCache::
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
    std::shared_ptr<habana::OptimizedJITGraphAndMetaData> val) {
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

} // namespace habana
