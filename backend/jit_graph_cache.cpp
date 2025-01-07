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
#include "backend/jit_graph_cache.h"
#include <sstream>
#define XXH_STATIC_LINKING_ONLY
#define XXH_IMPLEMENTATION
#include <utilities/xxhash.h>
#include "backend/helpers/dynamic_graph_utils.h"
#include "backend/lazy_to_backend.h"
#include "habana_eager/passes/detect_weights_tensors.cpp"

#include <torch/csrc/api/include/torch/jit.h>

namespace habana {

size_t GetWeightHash(
    const at::ArrayRef<torch::jit::IValue>& input_refs,
    const std::shared_ptr<torch::jit::Graph>& irgraph) {
  std::set<int> graph_weights;
  graph::pass::DetectWeightTensors(irgraph, graph_weights);
  size_t hash_code = 0;
  HABANA_ASSERT(input_refs.size() == irgraph->inputs().size());

  for (size_t i = 0; i < input_refs.size(); ++i) {
    if (graph_weights.count(i)) {
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

size_t GetDataChecksum(void* data, size_t dataSize) {
  uint64_t checksum = XXH3_64bits(data, dataSize);
  return checksum;
}

void ComputeGraphHashCode(
    const std::shared_ptr<torch::jit::Graph>& irgraph,
    const std::string& id,
    at::ArrayRef<torch::jit::IValue> input_refs,
    std::string& op_strs,
    size_t& graphHashCode,
    uint64_t unique_graph_cntr,
    std::vector<bool> node_bcast_details,
    bool dynamic_graph,
    const std::map<int64_t, std::vector<int64_t>> m_input_new_base_sizes) {
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
      if (dynamic_graph) {
        size_t pos_comment = cstr.find('#');
        if (pos_comment != std::string::npos && pos_comment < cstr.size() - 1) {
          cstr = cstr.substr(0, pos_comment - 1);
          cstr.append("\n");
        }
      }
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
  size_t constid_hash{0};
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
      if (habana::is_tensor_const_with_valid_const_id(pt_tensor)) {
        auto const_id = habana::get_tensor_const_id(pt_tensor);
        constid_hash = at::hash_combine(constid_hash, const_id);
      }
    }
  }
  // Handle the dims for strided base tensor
  size_t basedims_hash{0};
  for (auto& input : m_input_new_base_sizes) {
    int64_t dim = input.second.size();
    basedims_hash = at::hash_combine(basedims_hash, habana::mod_exp(dim));
  }
  size_t sym_hash = habana::ComputeSymSizeHashCode(input_refs);
  graphHashCode = at::hash_combine(graphHashCode, sym_hash);
  graphHashCode = at::hash_combine(graphHashCode, typedims_hash);
  graphHashCode = at::hash_combine(graphHashCode, basedims_hash);
  graphHashCode = at::hash_combine(graphHashCode, unique_graph_cntr);
  graphHashCode = at::hash_combine(graphHashCode, constid_hash);

  if (!node_bcast_details.empty()) {
    std::hash<std::vector<bool>> hash_bcast;
    graphHashCode =
        at::hash_combine(graphHashCode, hash_bcast(node_bcast_details));
  }
  if (dynamic_graph) {
    graphHashCode =
        at::hash_combine(graphHashCode, GetWeightHash(input_refs, irgraph));
  }
}

size_t ComputeNodeSymOutputHashCode(
    const std::shared_ptr<torch::jit::Graph>& jit_graph) {
  std::hash<std::string> str_hash;
  size_t sym_output_hash_code = 0;
  bool has_valid_hash = false;

  for (auto node : jit_graph->nodes()) {
    if ((torch::jit::prim::Constant != node->kind()) &&
        (torch::jit::prim::ListConstruct != node->kind())) {
      auto outputshapes_attr = c10::Symbol::attr("output_shapes");
      std::string shape_str = "";
      if (node->hasAttribute(outputshapes_attr)) {
        shape_str = node->s(outputshapes_attr);
      }
      bool is_symbolic = habana_helpers::is_symbolic_expr(shape_str);
      has_valid_hash = is_symbolic || (shape_str == "") ||
          (shape_str.find("[]") != std::string::npos);
      sym_output_hash_code =
          at::hash_combine(sym_output_hash_code, str_hash(shape_str));
    }
  }

  if (!has_valid_hash) {
    sym_output_hash_code = ULONG_MAX;
    PT_DYNAMIC_SHAPE_DEBUG(
        "Symbolic expressions doesnot contain real symbols, dynamic symbolic hash is invalid!!!");
  }

  return sym_output_hash_code;
}

size_t ComputePermutationHashCode(at::ArrayRef<torch::jit::IValue> input_refs) {
  size_t perm_hash_code = 0;
  uint32_t cnt = 0;
  for (auto& input : input_refs) {
    if (input.isTensor()) {
      auto tensor = input.toTensor();
      if (!habana::get_tensor_extra_meta(tensor)->is_shape_tensor()) {
        synapse_helpers::layouts::MemoryPermutation permutation;
        std::tie(permutation, std::ignore) =
            habana_helpers::get_tensor_memory_permutation(tensor);
        for (auto item : permutation) {
          perm_hash_code = at::hash_combine(perm_hash_code, cnt);
          perm_hash_code = at::hash_combine(perm_hash_code, item);
        }
      }
    }
    cnt++;
  }
  return perm_hash_code;
}

size_t ComputeSymSizeHashCode(at::ArrayRef<torch::jit::IValue> input_refs) {
  size_t sym_hash_code = 0;
  uint32_t cnt = 0;
  for (auto& input : input_refs) {
    if (!input.isTensor() && input.isScalar()) {
      // Add the hashing for SymInts/SymFloats
      auto scalar_input = input.toScalar();
      size_t symsize_hash{0};
      if (input.isInt()) {
        int64_t value = input.toScalar().toLong();
        std::hash<int64_t> valhash;
        symsize_hash = at::hash_combine(symsize_hash, valhash(value));
      } else if (input.isBool()) {
        auto value = input.toScalar().toBool();
        symsize_hash = at::hash_combine(symsize_hash, value);
      } else if (input.isDouble()) {
        auto value = input.toScalar().toDouble();
        std::hash<double> valhash;
        symsize_hash = at::hash_combine(symsize_hash, valhash(value));
      } else {
        HABANA_ASSERT("Unhandled Scalar");
      }
      sym_hash_code = at::hash_combine(sym_hash_code, cnt);
      sym_hash_code = at::hash_combine(sym_hash_code, symsize_hash);
    }
    cnt++;
  }
  return sym_hash_code;
}

OptimizedJITGraphAndMetaData::OptimizedJITGraphAndMetaData() {}

OptimizedJITGraphAndMetaData::OptimizedJITGraphAndMetaData(
    const std::shared_ptr<torch::jit::Graph> JitGraphToLowering,
    const at::ArrayRef<torch::jit::IValue>& input_refs,
    uint64_t ug_cntr,
    std::vector<bool> bcast_details,
    const std::string& id,
    const bool dynamic,
    const std::map<int64_t, std::vector<int64_t>> m_input_new_base_sizes)
    : jit_graph_to_lowering(JitGraphToLowering),
      unique_graph_cntr(ug_cntr),
      dynamic_graph(dynamic),
      node_bcast_details(bcast_details) {
  // Compute the graph hash
  ComputeGraphHashCode(
      JitGraphToLowering, input_refs, id, m_input_new_base_sizes);
}

void OptimizedJITGraphAndMetaData::ComputeGraphHashCode(
    const std::shared_ptr<torch::jit::Graph> JitGraphToLowering,
    const at::ArrayRef<torch::jit::IValue>& input_refs,
    const std::string& id,
    const std::map<int64_t, std::vector<int64_t>> m_input_new_base_sizes) {
  set_cached_graph_key(0);
  set_cached_opstrs(std::string());
  habana::ComputeGraphHashCode(
      JitGraphToLowering,
      id,
      input_refs,
      opstrs,
      graphKey,
      unique_graph_cntr,
      node_bcast_details,
      dynamic_graph,
      m_input_new_base_sizes);
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

void OptimizedJITGraphAndMetaData::SetOptimizedLazyEagerFlag(bool flag) {
  isOptimizedLazyEager = flag;
}

void OptimizedJITGraphAndMetaData::SetUserMarkDynamic(bool flag) {
  user_mark_dynamic = flag;
}

bool OptimizedJITGraphAndMetaData::IsUserMarkDynamic() {
  return (user_mark_dynamic == true);
}

void OptimizedJITGraphAndMetaData::SetUserRangesDynamic(
    std::vector<habana_helpers::RangeInfo>& range_infos) {
  m_range_infos = range_infos;
}

std::vector<habana_helpers::RangeInfo> OptimizedJITGraphAndMetaData::
    GetUserRangesDynamic() {
  return m_range_infos;
}

void SynBuildCache::clear_cached_outputs_tensors() {
  for (auto& metadatas : outputs_metadata) {
    for (auto& metadata : metadatas) {
      if (metadata.allocated_tensor.has_value()) {
        metadata.allocated_tensor.reset();
      }
    }
  }
}

void SynBuildCache::clear_cached_graph_info() {
  outputs_metadata.clear();
  prim_nodes_ivals.clear();
  new_positions.clear();
  is_in_graph_outputs.clear();
  is_control_edge_processing_required = false;
  is_complete_ = false;
}

void OptimizedJITGraphAndMetaData::SetFrontendType(
    habana_helpers::HabanaFrontendTypes type) {
  frontend_type = type;
}

const habana_helpers::HabanaFrontendTypes& OptimizedJITGraphAndMetaData::
    GetFrontendType() {
  return frontend_type;
}

// JitGraphCache Functions
//==========================
JitGraphCache::JitGraphCache() : m_mutex{} {}

std::shared_ptr<habana::OptimizedJITGraphAndMetaData> JitGraphCache::
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

void JitGraphCache::Add(
    size_t key,
    std::shared_ptr<habana::OptimizedJITGraphAndMetaData> val) {
  TORCH_CHECK(!IsCached(key), "This key is already cached!");

  std::unique_lock<std::mutex> lck(m_mutex);
  m_cache_map.emplace(key, val);
}

void JitGraphCache::RemoveGraph(size_t key) {
  std::unique_lock<std::mutex> lck(m_mutex);
  auto iter = m_cache_map.find(key);
  if (iter != m_cache_map.end()) {
    m_cache_map.erase(iter);
  }
}

bool JitGraphCache::IsCached(size_t key) {
  std::unique_lock<std::mutex> lck(m_mutex);
  auto iter = m_cache_map.find(key);
  if (!m_cache_map.empty() && iter != m_cache_map.end()) {
    return true;
  }
  return false;
}

bool JitGraphCache::Empty() {
  return (m_cache_map.size() == 0);
}

void JitGraphCache::Clear() {
  m_cache_map.clear();
}

JitGraphCache::~JitGraphCache() {
  Clear();
}

// OptimizedJitGraphCache Functions
//==========================
OptimizedJitGraphCache::OptimizedJitGraphCache() : m_mutex{} {}

std::shared_ptr<habana::OptimizedJITGraphAndMetaData> OptimizedJitGraphCache::
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

void OptimizedJitGraphCache::Add(
    size_t key,
    std::shared_ptr<habana::OptimizedJITGraphAndMetaData> val) {
  TORCH_CHECK(!IsCached(key), "This key is already cached!");

  std::unique_lock<std::mutex> lck(m_mutex);
  m_cache_map.emplace(key, val);
}

void OptimizedJitGraphCache::RemoveGraph(size_t key) {
  std::unique_lock<std::mutex> lck(m_mutex);
  auto iter = m_cache_map.find(key);
  if (iter != m_cache_map.end()) {
    m_cache_map.erase(iter);
  }
}

bool OptimizedJitGraphCache::IsCached(size_t key) {
  std::unique_lock<std::mutex> lck(m_mutex);
  auto iter = m_cache_map.find(key);
  if (!m_cache_map.empty() && iter != m_cache_map.end()) {
    return true;
  }
  return false;
}

size_t OptimizedJitGraphCache::CacheSize() {
  return m_cache_map.size();
}

bool OptimizedJitGraphCache::Empty() {
  return (m_cache_map.size() == 0);
}

void OptimizedJitGraphCache::Clear() {
  m_cache_map.clear();
}

} // namespace habana
