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

namespace habana_lazy {

std::unordered_map<size_t, std::shared_ptr<torch::jit::Graph>>
    LazyArgumentSpec::m_compiled_graph;

LazyArgumentSpec::LazyArgumentSpec(
    bool with_grad,
    const at::ArrayRef<torch::jit::IValue> input_refs,
    size_t post_order_nodes_hash,
    const ir::ValueList inputs,
    const ir::ValueNodeListMap value_input_nodes_map,
    const size_t num_outputs) {
  PT_LAZY_TRACE;
  // Create the ArgumentSpec from nodes and inputs
  // ArgumentSpec hash is created based on the inputs
  GetArgSpecKey(with_grad, input_refs, inputs, value_input_nodes_map);

  m_post_order_nodes_hash = post_order_nodes_hash;
  HABANA_ASSERT(m_post_order_nodes_hash > 0);

  // Create final hash_code by combining the ArgumentSpec
  // and post order nodes hash
  m_hash_code = at::hash_combine(m_hash_code, m_post_order_nodes_hash);

  // Include num_outputs also part of the hash code
  m_hash_code = at::hash_combine(m_hash_code, num_outputs);
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
    hash_val = at::hash_combine(input_connection_hash, hash_val);
  }
  return hash_val;
}

void LazyArgumentSpec::GetArgSpecKey(
    bool with_grad,
    const at::ArrayRef<torch::jit::IValue>& input_refs,
    const ir::ValueList& inputs,
    const ir::ValueNodeListMap& value_input_nodes_map) {
  // ArgumentSpecCreator requires a JIT graph to be
  // passed, where the JIT graph inputs are the only
  // content used.
  PT_LAZY_TRACE;
  std::string jit_graph_inp_str;
  auto num_inputs = input_refs.size();
  unsigned i = 0;
  for (; i < num_inputs - 1; ++i) {
    jit_graph_inp_str.append("n" + std::to_string(i) + ",");
  }
  if (i < num_inputs) {
    jit_graph_inp_str.append("n" + std::to_string(i));
  }
  std::string jit_graph_str = "def fn(" + jit_graph_inp_str +
      "):"
      "  return " +
      jit_graph_inp_str + "\n";

  // Create a JIT graph with dummy inputs for
  // ArgumentSpecCreator.
  std::shared_ptr<torch::jit::Graph> graph;
  m_hash_code = at::hash_combine(m_hash_code, at::get_hash(jit_graph_str));
  m_hash_code = at::hash_combine(
      m_hash_code, GetInputHash(inputs, value_input_nodes_map));

  if (0 == LazyArgumentSpec::m_compiled_graph.count(m_hash_code)) {
    graph = torch::jit::compile(jit_graph_str)->get_function("fn").graph();
    LazyArgumentSpec::m_compiled_graph.insert({m_hash_code, graph});
  } else {
    graph = LazyArgumentSpec::m_compiled_graph.at(m_hash_code);
  }
  torch::jit::ArgumentSpecCreator arg_spec_creator_(*graph);

  // arg_spec_creator_.create takes into account the input tensors.
  torch::jit::ArgumentSpec as =
      arg_spec_creator_.create(with_grad, CreateStack(input_refs));
  m_hash_code = at::hash_combine(m_hash_code, as.hashCode());

  // Incorporate the memory format of the inputs within hash
  i = 0;
  std::ostringstream oss;
  oss << '(';
  for (auto const& input_ival : input_refs) {
    if (input_ival.isTensor()) {
      oss << (i ? "," : "") << i << '_'
          << input_ival.toTensor().suggest_memory_format();
    }
    i++;
  }
  oss << ')';
  std::string mf_str{oss.str()};
  std::hash<std::string> str_hash;
  m_hash_code = at::hash_combine(m_hash_code, str_hash(mf_str));
}

// LazyGraphCache Functions
//==========================
LazyGraphCache::LazyGraphCache() : m_mutex{} {}

std::shared_ptr<torch::jit::Graph> LazyGraphCache::GetOptimizedJITGraph(
    size_t key) {
  std::unique_lock<std::mutex> lck(m_mutex);
  auto iter = m_cache_map.find(key);
  if (iter != m_cache_map.end()) {
    // We found the graph in cache
    // return the optimized graph from the cache
    return iter->second;
  }
  return nullptr;
}

void LazyGraphCache::Add(size_t key, std::shared_ptr<torch::jit::Graph> val) {
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

} // namespace habana_lazy
