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
#include <torch/csrc/api/include/torch/jit.h>

namespace habana_lazy {
LazyArgumentSpec::LazyArgumentSpec(
    bool with_grad,
    const ir::NodePtrList& post_order_graph,
    const at::ArrayRef<torch::jit::IValue> input_refs,
    std::string post_order_str) {
  // Create the ArgumentSpec from nodes and inputs
  // ArgumentSpec hash is created based on the inputs
  GetArgSpecKey(with_grad, post_order_graph, input_refs);

  std::hash<std::string> str_hash;
  HABANA_ASSERT(!post_order_str.empty());
  m_opstrs = post_order_str;

  // Create final hash_code by combining the ArgumentSpec
  // and post order nodes strings
  m_hash_code = torch::hash_combine(m_hash_code, str_hash(m_opstrs));
}

torch::jit::Stack LazyArgumentSpec::CreateStack(
    const at::ArrayRef<torch::jit::IValue>& list) {
  // Create a torch::jit::Stack from the IValues
  return torch::jit::Stack(
      std::make_move_iterator(list.begin()),
      std::make_move_iterator(list.end()));
}

void LazyArgumentSpec::GetArgSpecKey(
    bool with_grad,
    const ir::NodePtrList& post_order_graph,
    const at::ArrayRef<torch::jit::IValue>& input_refs) {
  // ArgumentSpecCreator requires a JIT graph to be
  // passed, where the JIT graph inputs are the only
  // content used.
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
  auto graph = torch::jit::compile(jit_graph_str)->get_function("fn").graph();

  torch::jit::ArgumentSpecCreator arg_spec_creator_(*graph);

  // arg_spec_creator_.create takes into account the input tensors.
  torch::jit::ArgumentSpec as =
      arg_spec_creator_.create(with_grad, CreateStack(input_refs));
  m_hash_code = as.hashCode();
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
