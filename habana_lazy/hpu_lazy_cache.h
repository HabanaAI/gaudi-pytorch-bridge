/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <mutex>
#include "habana_lazy/ir.h"

namespace habana_lazy {
/**
 * LazyGraphCache
 * ----------------
 *
 * Description
 * -----------
 *  This cache holds the optimized JIT graph given an JIT graph.
 *  The execution trigger on lazy mode will do a post-order traversal
 *  of accumulated nodes and create a JIT sub-graph for execution.
 *  If this subgraph has been used before, this cache will return the
 *  optimized JIT graph.
 *
 * Why do we need this cache?
 * --------------------------
 *  The overall goal is to make the critical path of subgraph
 *  execution as fast as possible by avoiding the following on
 *  a cache hit -
 *   - Subgraph (post order) to JIT IR creation
 *   - Optimizing JIT IR graph via JIT compiler
 *   - Creating and compiling a synapse graph via graph compiler
 *
 *  PyTorch TorchScript JIT compiler is triggered on a cache lookup
 *  that is shape unaware.
 *  Consider the following two graphs -
 *  Graph 1:                    Graph 2:
 *    a = tensor(2 x 3)           a' = tensor(200 x 300)
 *    b = tensor(2 x 3)           b' = tensor(200 x 300)
 *    c = a + b                   c' = a' + b'
 *  On JIT IR, its a cache hit (both graphs are on 2D tensors)
 *  On synapse, its a cache miss (tensor shapes are different)
 *
 *  Hence, we need two level of caching. This cache is for the
 *  JIR IR, which detects a cache hit with ArgumentSpec only.
 *
 *  During execution trigger of a lazy subgraph, the expected flow -
 *   - given a sungraph (post order) and its inputs
 *     - do we have a optimized JIT IR graph already?
 *       - If yes,
 *           get the cached optimized JIT graph
 *       - If no,
 *           create a JIT graph from subgraph (post order)
 *           optimize the JIT graph with JIT compiler passes
 *           cache the optimized JIT graph against the subgraph
 *             (post order) and input (dimensions, data types)
 *
 *     - call habana lowering with optimized JIT graph
 *     [This flow below is from the TorchScript lowering bridge code]
 *       - do we have a recipe cached against this optimized JIT graph?
 *         - If Yes, invoke recipe
 *         - If no, create aynspase graphm compile and invoke recipe
 */

/**
 * LazyArgumentSpec
 *
 * This spec creates a hash_code for a given post_order graph
 * and input IValues.
 */
class LazyArgumentSpec {
 public:
  LazyArgumentSpec(
      bool with_grad,
      const ir::NodePtrList& post_order_graph,
      const at::ArrayRef<torch::jit::IValue> input_refs,
      std::string post_order_str);

  bool operator==(const LazyArgumentSpec& rv) const {
    return m_hash_code == rv.m_hash_code && m_opstrs == rv.m_opstrs;
  }

  bool operator!=(const LazyArgumentSpec& rv) const {
    return !(*this == rv);
  }

  size_t hashCode() const {
    return m_hash_code;
  }

 private:
  torch::jit::Stack CreateStack(const at::ArrayRef<torch::jit::IValue>& list);

  void GetArgSpecKey(
      bool with_grad,
      const ir::NodePtrList& post_order_graph,
      const at::ArrayRef<torch::jit::IValue>& input_refs);

  std::string m_opstrs;
  size_t m_hash_code;
};

/**
 * LazyGraphCache
 *
 * This is the Lazy Graph cache.
 * Given a hash_code derived from LazyArgumentSpec for a
 * post order graph and inputs, this cache can be looked up
 * for finding an optimize JIT graph.
 *
 * On a cache miss, the caller is expected to create the optimized
 * JIT graph and add to cache.
 *
 * Cache Lookup
 * ============
 * auto las = LazyArgumentSpec(true, post_order_graph, input_tensors);
 * auto jit_graph =
 * LazyGraphCache::GetLazyCache().GetOptimizedJITGraph(las.hashCode());
 *
 * Cache hit
 * =========
 * if (jit_graph != nullptr) lower_jit_graph(...)
 *
 * Cache miss handling
 * ===================
 * // Create a JIT graph from the post order graph
 * auto jit_graph = Create(post_order_graph, input_tensors);
 * // Create a LazyArgumentSpec
 * auto las = LazyArgumentSpec(true, post_order_graph, input_tensors);
 * LazyGraphCache::GetLazyCache().Add(las.hashCode, jit_graph);
 * lower_jit_graph(...)
 *
 */
class LazyGraphCache {
 public:
  static LazyGraphCache& GetLazyCache() {
    static LazyGraphCache* mp_instance;
    if (!mp_instance) {
      mp_instance = new LazyGraphCache();
    }
    return *mp_instance;
  }

  LazyGraphCache(const LazyGraphCache&) = delete;
  LazyGraphCache(LazyGraphCache&&) = delete;
  LazyGraphCache& operator=(const LazyGraphCache&) = delete;
  LazyGraphCache& operator=(LazyGraphCache&&) = delete;

  ~LazyGraphCache();

  std::shared_ptr<torch::jit::Graph> GetOptimizedJITGraph(size_t key);
  void Add(size_t key, std::shared_ptr<torch::jit::Graph> val);
  void RemoveGraph(size_t key);
  bool IsCached(size_t key);
  bool Empty();
  void Clear();

 private:
  explicit LazyGraphCache();

  std::mutex m_mutex;
  ;
  // Cache stores a JIT graph shared_ptr for a given hash key
  std::unordered_map<size_t, std::shared_ptr<torch::jit::Graph>> m_cache_map;
};

} // namespace habana_lazy
