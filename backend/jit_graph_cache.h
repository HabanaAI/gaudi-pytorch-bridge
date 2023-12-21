/*******************************************************************************
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
#pragma once
#include <synapse_api.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <mutex>
#include "backend/habana_operator.h"
#include "backend/helpers/habana_types.h"
#include "backend/kernel/hpu_habana_cache.h"
#include "backend/synapse_helpers/device.h"

namespace habana {

size_t ComputePermutationHashCode(at::ArrayRef<torch::jit::IValue> input_refs);
size_t ComputeSymSizeHashCode(at::ArrayRef<torch::jit::IValue> input_refs);
// Functionality to calculate the graph hash on the JIT graph
void ComputeGraphHashCode(
    const std::shared_ptr<torch::jit::Graph>& irgraph,
    const std::string& id,
    at::ArrayRef<torch::jit::IValue> input_refs,
    std::string& op_strs,
    size_t& graphHashCode,
    uint64_t unique_graph_cntr = 0,
    std::vector<bool> node_bcast_details = {},
    bool dynamic_graph = false);

size_t GetDataChecksum(void* data, size_t dataSize);

//  SynBuildCache class contains information which are calculated during build
//  synapse graph and can be reused for build that graph again. We store in
//  order to speed up build process

class SynBuildCache {
 public:
  template <auto SynBuildCache::*member, typename Func>
  auto& get_or_compute(Func&& comp_func, size_t index) {
    static_assert(
        std::is_member_pointer_v<decltype(member)>,
        "member must be a member pointer");

    if (!is_complete_) {
      HABANA_ASSERT(index == (this->*member).size())
      (this->*member).emplace_back(comp_func());
    }
    return (this->*member).at(index);
  }

  template <auto SynBuildCache::*member, typename Func>
  auto get_or_compute_val(Func&& comp_func, size_t index) {
    static_assert(
        std::is_member_pointer_v<decltype(member)>,
        "member must be a member pointer");

    if (!is_complete_) {
      HABANA_ASSERT(index == (this->*member).size())
      (this->*member).emplace_back(comp_func());
    }
    return (this->*member).at(index);
  }

  void clear_cached_outputs_tensors();
  void clear_cached_graph_info();

  void set_is_control_edge_processing_required() {
    is_control_edge_processing_required = true;
  }

  bool get_is_control_edge_processing_required() {
    return is_control_edge_processing_required;
  }

  void complete() {
    is_complete_ = true;
  }

  bool is_complete() const {
    return is_complete_;
  }

  bool is_complete_ = false;
  std::vector<habana::OutputMetaDataVector> outputs_metadata{};
  VecOfIValPtrSh prim_nodes_ivals{};
  std::vector<std::vector<int64_t>> new_positions{};
  std::vector<bool> is_in_graph_outputs{};
  bool is_control_edge_processing_required = false;
};

struct OptimizedJITGraphAndMetaData {
  OptimizedJITGraphAndMetaData();

  OptimizedJITGraphAndMetaData(
      const std::shared_ptr<torch::jit::Graph> JitGraphToLowering,
      const at::ArrayRef<torch::jit::IValue>& input_refs,
      uint64_t ug_cntr = 0,
      std::vector<bool> node_bcast_details = {},
      const std::string& id = "",
      const bool dynamic = false);

  void ComputeGraphHashCode(
      const std::shared_ptr<torch::jit::Graph> JitGraphToLowering,
      const at::ArrayRef<torch::jit::IValue>& input_refs,
      const std::string& id = "");

  std::shared_ptr<torch::jit::Graph> get_cached_graph() const {
    return jit_graph_to_lowering;
  }

  void set_cached_graph(std::shared_ptr<torch::jit::Graph> graph) {
    jit_graph_to_lowering = graph;
  }

  std::string get_cached_opstrs() {
    return opstrs;
  }

  void set_cached_opstrs(std::string op_strs) {
    opstrs = op_strs;
  }

  size_t get_cached_graph_key() {
    return graphKey;
  }

  void set_cached_graph_key(size_t key) {
    graphKey = key;
  }

  std::string& GetOpName();

  void SetOpName(std::string name);

  size_t GetGraphIndex();

  void SetGraphIndex(size_t index);

  bool GetDbgFlag();

  void SetDbgFlag(bool flag);

  bool GetOptimizedLazyEagerFlag();

  void SetOptimizedLazyEagerFlag(bool flag);

  void set_jit_cached_graph_info_available_flag() {
    syn_build_cache_.complete();
  };

  bool get_jit_cached_graph_info_available_flag() {
    return syn_build_cache_.is_complete();
  };

  void clear_cached_outputs_tensors() {
    syn_build_cache_.clear_cached_outputs_tensors();
  };

  void clear_cached_graph_info() {
    syn_build_cache_.clear_cached_graph_info();
  };

  void set_is_control_edge_processing_required();

  bool get_is_control_edge_processing_required();

  void SetFrontendType(habana_helpers::HabanaFrontendTypes type);

  const habana_helpers::HabanaFrontendTypes& GetFrontendType();

  void set_syn_graph_empty_flag(bool flag) {
    is_syn_graph_empty = flag;
  }

  bool get_syn_graph_empty_flag() const {
    return is_syn_graph_empty;
  }

  void SetHPUStream(synapse_helpers::hpuStream_t stream) {
    hpu_stream = stream;
  }

  synapse_helpers::hpuStream_t GetHPUStream() {
    return hpu_stream;
  }

  std::shared_ptr<habana::RecipeValueSpec> get_shape_agnostic_recipe() {
    return cur_shape_agnostic_rvalpsh;
  }

  void set_shape_agnostic_recipe(
      std::shared_ptr<habana::RecipeValueSpec> shape_agnostic_recipe) {
    cur_shape_agnostic_rvalpsh = shape_agnostic_recipe;
  }

  bool get_is_shape_agnostic_supported() const {
    return is_shape_agnostic_supported;
  }

  void set_is_shape_agnostic_supported(const bool flag) {
    is_shape_agnostic_supported = flag;
  }

  bool get_is_synapse_shape_inf_required() const {
    return is_synapse_sif_required;
  }

  void set_is_synapse_shape_inf_required(const bool flag) {
    is_synapse_sif_required = flag;
  }

  void set_fwd_graph_builder_stack_map(std::vector<uint64_t> stack_idx_map) {
    stack_idx_fwd_graph_builder = stack_idx_map;
  }

  std::vector<uint64_t> get_fwd_graph_builder_stack_map() {
    return stack_idx_fwd_graph_builder;
  }

  void set_is_eager_compiler_supported(bool flag) {
    is_eager_compiler_supported = flag;
  }

  bool get_is_eager_compiler_supported() const {
    return is_eager_compiler_supported;
  }

  void SetDynamicGraph(bool flag) {
    dynamic_graph = flag;
  }

  bool GetDynamicGraph() {
    return dynamic_graph;
  }

  bool get_is_pipeline_supported() const {
    return is_pipeline_supported_;
  }

  void set_is_pipeline_supported(bool is_pipeline_supported) {
    is_pipeline_supported_ = is_pipeline_supported;
  }

  struct PermutationWithOutputPosition {
    uint64_t output_index;
    synapse_helpers::layouts::MemoryPermutation permutation;
  };

  using PermutationInfo = std::vector<PermutationWithOutputPosition>;

  const PermutationInfo& get_permute() const {
    HABANA_ASSERT(permutation_info_.has_value());
    return permutation_info_.value();
  }

  bool is_permute_set() const {
    return permutation_info_.has_value();
  }

  void store_permutation_info(PermutationInfo&& permutation_info) {
    permutation_info_ = std::move(permutation_info);
  }

  SynBuildCache syn_build_cache_;

 private:
  std::shared_ptr<torch::jit::Graph> jit_graph_to_lowering = nullptr;
  std::string opstrs = std::string();
  size_t graphKey = 0;
  bool dbg = false;
  size_t graph_index = 0;
  uint64_t unique_graph_cntr = 0;
  bool dynamic_graph = false;
  std::vector<bool> node_bcast_details;
  std::string op_name = std::string();
  bool isOptimizedLazyEager = false;
  bool is_syn_graph_empty{false};
  synapse_helpers::hpuStream_t hpu_stream = 0;
  std::shared_ptr<habana::RecipeValueSpec> cur_shape_agnostic_rvalpsh{nullptr};
  bool is_shape_agnostic_supported = true;
  bool is_synapse_sif_required = false;
  std::vector<uint64_t> stack_idx_fwd_graph_builder{};
  habana_helpers::HabanaFrontendTypes frontend_type =
      habana_helpers::HabanaFrontendTypes::INVALID;
  bool is_eager_compiler_supported = true;
  bool is_pipeline_supported_ = false;
  std::optional<PermutationInfo> permutation_info_{};
};

/**
 * JitGraphCache
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
class JitGraphCache {
 public:
  static JitGraphCache& GetJitCache() {
    static JitGraphCache mp_instance;
    return mp_instance;
  }

  JitGraphCache(const JitGraphCache&) = delete;
  JitGraphCache(JitGraphCache&&) = delete;
  JitGraphCache& operator=(const JitGraphCache&) = delete;
  JitGraphCache& operator=(JitGraphCache&&) = delete;

  ~JitGraphCache();

  std::shared_ptr<habana::OptimizedJITGraphAndMetaData>
  GetOptimizedJITGraphAndMetaData(size_t key);
  void Add(
      size_t key,
      std::shared_ptr<habana::OptimizedJITGraphAndMetaData> val);
  void RemoveGraph(size_t key);
  bool IsCached(size_t key);
  bool Empty();
  void Clear();

 private:
  explicit JitGraphCache();

  std::mutex m_mutex;
  // Cache stores a JIT graph shared_ptr and meta data for a given hash key
  std::unordered_map<
      size_t,
      std::shared_ptr<habana::OptimizedJITGraphAndMetaData>>
      m_cache_map;
};

class OptimizedJitGraphCache {
 public:
  static OptimizedJitGraphCache& GetOptimizedJitCache() {
    static OptimizedJitGraphCache optimized_mp_instance;
    return optimized_mp_instance;
  }

  OptimizedJitGraphCache(const OptimizedJitGraphCache&) = delete;
  OptimizedJitGraphCache(OptimizedJitGraphCache&&) = delete;
  OptimizedJitGraphCache& operator=(const OptimizedJitGraphCache&) = delete;
  OptimizedJitGraphCache& operator=(OptimizedJitGraphCache&&) = delete;

  std::shared_ptr<habana::OptimizedJITGraphAndMetaData>
  GetOptimizedJITGraphAndMetaData(size_t key);
  void Add(
      size_t key,
      std::shared_ptr<habana::OptimizedJITGraphAndMetaData> val);
  void RemoveGraph(size_t key);
  bool IsCached(size_t key);
  size_t CacheSize();
  bool Empty();
  void Clear();

 private:
  explicit OptimizedJitGraphCache();

  void swap(OptimizedJitGraphCache& cache) {
    std::swap(m_cache_map, cache.m_cache_map);
  }

  std::mutex m_mutex;

  // Cache stores a JIT graph shared_ptr and meta data for a given hash key
  std::unordered_map<
      size_t,
      std::shared_ptr<habana::OptimizedJITGraphAndMetaData>>
      m_cache_map;

  friend class OptimizedJitGraphCacheBackup;
};

class OptimizedJitGraphCacheBackup {
 public:
  OptimizedJitGraphCacheBackup() {
    OptimizedJitGraphCache::GetOptimizedJitCache().swap(cache_);
  }
  ~OptimizedJitGraphCacheBackup() {
    OptimizedJitGraphCache::GetOptimizedJitCache().swap(cache_);
  }

 private:
  OptimizedJitGraphCache cache_{};
};

} // namespace habana
