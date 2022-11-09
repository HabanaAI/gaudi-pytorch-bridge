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
#include <synapse_api.h>
#include <synapse_helpers/device.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <mutex>
#include "habana_bridge/kernel/hpu_habana_cache.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/habana_operator.h"
#include "habana_lazy/ir.h"
#include "habana_lazy/ir_utils.h"

namespace habana_lazy {

size_t ComputePermutationHashCode(at::ArrayRef<torch::jit::IValue> input_refs);
// Functionality to calculate the graph hash on the JIT graph
void ComputeGraphHashCode(
    const std::shared_ptr<torch::jit::Graph>& irgraph,
    const std::string& id,
    at::ArrayRef<torch::jit::IValue> input_refs,
    std::string& op_strs,
    size_t& graphHashCode,
    uint64_t unique_graph_cntr = 0,
    std::vector<bool> node_bcast_details = {});
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
      const at::ArrayRef<torch::jit::IValue>& input_refs,
      size_t post_order_nodes_hash,
      const ir::ValueList& inputs,
      const ir::ValueNodeListMap& value_input_nodes_map,
      const ir::ValueList& outputs,
      const std::vector<size_t>& parent_vec,
      const std::vector<bool>& node_bcast_map = {});

  bool operator==(const LazyArgumentSpec& rv) const {
    return m_hash_code == rv.m_hash_code &&
        m_post_order_nodes_hash == rv.m_post_order_nodes_hash;
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
      const at::ArrayRef<torch::jit::IValue>& input_refs,
      const ir::ValueList& inputs,
      const ir::ValueNodeListMap& value_input_nodes_map,
      const ir::ValueList& outputs);

  size_t GetInputHash(
      const ir::ValueList& inputs,
      const ir::ValueNodeListMap& value_input_nodes_map);

  size_t GetOutputHash(const ir::ValueList& outputs);

  size_t m_post_order_nodes_hash;
  size_t m_hash_code = 0;

  //
  // Cache for storing the compiled graph
  static std::unordered_map<size_t, std::shared_ptr<torch::jit::Graph>>
      m_compiled_graph;
};

struct OptimizedJITGraphAndMetaData {
  OptimizedJITGraphAndMetaData();
  OptimizedJITGraphAndMetaData(
      const std::shared_ptr<torch::jit::Graph> JitGraphToLowering,
      const at::ArrayRef<torch::jit::IValue>& input_refs,
      uint64_t ug_cntr = 0,
      std::vector<bool> node_bcast_details = {});

  void ComputeGraphHashCode(
      const std::shared_ptr<torch::jit::Graph> JitGraphToLowering,
      const at::ArrayRef<torch::jit::IValue>& input_refs);

  std::shared_ptr<torch::jit::Graph> get_cached_graph() {
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

  void set_jit_cached_graph_info_available_flag(bool flag);

  bool get_jit_cached_graph_info_available_flag();

  void set_outputs_metadata(habana::OutputMetaDataVector meta_data);

  habana::OutputMetaDataVector& get_outputs_metadata(size_t index);

  void clear_cached_graph_info();

  void set_prim_nodes_ival(IValPtrShared ival);

  IValPtrShared get_prim_nodes_ival(size_t index);

  void set_new_pos(std::vector<int64_t> pos);

  std::vector<int64_t>& get_new_pos(size_t index);

  void set_is_in_graph_outputs(bool is_graph_output);

  bool get_is_in_graph_outputs(size_t index);

  void set_is_control_edge_processing_required(bool is_c_edge_required);

  bool get_is_control_edge_processing_required();

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

  void SetEventHandle(synEventHandle handle) {
    event_handle = handle;
  }

  synEventHandle GetEventHandle() {
    return event_handle;
  }

  void SetEventRecordStream(synapse_helpers::hpuStream_t stream) {
    event_stream = stream;
  }

  synapse_helpers::hpuStream_t GetEventRecordStream() {
    return event_stream;
  }

  void SetEventFlag(bool flag) {
    event_flag = flag;
  }

  bool GetEventFlag() {
    return event_flag;
  }

  std::vector<std::vector<int64_t>> get_output_shapes() {
    return output_shapes;
  }

  void set_output_shapes(std::vector<std::vector<int64_t>> shapes) {
    output_shapes = shapes;
  }

  std::shared_ptr<habana::RecipeValueSpec> get_shape_agnostic_recipe() {
    return cur_shape_agnostic_rvalpsh;
  }

  void set_shape_agnostic_recipe(
      std::shared_ptr<habana::RecipeValueSpec> shape_agnostic_recipe) {
    cur_shape_agnostic_rvalpsh = shape_agnostic_recipe;
  }

  std::unordered_map<uint64_t, synTensor>
  get_syn_tensor_id_to_tensor_handle_map() {
    return syn_tensor_id_to_tensor_handle;
  }

  void set_syn_tensor_id_to_tensor_handle_map(
      std::unordered_map<uint64_t, synTensor> tensor_id_to_tensor_handle_map) {
    syn_tensor_id_to_tensor_handle = tensor_id_to_tensor_handle_map;
  }

  bool get_is_shape_agnostic_supported() {
    return is_shape_agnostic_supported;
  }

  void set_is_shape_agnostic_supported(bool flag) {
    is_shape_agnostic_supported = flag;
  }

  void set_fwd_graph_builder_stack_map(std::vector<uint64_t> stack_idx_map) {
    stack_idx_fwd_graph_builder = stack_idx_map;
  }

  std::vector<uint64_t> get_fwd_graph_builder_stack_map() {
    return stack_idx_fwd_graph_builder;
  }

 private:
  std::shared_ptr<torch::jit::Graph> jit_graph_to_lowering = nullptr;
  std::string opstrs = std::string();
  size_t graphKey = 0;
  bool dbg = false;
  size_t graph_index = 0;
  uint64_t unique_graph_cntr = 0;
  std::vector<bool> node_bcast_details;
  std::string op_name = std::string();
  bool isOptimizedLazyEager = false;
  bool isJITCachedGraphInfoAvailable = false;
  std::vector<habana::OutputMetaDataVector> outputs_metadata{};
  std::vector<IValPtrShared> prim_nodes_ivals{};
  std::vector<std::vector<int64_t>> new_positions{};
  std::vector<bool> is_in_graph_outputs{};
  bool is_control_edge_processing_required = false;
  bool is_syn_graph_empty{false};
  synapse_helpers::hpuStream_t hpu_stream = 0;
  synEventHandle event_handle{nullptr};
  synapse_helpers::hpuStream_t event_stream = 0;
  bool event_flag = false;
  std::vector<std::vector<int64_t>> output_shapes{};
  std::shared_ptr<habana::RecipeValueSpec> cur_shape_agnostic_rvalpsh{nullptr};
  std::unordered_map<uint64_t, synTensor> syn_tensor_id_to_tensor_handle{};
  bool is_shape_agnostic_supported = true;
  std::vector<uint64_t> stack_idx_fwd_graph_builder{};
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
 * auto jit_graph_and_meta_data =
 * LazyGraphCache::GetLazyCache().GetOptimizedJITGraphAndMetaData(las.hashCode());
 *
 * Cache hit
 * =========
 * if (jit_graph_and_metat_data != nullptr) lower_jit_graph(...)
 *
 * Cache miss handling
 * ===================
 * // Create a JIT graph from the post order graph
 * auto jit_graph = Create(post_order_graph, input_tensors);
 * // Create a LazyArgumentSpec
 * auto las = LazyArgumentSpec(true, post_order_graph, input_tensors);
 * // Compute meta data for JIT graph and store in cache along with JIT graph
 * auto jit_graph_and_meta_data =
 * std::make_shared<OptimizedJITGraphAndMetaData>(jit_graph, input_refs);
 * LazyGraphCache::GetLazyCache().Add(las.hashCode, jit_graph_and_meta_data);
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

  std::shared_ptr<OptimizedJITGraphAndMetaData> GetOptimizedJITGraphAndMetaData(
      size_t key);
  void Add(size_t key, std::shared_ptr<OptimizedJITGraphAndMetaData> val);
  void RemoveGraph(size_t key);
  bool IsCached(size_t key);
  bool Empty();
  void Clear();

 private:
  explicit LazyGraphCache();

  std::mutex m_mutex;
  ;
  // Cache stores a JIT graph shared_ptr and meta data for a given hash key
  std::unordered_map<size_t, std::shared_ptr<OptimizedJITGraphAndMetaData>>
      m_cache_map;
};

class OptimizedLazyGraphCache {
 public:
  static OptimizedLazyGraphCache& GetOptimizedLazyCache() {
    static OptimizedLazyGraphCache* optimized_mp_instance;
    if (!optimized_mp_instance) {
      optimized_mp_instance = new OptimizedLazyGraphCache();
    }
    return *optimized_mp_instance;
  }

  OptimizedLazyGraphCache(const OptimizedLazyGraphCache&) = delete;
  OptimizedLazyGraphCache(OptimizedLazyGraphCache&&) = delete;
  OptimizedLazyGraphCache& operator=(const OptimizedLazyGraphCache&) = delete;
  OptimizedLazyGraphCache& operator=(OptimizedLazyGraphCache&&) = delete;

  ~OptimizedLazyGraphCache();

  std::shared_ptr<OptimizedJITGraphAndMetaData> GetOptimizedJITGraphAndMetaData(
      size_t key);
  void Add(size_t key, std::shared_ptr<OptimizedJITGraphAndMetaData> val);
  void RemoveGraph(size_t key);
  bool IsCached(size_t key);
  bool Empty();
  void Clear();

 private:
  explicit OptimizedLazyGraphCache();

  std::mutex m_mutex;

  // Cache stores a JIT graph shared_ptr and meta data for a given hash key
  std::unordered_map<size_t, std::shared_ptr<OptimizedJITGraphAndMetaData>>
      m_cache_map;
};

} // namespace habana_lazy
