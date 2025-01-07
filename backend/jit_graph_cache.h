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
#pragma once
#include <synapse_api.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <mutex>
#include <string_view>
#include "backend/habana_operator.h"
#include "backend/helpers/dynamic_shape_infer.h"
#include "backend/helpers/habana_types.h"
#include "backend/kernel/hpu_habana_cache.h"
#include "backend/synapse_helpers/device.h"

namespace habana {
using namespace std::literals;

size_t ComputePermutationHashCode(at::ArrayRef<torch::jit::IValue> input_refs);
size_t ComputeSymSizeHashCode(at::ArrayRef<torch::jit::IValue> input_refs);
size_t ComputeNodeSymOutputHashCode(
    const std::shared_ptr<torch::jit::Graph>& jit_graph);

// Functionality to calculate the graph hash on the JIT graph
void ComputeGraphHashCode(
    const std::shared_ptr<torch::jit::Graph>& irgraph,
    const std::string& id,
    at::ArrayRef<torch::jit::IValue> input_refs,
    std::string& op_strs,
    size_t& graphHashCode,
    uint64_t unique_graph_cntr = 0,
    std::vector<bool> node_bcast_details = {},
    bool dynamic_graph = false,
    const std::map<int64_t, std::vector<int64_t>> m_input_new_base_sizes = {});

size_t GetDataChecksum(void* data, size_t dataSize);

//  SynBuildCache class contains information which are calculated during build
//  synapse graph and can be reused for build that graph again. We store in
//  order to speed up build process

class SynBuildCache {
 public:
  template <auto SynBuildCache::*member, typename Func>
  auto& get_or_compute_ref(Func&& comp_func, size_t index) {
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
  auto get_or_compute(Func&& comp_func, size_t index) {
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
    clear_cached_outputs_tensors();
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

// Param types for patching node params at the lowering
enum class NodeParamType {
  METADATA = 0,
  VIEW_SIZES = 1,
  VIEW_STRIDES = 2,
  VIEW_OFFSET = 3,
};

using CValPtr = const torch::jit::Value*;
using CValPtrMap =
    std::unordered_map<CValPtr, std::tuple<NodeParamType, size_t, size_t>>;
using CValPtrtoIValueMap = std::unordered_map<CValPtr, torch::jit::IValue>;

inline bool is_eager_caching_supported() {
  return ((habana::HPUDeviceContext::get_device().type() == synDeviceGaudi) &&
          GET_ENV_FLAG_NEW(PT_HPU_PGM_ENABLE_CACHE)) ||
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EAGER_CACHE);
}

// JIT IR Ops supporting node params patching at the lowering
// ToDo: Remove this list for supporting generic node params patching
//       once all ops/kernels SIF method populates node params.
class NodeParamAgnosticOpList {
 private:
  static const std::unordered_set<c10::Symbol>& param_agnostic_ops() {
    static const std::unordered_set<c10::Symbol> ops_list{
        c10::Symbol::fromQualString("aten::fill_"),
        c10::Symbol::fromQualString("aten::as_strided"),
        c10::Symbol::fromQualString("hpu::strided_view"),
        c10::Symbol::fromQualString("hpu::strided_insert"),
        c10::Symbol::fromQualString("hpu::strided_insert_"),
        c10::Symbol::fromQualString("aten::sort"),
        c10::Symbol::fromQualString("aten::cat"),
        c10::Symbol::fromQualString("aten::topk"),
        c10::Symbol::fromQualString("aten::arange"),
        c10::Symbol::fromQualString("aten::upsample_nearest2d"),
        c10::Symbol::fromQualString("aten::upsample_bicubic2d"),
        c10::Symbol::fromQualString("aten::upsample_bicubic2d_backward"),
        c10::Symbol::fromQualString("aten::upsample_bilinear2d"),
        c10::Symbol::fromQualString("aten::upsample_bilinear2d_backward"),
        // TODO: [SW-198691] investigate and try to reenable, or remove from the list
        //c10::Symbol::fromQualString("aten::upsample_linear1d"),
        //c10::Symbol::fromQualString("aten::upsample_linear1d_backward"),
        c10::Symbol::fromQualString("aten::upsample_nearest1d"),
        c10::Symbol::fromQualString("aten::upsample_nearest1d_backward"),
        c10::Symbol::fromQualString("aten::upsample_nearest2d_backward"),
        c10::Symbol::fromQualString("hpu::upsample_nearest2d_backward"),
        c10::Symbol::fromQualString("aten::upsample_nearest3d"),
        c10::Symbol::fromQualString("aten::upsample_nearest3d_backward"),
        c10::Symbol::fromQualString("aten::resize_"),
        c10::Symbol::fromQualString("aten::masked_fill_"),
        c10::Symbol::fromQualString("aten::_efficientzerotensor")};
    return ops_list;
  }

  static const std::unordered_set<c10::Symbol>& scalar_not_patchable_ops() {
    static const std::unordered_set<c10::Symbol> scalar_not_patchable_ops_list{
        c10::Symbol::fromQualString("aten::_efficientzerotensor")};
    return scalar_not_patchable_ops_list;
  }

 public:
  static bool isNodeParamAgnosticOp(const c10::Symbol op) {
    // node params agnostic not supported when recipe cache is enabled
    if (is_eager_caching_supported()) {
      return false;
    }
    return (param_agnostic_ops().find(op) != param_agnostic_ops().end());
  }

  static bool IsScalarNotPatchableOp(const c10::Symbol op) {
    return (
        scalar_not_patchable_ops().find(op) !=
        scalar_not_patchable_ops().end());
  }
};

struct OptimizedJITGraphAndMetaData {
  OptimizedJITGraphAndMetaData();

  OptimizedJITGraphAndMetaData(
      const std::shared_ptr<torch::jit::Graph> JitGraphToLowering,
      const at::ArrayRef<torch::jit::IValue>& input_refs,
      uint64_t ug_cntr = 0,
      std::vector<bool> node_bcast_details = {},
      const std::string& id = "",
      const bool dynamic = false,
      const std::map<int64_t, std::vector<int64_t>> m_input_new_base_sizes =
          {});

  void ComputeGraphHashCode(
      const std::shared_ptr<torch::jit::Graph> JitGraphToLowering,
      const at::ArrayRef<torch::jit::IValue>& input_refs,
      const std::string& id = "",
      const std::map<int64_t, std::vector<int64_t>> m_input_new_base_sizes =
          {});

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

  void SetOptimizedLazyEagerFlag(bool flag);

  void SetUserMarkDynamic(bool flag);

  void SetUserRangesDynamic(
      std::vector<habana_helpers::RangeInfo>& range_infos);

  std::vector<habana_helpers::RangeInfo> GetUserRangesDynamic();

  bool IsUserMarkDynamic();

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

  size_t get_sym_expr_hash() {
    return sym_expr_hash_;
  }

  void set_sym_expr_hash(size_t hash) {
    sym_expr_hash_ = hash;
  }

  size_t get_graph_symint_hash() {
    return graph_symint_hash_;
  }

  void set_graph_symint_hash(size_t hash) {
    graph_symint_hash_ = hash;
  }

  size_t get_graph_key_with_perm() {
    return graph_key_with_perm_;
  }

  void set_graph_key_with_perm(size_t key) {
    graph_key_with_perm_ = key;
  }

  size_t get_graph_perm_hash() {
    return graph_perm_hash_;
  }

  void set_graph_perm_hash(size_t hash) {
    graph_perm_hash_ = hash;
  }

  void set_enable_optim_output_sif(bool enable_optim_output_sif) {
    enable_optim_output_sif_ = enable_optim_output_sif;
  }

  bool get_enable_optim_output_sif() {
    return enable_optim_output_sif_;
  }

  void set_maybe_static_recipe(bool maybe_static_recipe) {
    maybe_static_recipe_ = maybe_static_recipe;
  }

  bool get_maybe_static_recipe() {
    return maybe_static_recipe_;
  }

  void set_curr_symval_hash(size_t symval_hash) {
    curr_symval_hash_ = symval_hash;
  }

  size_t get_curr_symval_hash() {
    return curr_symval_hash_;
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

  bool get_is_param_agnostic_supported() const {
    return is_param_agnostic_supported_;
  }

  void set_is_param_agnostic_supported(const bool flag) {
    is_param_agnostic_supported_ = flag;
  }

  const CValPtrMap& get_param_jit_val_map() {
    return param_jit_val_map_;
  }

  void set_param_jit_val_map(CValPtrMap& val_map) {
    param_jit_val_map_ = val_map;
  }

  const CValPtrtoIValueMap& get_param_jit_val_to_ivalue_map() {
    return param_jit_val_to_ivalue_map_;
  }

  void set_param_jit_val_to_ivalue_map(CValPtrtoIValueMap& val_to_ivalue_map) {
    param_jit_val_to_ivalue_map_ = val_to_ivalue_map;
  }

  void set_new_strided_insert_output_shape(
      const std::vector<int64_t>& new_strided_insert_output_shape) {
    new_strided_insert_output_shape_ = new_strided_insert_output_shape;
  }

  const std::vector<int64_t>& get_new_strided_insert_output_shape() const {
    return new_strided_insert_output_shape_;
  }

  bool is_skip_tensor_permutation() const {
    return skip_tensor_permutation_;
  }

  void set_skip_tensor_permutation() {
    skip_tensor_permutation_ = true;
  }

  void increment_jit_cache_hit_count() {
    jit_cache_hit_count_++;
  }

  size_t get_jit_cache_hit_count() const {
    return jit_cache_hit_count_;
  }

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
  size_t sym_expr_hash_ = 0;
  size_t graph_key_with_perm_ = 0;
  size_t graph_symint_hash_ = 0;
  size_t graph_perm_hash_ = 0;
  bool enable_optim_output_sif_ = false;
  bool maybe_static_recipe_ = true;
  size_t curr_symval_hash_ = 0;
  std::optional<PermutationInfo> permutation_info_{};
  bool is_param_agnostic_supported_ = false;
  CValPtrMap param_jit_val_map_{};
  CValPtrtoIValueMap param_jit_val_to_ivalue_map_{};
  std::vector<int64_t> new_strided_insert_output_shape_;
  bool user_mark_dynamic = false;
  std::vector<habana_helpers::RangeInfo> m_range_infos;
  bool skip_tensor_permutation_{false};
  size_t jit_cache_hit_count_ = 0;
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
  std::unordered_map<
      size_t,
      std::shared_ptr<habana::OptimizedJITGraphAndMetaData>>
  get_m_cache_map() const {
    return m_cache_map;
  }
  const std::unordered_set<std::string_view>&
  get_eager_compiler_unsupported_op_prefixes() {
    return eager_compiler_unsupported_op_prefixes;
  };

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
  const std::unordered_set<std::string_view>
      eager_compiler_unsupported_op_prefixes = {
          "hpu::optimizer"sv,
          "hpu::fused_norm_lazy"sv,
          "hpu::fused_clip_norm"sv,
          "hpu::custom_foreach_add_"sv,
          "hpu::sdpa"sv,
          "hpu::fp8_sdpa"sv};
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
