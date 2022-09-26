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

#include <atomic>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>

#include <ATen/Tensor.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include "habana_bridge/kernel/hpu_shape_inference.h"
#include "habana_helpers/collective_kernel_info.h"
#include "habana_helpers/dynamic_bucket_info.h"
#include "habana_helpers/habana_serialization/include/habana_serialization/recipe_cache.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_info.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "synapse_helpers/env_flags.h"
#include "synapse_helpers/graph.h"
#include "synapse_helpers/time_slot.h"

#define PGM_LRU_MAX_EAGER_NRECIPES 100000
#define PGM_LRU_MAX_LAZY_NRECIPES 30000
#define PGM_LRU_MIN_NRECIPES 3

namespace habana {

/*
 * Overrides Pytorch's Complete Argument Spec
 */
class HbCas {
 public:
  explicit HbCas(bool with_grad, at::ArrayRef<c10::IValue> inputs);

  size_t hashCode() const {
    return p_cas->hashCode();
  }

  bool operator==(const HbCas& spec) const {
    return *p_cas == *spec.Cas();
  }

  bool operator!=(const HbCas& spec) const {
    return !(*this == spec);
  }

  std::shared_ptr<torch::jit::CompleteArgumentSpec> Cas() const {
    return p_cas;
  }

 private:
  std::shared_ptr<torch::jit::CompleteArgumentSpec> p_cas;
};

// Adding the op strings to the key for recipe
// Later the drop the storage for the vector of strings
//   if possible pass the subgraph as argument
//   compute the hash directly from the subgraph within the constructor
struct RecipeArgumentSpec {
  RecipeArgumentSpec(
      at::ArrayRef<torch::jit::IValue> input_refs,
      const size_t& graphKey,
      const std::string& op_strs);

  RecipeArgumentSpec(
      at::ArrayRef<torch::jit::IValue> input_refs,
      const size_t& graphKey,
      const std::string& op_strs,
      const uint64_t token);

  RecipeArgumentSpec(
      bool with_grad,
      at::ArrayRef<torch::jit::IValue> input_refs,
      const std::shared_ptr<torch::jit::Graph>& irgraph,
      const size_t& graphKey,
      const std::string& op_strs);

  bool operator==(const RecipeArgumentSpec& arg) const {
    bool ret = (opstrs == arg.opstrs && token_ == arg.token_);

    if (hash_code == graph_hash_code) {
      return ret;
    }

    if (hash_code == dynamic_hash_code) {
      return ret;
    }

    if (hash_code == graph_with_permute_hash_code) {
      return ret;
    }

    ret &= (cas == arg.cas);
    return ret;
  }

  size_t hashCode() const {
    return hash_code;
  }

  size_t graphHashCode() const {
    return graph_hash_code;
  }

  size_t offsetHashCode() const {
    return offset_hash_code;
  }

  size_t cArgSpecHashCode() const {
    return cargspec_hash_code;
  }

  size_t dynamicHashCode() const {
    return dynamic_hash_code;
  }

  bool hasToken() const {
    return (token_ != 0);
  }

  size_t graphWithPermuteHashCode() const {
    return graph_with_permute_hash_code;
  }

  std::string get_op_strs() {
    return opstrs;
  }

  size_t Size() const {
    size_t size = sizeof(*this);
    size += opstrs.size() * sizeof(decltype(opstrs)::value_type);
    return size;
  }

  friend std::ostream& operator<<(std::ostream& O, const RecipeArgumentSpec& v);

  void Serialize(std::ostream& os) const {
    using namespace serialization;
    serialize(os, opstrs);
    serialize(os, hash_code);
    serialize(os, graph_hash_code);
    serialize(os, offset_hash_code);
    serialize(os, cargspec_hash_code);
    serialize(os, dynamic_hash_code);
    serialize(os, graph_with_permute_hash_code);
    serialize(os, token_);
  }

  RecipeArgumentSpec(std::istream& is)
      : cas(false, {at::IValue{torch::empty({0}, "hpu")}}) {
    using namespace serialization;
    deserialize(is, opstrs);
    deserialize(is, hash_code);
    deserialize(is, graph_hash_code);
    deserialize(is, offset_hash_code);
    deserialize(is, cargspec_hash_code);
    deserialize(is, dynamic_hash_code);
    deserialize(is, graph_with_permute_hash_code);
    deserialize(is, token_);
  }

 private:
  void ComputeOffsetHashCode(at::ArrayRef<torch::jit::IValue> input_refs);

  HbCas cas;
  std::string opstrs;
  size_t hash_code{0};
  size_t graph_hash_code{0};
  size_t offset_hash_code{0};
  size_t cargspec_hash_code{0};
  size_t dynamic_hash_code{0};
  size_t graph_with_permute_hash_code{0};
  uint64_t token_{0};
};

// Hash functor for RecipeArgumentSpec
struct RecipeArgumentSpecHash {
 public:
  size_t operator()(const std::shared_ptr<RecipeArgumentSpec>& v) const {
    return v->hashCode();
  }
};

// Comparator for RecipeArgumentSpec
struct RecipeArgumentSpecEqual {
 public:
  bool operator()(
      const std::shared_ptr<RecipeArgumentSpec>& v1,
      const std::shared_ptr<RecipeArgumentSpec>& v2) const {
    if (nullptr == v1 && nullptr == v2)
      return true;
    if (nullptr == v1)
      return false;
    if (nullptr == v2)
      return false;

    return (*v1) == (*v2);
  }
};

// Memory management is outside the scope of caching
// Input and output buffers need to be passed to the recipe
// The order of the inputs are according to the input stack
// The order of the outputs will match the order they appears within the
// subgraph
struct RecipeValueSpec {
  RecipeValueSpec(
      std::shared_ptr<synapse_helpers::graph::recipe_handle> r = nullptr,
      std::shared_ptr<torch::jit::Graph> g = nullptr)
      : recipe(r), dtensorinfos(nullptr), aten_outputs(nullptr), jit_graph_(g) {
    count++;
    id = count;
  }

  RecipeValueSpec(std::istream& is);

  ~RecipeValueSpec();

  friend std::ostream& operator<<(std::ostream& O, const RecipeValueSpec& v);

  void SelfCheck() {
    if (recipe != nullptr || !collective_kernels_info.empty()) {
      TORCH_CHECK(dtensorinfos != nullptr);
      TORCH_CHECK(dtensorinfos->size() == num_tinfos);
      TORCH_CHECK(!aten_outputs->empty());
    }
  }

  bool get_use_flag() {
    return in_use.load(std::memory_order_relaxed);
  }

  void set_use_flag(bool flag) {
    in_use.store(flag, std::memory_order_relaxed);
  }

  void print_hbuff(
      size_t buf_idx,
      std::ofstream& out,
      size_t iteration_count,
      int numel = -1);
  void d2h_dbuff(size_t buf_idx);

  std::string header_str();
  std::string build_header_str() const;
  std::string digest_str();
  int update_hit_count();
  void update_patching_table(
      at::ArrayRef<torch::jit::IValue>& input_refs,
      std::shared_ptr<std::vector<IValPtrShared>>& intermediate_tensors_ptr,
      std::shared_ptr<std::vector<IValPtrShared>>& dma_inputs_ptr,
      const habana::IdShapeMap& m_actual_shapes,
      std::optional<
          std::reference_wrapper<const std::unordered_map<int64_t, at::Tensor>>>
          tidx_to_tensor_map_opt = std::nullopt);
  void populate_syn_tensor_ids();
  void patch_launch_info(
      std::vector<synLaunchTensorInfoExt>& syn_launch_info_vec,
      std::vector<size_t>& external_tensor_info_indexes);
  void PrintDebugInfo(
      at::ArrayRef<torch::jit::IValue>& input_refs,
      std::shared_ptr<std::vector<IValPtrShared>>& intermediate_tensors_ptr);
  void launch(
      synapse_helpers::hpuStream_t hpu_stream,
      synEventHandle event_handle,
      synapse_helpers::hpuStream_t event_stream,
      bool event_flag,
      at::ArrayRef<torch::jit::IValue>& input_refs,
      std::shared_ptr<std::vector<IValPtrShared>>& intermediate_tensors_ptr,
      std::shared_ptr<std::vector<IValPtrShared>> dma_inputs_ptr = nullptr);

  void create_outdup(
      size_t ti_idx,
      std::unordered_map<size_t, IValPtrShared>& parent_ivpsh_map,
      std::string map_name);

  static size_t get_recipe_count() {
    return recipe_count;
  }
  static size_t get_dynamic_recipe_count() {
    return dynamic_recipe_count;
  }

  static void increment_compile_count() {
    compile_count++;
  }
  static size_t get_compile_count() {
    return compile_count;
  }
  static void increment_launch_count() {
    launch_count++;
  }
  static size_t get_launch_count() {
    return launch_count;
  }

  bool get_refined() {
    return is_refined;
  }
  void set_refined() {
    is_refined = true;
  }

  bool get_refined_wirt() {
    return is_refined_wirt;
  }
  void set_refined_wirt() {
    is_refined_wirt = true;
  }

  void increment_recipe_count() {
    recipe_count++;
    if (dynamic_graph) {
      RecipeValueSpec::dynamic_recipe_count++;
    }
  }

  void decrement_recipe_count() {
    recipe_count--;
    if (dynamic_graph) {
      RecipeValueSpec::dynamic_recipe_count--;
    }
  }

  size_t get_key() {
    return key;
  }
  void set_key(size_t k) {
    key = k;
  }

  size_t get_graph_key() {
    return graph_key;
  }
  void set_graph_key(size_t k) {
    graph_key = k;
  }

  std::string get_op_strs() {
    return opstrs;
  }
  void set_op_strs(std::string s) {
    opstrs = s;
  }

  std::string get_graph_name() {
    return graph_name;
  }
  void set_graph_name(const std::string& name) {
    graph_name = name;
  }

  void Serialize(std::ostream& os) const;

  size_t Size() const {
    size_t size = sizeof(*this);
    size += num_tensors * sizeof(tensor_ids);
    size += num_tensors * sizeof(tensor_names);
    for (const auto& kernel_info : collective_kernels_info) {
      size += kernel_info->Size();
    }
    for (const auto& tensor_info : *dtensorinfos) {
      size += tensor_info->Size();
    }
    return size;
  }

  std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe;
  std::shared_ptr<std::vector<PtTensorInfoShared>> dtensorinfos;
  std::shared_ptr<std::vector<IValPtrShared>> aten_outputs;
  std::vector<std::shared_ptr<habana_helpers::collective_kernel_info>>
      collective_kernels_info;
  std::unordered_map<int64_t, PtTensorInfoShared> sif_tidx_to_tinfo_map;
  uint64_t workspace_size;

  uint64_t htensor_wbuff = 0;
  unsigned htensor_wbuff_size = 0;

  size_t id{0};
  size_t iter_idx{0};
  size_t num_tinfos{0};

  size_t num_inputs{0};
  size_t num_induplicates{0};
  size_t num_dma_inputs{0};
  size_t num_shape_tensors{0};
  size_t num_intermediates{0};
  size_t num_outputs{0};
  size_t num_outduplicates{0};
  size_t num_input_to_outduplicates{0};
  size_t num_intermediate_to_outduplicates{0};
  size_t num_output_to_outduplicates{0};
  size_t num_launches{0};

  size_t ntensorbytes{0};

  size_t key{0};
  size_t graph_key{0};
  std::string opstrs;

  std::string header;
  std::string graph_name;
  size_t num_tensors{0};
  uint64_t* tensor_ids{nullptr};
  const char** tensor_names{nullptr};
  bool dynamic_graph{false};
  bool enable_time_scope{false};
  // is_refine becomes true if the recipe is created from the refinement thread
  bool is_refined{false};
  // is_refine_wirt becomes true if the recipe is created from the refinement
  // thread and it the runtime improvement condition for refinement is
  // satisfied. The base time is not available for the first refinement for a
  // graph, so the runtime improvement condition is not applicable for the first
  // refinement.
  bool is_refined_wirt{false};

  // Multiple recipes can be queued up, so each recipe would need
  // a dedicated time slot for itself
  std::shared_ptr<synapse_helpers::TimeSlot> time_slot_;
  std::shared_ptr<torch::jit::Graph> jit_graph_{nullptr};

  static size_t current_id_;
  static size_t count;
  static size_t recipe_count;
  static size_t dynamic_recipe_count;
  static size_t total_recipe_ntbytes;
  static size_t compile_count;
  static size_t launch_count;

 private:
  std::atomic<bool> in_use{false};
};

class DiskCache {
 public:
  DiskCache(std::string cache_path);
  void Add(const RecipeValueSpec& recipe, const RecipeArgumentSpec& spec);
  std::shared_ptr<RecipeValueSpec> Find(const RecipeArgumentSpec& spec);
  // in case RecipeValueSpec creation failed, DiskCache is leaving lock files on
  // disk. This ensures a cleanup.

 private:
  serialization::RecipeCache recipe_cache_;
  // library specific suffix to determine for what TF, Synapse, etc. the cache
  // entry was produced
  std::string cache_id_suffix_;
};

class RecipeCacheLRU {
 public:
  static RecipeCacheLRU& get_cache() {
    std::lock_guard<std::mutex> lg(mutex_);
    if (!instance_) {
      instance_ = new RecipeCacheLRU();
      // PT_HPU_LAZY_MODE = 0 is Pure Eager and 2 is Eager through Lazy
      if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 1)
        max_size_ = PGM_LRU_MAX_LAZY_NRECIPES;
      else
        max_size_ = PGM_LRU_MAX_EAGER_NRECIPES;
      char* smaxsize = getenv("HABANA_PGM_LRU_MAX");
      if (smaxsize != nullptr) {
        max_size_ = std::max(PGM_LRU_MIN_NRECIPES, atoi(smaxsize));
      }
    }
    return *instance_;
  }

  bool empty() {
    return (map_.size() == 0);
  }

  size_t get_length() {
    return list_.size();
  }

  void clear() {
    map_.clear();
    list_.clear();
  }

  bool exists(std::shared_ptr<RecipeArgumentSpec>& key) {
    bool ret_flag{false};
    if (!empty() && map_.end() != map_.find(key)) {
      ret_flag = true;
    }
    return ret_flag;
  }

  std::pair<
      std::shared_ptr<RecipeArgumentSpec>,
      std::shared_ptr<RecipeValueSpec>>
      dropped_recipe;
  void add(
      std::shared_ptr<RecipeArgumentSpec>& key,
      std::shared_ptr<RecipeValueSpec>& val);
  std::shared_ptr<RecipeValueSpec> get(
      std::shared_ptr<RecipeArgumentSpec>& key);
  bool drop_lru(size_t& num_recipes);
  void remove_oldest();
  void ResetDiskCache();
  void Serialize(std::string recipe_cache_path);
  void Deserialize(std::string recipe_cache_path);

  static void SetHostMemoryThreshold(
      uint32_t host_memory_threshold = default_host_memory_threshold);

  size_t Size() const;
  size_t SynapseRecipeSize() const;
  static void DumpRecipeMemoryStat();
  static void DumpSynapseRecipeMemoryStat();
  static void DumpDynamicShapeMemoryStat();

  // friend std::ostream& operator<<(std::ostream& O, const RecipeCacheLRU& v);

 private:
  RecipeCacheLRU();
  ~RecipeCacheLRU() = default;
  RecipeCacheLRU(const RecipeCacheLRU&) = delete;
  RecipeCacheLRU& operator=(const RecipeCacheLRU&) = delete;
  bool drop_lru_impl(size_t& recipe_count, bool mem_exhausted = false);
  void insert(
      std::shared_ptr<RecipeArgumentSpec>& key,
      std::shared_ptr<RecipeValueSpec>& val);
  void InitDiskCache();

  static std::mutex mutex_;
  static RecipeCacheLRU* instance_;
  static size_t max_size_;
  std::unique_ptr<DiskCache> disk_cache_;
  static const uint32_t default_host_memory_threshold = 90;

  std::list<std::pair<
      std::shared_ptr<RecipeArgumentSpec>,
      std::shared_ptr<RecipeValueSpec>>>
      list_;

  std::unordered_map<
      std::shared_ptr<RecipeArgumentSpec>,
      std::list<std::pair<
          std::shared_ptr<RecipeArgumentSpec>,
          std::shared_ptr<RecipeValueSpec>>>::iterator,
      RecipeArgumentSpecHash,
      RecipeArgumentSpecEqual>
      map_;
};

class DynamicBucketInfoMap {
 public:
  static DynamicBucketInfoMap& get_instance() {
    std::lock_guard<std::mutex> lg(mutex_);
    static DynamicBucketInfoMap instance_;
    return instance_;
  }

  bool empty() {
    return (map_.size() == 0);
  }

  void add(
      std::shared_ptr<RecipeArgumentSpec>& key,
      std::shared_ptr<habana_helpers::DynamicBucketInfo>& val);
  std::shared_ptr<habana_helpers::DynamicBucketInfo> get(
      std::shared_ptr<RecipeArgumentSpec>& key);

  void refine_graph(size_t graph_key, size_t step);
  size_t Size() const;
  size_t HistSize() const;
  static void DumpBucketMemoryStat();
  static void DumpHistoryMemoryStat();
  void clear();

  static void load_ds_checkpoint(std::string checkpoint_path);
  static void save_ds_checkpoint(std::string checkpoint_path);
  void Serialize(std::ostream& os) const;
  void Deserialize(std::istream& is);

 private:
  DynamicBucketInfoMap() = default;
  ~DynamicBucketInfoMap() = default;
  DynamicBucketInfoMap(const DynamicBucketInfoMap&) = delete;
  DynamicBucketInfoMap& operator=(const DynamicBucketInfoMap&) = delete;

  static std::mutex mutex_;

  std::unordered_map<
      std::shared_ptr<RecipeArgumentSpec>,
      std::shared_ptr<habana_helpers::DynamicBucketInfo>,
      RecipeArgumentSpecHash,
      RecipeArgumentSpecEqual>
      map_;

  bool exists(std::shared_ptr<RecipeArgumentSpec>& key) {
    bool ret_flag{false};
    if (!empty() && map_.end() != map_.find(key)) {
      ret_flag = true;
    }

    return ret_flag;
  }
};

void ClearDynamicBucketRecipeInfo();
} // namespace habana
