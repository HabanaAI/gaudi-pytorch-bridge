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
#include "habana_helpers/dynamic_bucket_info.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_info.h"
#include "habana_serialization/recipe_cache.h"
#include "synapse_helpers/env_flags.h"
#include "synapse_helpers/graph.h"
#include "synapse_helpers/time_slot.h"

#define PGM_LRU_MAX_EAGER_NRECIPES 9000
#define PGM_LRU_MAX_LAZY_NRECIPES 2500
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
      const std::shared_ptr<torch::jit::Graph>& irgraph,
      at::ArrayRef<torch::jit::IValue> input_refs,
      std::string id = std::string());

  RecipeArgumentSpec(
      at::ArrayRef<torch::jit::IValue> input_refs,
      const std::shared_ptr<torch::jit::Graph>& irgraph,
      const uint64_t token = 0,
      const std::string id = std::string());

  RecipeArgumentSpec(
      bool with_grad,
      at::ArrayRef<torch::jit::IValue> input_refs,
      const std::shared_ptr<torch::jit::Graph>& irgraph,
      const std::string& id);

  bool operator==(const RecipeArgumentSpec& arg) const {
    bool ret = (opstrs == arg.opstrs);

    if (hash_code == graph_hash_code) {
      return ret;
    }

    if (hash_code == dynamic_hash_code) {
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

  friend std::ostream& operator<<(std::ostream& O, const RecipeArgumentSpec& v);

 private:
  void ComputeGraphHashCode(
      const std::shared_ptr<torch::jit::Graph>& irgraph,
      const std::string& id,
      at::ArrayRef<torch::jit::IValue> input_refs);
  void ComputeOffsetHashCode(at::ArrayRef<torch::jit::IValue> input_refs);

  HbCas cas;
  std::string opstrs;
  size_t hash_code{0};
  size_t graph_hash_code{0};
  size_t offset_hash_code{0};
  size_t cargspec_hash_code{0};
  size_t dynamic_hash_code{0};
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
      std::shared_ptr<synapse_helpers::graph::recipe_handle> r = nullptr)
      : recipe(r), dtensorinfos(nullptr), aten_outputs(nullptr) {
    count++;
    id = count;
  }

  RecipeValueSpec(std::istream& is);

  ~RecipeValueSpec();

  friend std::ostream& operator<<(std::ostream& O, const RecipeValueSpec& v);

  void SelfCheck() {
    TORCH_CHECK(recipe != nullptr)
    TORCH_CHECK(dtensorinfos != nullptr);
    TORCH_CHECK(dtensorinfos->size() == num_tinfos);
    TORCH_CHECK(!aten_outputs->empty());
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
      std::shared_ptr<std::vector<IValPtrShared>>& dma_inputs,
      const habana::NameShapeMap& m_actual_shapes,
      bool enable_tensor_release = true);
  void populate_syn_tensor_ids();
  void patch_launch_info(
      std::vector<synLaunchTensorInfoExt>& syn_launch_info_vec);
  void launch(
      at::ArrayRef<torch::jit::IValue> input_refs,
      std::shared_ptr<std::vector<IValPtrShared>> dma_inputs = nullptr);

  void create_outdup(PtTensorInfo& ti, at::Tensor orig);
  void create_outdup(size_t ti_idx, IValPtrShared& ivpsh_parent);
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

  bool get_enable_time_scope() {
    return enable_time_scope;
  }

  void set_enable_time_scope(bool flag) {
    enable_time_scope = flag;
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

  void Serialize(std::ostream& os) const;

  std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe;
  std::shared_ptr<std::vector<PtTensorInfo>> dtensorinfos;
  std::shared_ptr<std::vector<IValPtrShared>> aten_outputs;
  // We keep two separate arrays for storing persistent intermediate.
  // aten_intermediates is used for storing intermediates which are usually
  // marked persistent by persistenceMarkingPass. aten_dma_inputs is
  // used for storing the seed tensors needed for dropout kernel within the
  // recipe.
  std::vector<at::Tensor> aten_dma_inputs;
  std::vector<at::Tensor> aten_intermediates;
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

  std::string header;
  size_t num_tensors{0};
  uint64_t* tensor_ids{nullptr};
  const char** tensor_names{nullptr};
  bool dynamic_graph{false};
  bool enable_time_scope{false};

  // Multiple recipes can be queued up, so each recipe would need
  // a dedicated time slot for itself
  std::shared_ptr<synapse_helpers::TimeSlot> time_slot_;

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

  bool exists(std::shared_ptr<RecipeArgumentSpec>& key) {
    bool ret_flag{false};
    if (!empty() && map_.end() != map_.find(key)) {
      ret_flag = true;
    }
    return ret_flag;
  }

  void add(
      std::shared_ptr<RecipeArgumentSpec>& key,
      std::shared_ptr<RecipeValueSpec>& val);
  std::shared_ptr<RecipeValueSpec> get(
      std::shared_ptr<RecipeArgumentSpec>& key);
  bool drop_lru(size_t& num_recipes);
  void remove_oldest();
  void ResetDiskCache();

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
    if (!instance_) {
      instance_ = new DynamicBucketInfoMap();
    }
    return *instance_;
  }

  bool empty() {
    return (map_.size() == 0);
  }

  void add(
      std::shared_ptr<RecipeArgumentSpec>& key,
      std::shared_ptr<habana_helpers::DynamicBucketInfo>& val);
  std::shared_ptr<habana_helpers::DynamicBucketInfo> get(
      std::shared_ptr<RecipeArgumentSpec>& key);

  // Add print function for DynamicBucket
  // friend std::ostream& operator<<(std::ostream& O, const
  // DynamicBucketInfoMap& v);

 private:
  DynamicBucketInfoMap() = default;
  ~DynamicBucketInfoMap() = default;
  DynamicBucketInfoMap(const DynamicBucketInfoMap&) = delete;
  DynamicBucketInfoMap& operator=(const DynamicBucketInfoMap&) = delete;

  static std::mutex mutex_;
  static DynamicBucketInfoMap* instance_;

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

} // namespace habana
