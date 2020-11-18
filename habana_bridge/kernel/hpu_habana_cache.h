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
#include <string>
#include <mutex>

#include <ATen/Tensor.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_info.h"
#include "synapse_helpers/graph.h"

#define PGM_LRU_MAX_NRECIPES 700
#define PGM_LRU_MIN_NRECIPES 3

enum class PGMCachingPolicy {
  simple,
  single,
  lru
};

std::ostream & operator<<(std::ostream & O, PGMCachingPolicy P);

// Adding the op strings to the key for recipe
// Later the drop the storage for the vector of strings
//   if possible pass the subgraph as argument
//   compute the hash directly from the subgraph within the constructor
struct RecipeArgumentSpec {
  RecipeArgumentSpec(
      bool with_grad,
      at::ArrayRef<torch::jit::IValue> input_refs,
      const std::shared_ptr<torch::jit::Graph>& irgraph,
      const std::string &id);

  bool operator==(const RecipeArgumentSpec& arg) const {
    bool ret = (cas == arg.cas && opstrs == arg.opstrs);
    return ret;
  }

  size_t hashCode() const {
    return hash_code;
  }

  friend std::ostream& operator<<(std::ostream& O, const RecipeArgumentSpec& v);

 private:
  torch::jit::CompleteArgumentSpec cas;
  size_t hash_code;
  std::string opstrs;
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
      : recipe(r),
        dtensorinfos(nullptr),
        aten_outputs(nullptr) {
    count++;
    id = count;
  }

  ~RecipeValueSpec();

  void SelfCheck() {
    TORCH_CHECK(recipe != nullptr)
    TORCH_CHECK(dtensorinfos != nullptr);
    TORCH_CHECK(dtensorinfos->size() == num_tensors);
    TORCH_CHECK(!aten_outputs->empty());
  }

  void launch(at::ArrayRef<torch::jit::IValue> input_refs);

  void print_hbuff(
      size_t buf_idx,
      std::ofstream& out,
      size_t iteration_count,
      int numel = -1);
  void d2h_dbuff(size_t buf_idx);

  bool get_use_flag() {
    return in_use.load(std::memory_order_relaxed);
  }

  void set_use_flag(bool flag) {
    in_use.store(flag, std::memory_order_relaxed);
  }

  friend std::ostream& operator<<(std::ostream& O, const RecipeValueSpec& v);

  std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe;
  std::shared_ptr<std::vector<PtTensorInfo>> dtensorinfos;
  std::shared_ptr<std::vector<IValPtrShared>> aten_outputs;
  std::vector<at::Tensor> aten_intermediates;

  uint64_t htensor_wbuff = 0;
  unsigned htensor_wbuff_size = 0;

  size_t id{0};
  size_t iter_idx{0};
  size_t num_tensors{0};

  size_t num_inputs{0};
  size_t num_duplicates{0};
  size_t num_interims{0};
  size_t num_outputs{0};

  size_t ntensorbytes{0};

  size_t key{0};

  static size_t count;
  static size_t recipe_count;
  static size_t total_recipe_ntbytes;

 private:
  std::atomic<bool> in_use {false};
};

struct RecipeCacheSimple {
  std::unordered_map<
      std::shared_ptr<RecipeArgumentSpec>,
      std::shared_ptr<RecipeValueSpec>,
      RecipeArgumentSpecHash,
      RecipeArgumentSpecEqual>
      map_;

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

  std::shared_ptr<RecipeValueSpec> get(std::shared_ptr<RecipeArgumentSpec>& key) {
    if (exists(key)) {
      return map_[key];
    }

    return {nullptr};
  }

  void add(std::shared_ptr<RecipeArgumentSpec>& key, std::shared_ptr<RecipeValueSpec>& val);

  friend std::ostream& operator<<(std::ostream& O, const RecipeCacheSimple& v);
};

struct RecipeCacheSingle {
  std::shared_ptr<RecipeArgumentSpec> last_rargpsh {nullptr};
  std::shared_ptr<RecipeValueSpec> last_rvalpsh {nullptr};
  bool is_valid {false};

  bool empty() {
    return (!is_valid);
  }

  bool exists(std::shared_ptr<RecipeArgumentSpec>& key) {
    bool ret_flag{false};
    if (!empty() && *last_rargpsh == *key) {
      ret_flag = true;
    }
    return ret_flag;
  }

  std::shared_ptr<RecipeValueSpec> get(std::shared_ptr<RecipeArgumentSpec>& key) {
    if (exists(key)) {
      TORCH_CHECK(is_valid, "recipe.get is called on an empty cache");
      return last_rvalpsh;
    }

    return {nullptr};
  }

  void add(std::shared_ptr<RecipeArgumentSpec> &rargpsh,
      std::shared_ptr<RecipeValueSpec> &rvalpsh);

  friend std::ostream& operator<<(std::ostream& O, const RecipeCacheSimple& v);
};

class RecipeCacheLRU {
 public:
  static RecipeCacheLRU& get_cache(){
    std::lock_guard<std::mutex> lg(mutex_);
    if ( !instance_ ) {
      instance_ = new RecipeCacheLRU();

      char* smaxsize = getenv("HABANA_PGM_LRU_MAX");
      if (smaxsize != nullptr) {
        max_size_ = std::max(PGM_LRU_MIN_NRECIPES, atoi(smaxsize));
      }
      PT_BRIDGE_DEBUG("Creating : cache with lru replacement policy, max size ", max_size_);
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

  void remove_oldest();
  bool drop_lru(size_t &recipe_count);
  void add(std::shared_ptr<RecipeArgumentSpec>& key, std::shared_ptr<RecipeValueSpec>& val);

  std::shared_ptr<RecipeValueSpec> get(std::shared_ptr<RecipeArgumentSpec>& key);

  //friend std::ostream& operator<<(std::ostream& O, const RecipeCacheLRU& v);

 private:
  RecipeCacheLRU() = default;
  ~RecipeCacheLRU() = default;
  RecipeCacheLRU(const RecipeCacheLRU&) = delete;
  RecipeCacheLRU& operator=(const RecipeCacheLRU&) = delete;
  bool drop_lru_impl(size_t &recipe_count, bool mem_exhausted = false);

  static std::mutex mutex_;
  static RecipeCacheLRU* instance_;
  static size_t max_size_;

  std::list<std::pair<std::shared_ptr<RecipeArgumentSpec>,
      std::shared_ptr<RecipeValueSpec>>> list_;

  std::unordered_map<
      std::shared_ptr<RecipeArgumentSpec>,
      std::list<
          std::pair<std::shared_ptr<RecipeArgumentSpec>,
              std::shared_ptr<RecipeValueSpec>>>::iterator,
      RecipeArgumentSpecHash,
      RecipeArgumentSpecEqual> map_;
};
