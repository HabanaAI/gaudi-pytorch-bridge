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

#include <torch/csrc/jit/runtime/argument_spec.h>
#include <atomic>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include "backend/habana_operator.h"
#include "backend/helpers/collective_kernel_info.h"
#include "backend/helpers/symbolic_expression.h"
#include "backend/helpers/tensor_info.h"
#include "backend/kernel/hpu_shape_inference.h"
#include "backend/synapse_helpers/env_flags.h"
#include "backend/synapse_helpers/graph.h"

#include "backend/kernel/hpu_recipe_cache.h"
#include "habana_helpers/logging.h"
#include "synapse_common_types.h"

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
      at::ArrayRef<torch::jit::IValue> input_refs,
      const size_t& graphKey,
      const size_t& graph_sym_hash,
      const size_t& graph_perm_hash,
      const std::string& op_strs);

  RecipeArgumentSpec(
      at::ArrayRef<torch::jit::IValue> input_refs,
      const size_t& graphKey,
      const size_t& graph_sym_hash,
      const size_t& graph_perm_hash,
      const std::string& op_strs,
      const uint64_t token);

  RecipeArgumentSpec(
      bool with_grad,
      at::ArrayRef<torch::jit::IValue> input_refs,
      const std::shared_ptr<torch::jit::Graph>& irgraph,
      const size_t& graphKey,
      const std::string& op_strs,
      size_t symhash,
      size_t permhash);

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

  size_t h2dHashCode() const {
    return h2d_hash_code;
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
    serialize(os, h2d_hash_code);
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
    deserialize(is, h2d_hash_code);
    deserialize(is, cargspec_hash_code);
    deserialize(is, dynamic_hash_code);
    deserialize(is, graph_with_permute_hash_code);
    deserialize(is, token_);
  }

 private:
  void ComputeOffsetHashCode(at::ArrayRef<torch::jit::IValue> input_refs);
  void ComputeH2DHashCode(at::ArrayRef<torch::jit::IValue> input_refs);

  HbCas cas;
  std::string opstrs;
  size_t hash_code{0};
  size_t graph_hash_code{0};
  size_t offset_hash_code{0};
  size_t h2d_hash_code{0};
  size_t cargspec_hash_code{0};
  size_t dynamic_hash_code{0};
  size_t graph_with_permute_hash_code{0};
  uint64_t token_{0};
};

// Hash functor for RecipeArgumentSpec
// Memory management is outside the scope of caching
// Input and output buffers need to be passed to the recipe
// The order of the inputs are according to the input stack
// The order of the outputs will match the order they appears within the
// subgraph
struct RecipeValueSpec {
  RecipeValueSpec(std::shared_ptr<torch::jit::Graph> g = nullptr)
      : collective_kernels_info(
            std::make_shared<habana_helpers::CollectiveKernelInfos>()),
        jit_graph_(g) {
    count++;
    id = count;
  }

  RecipeValueSpec(std::istream& is);

  ~RecipeValueSpec();

  friend std::ostream& operator<<(std::ostream& O, const RecipeValueSpec& v);

  std::string header_str();
  std::string build_header_str() const;
  std::string digest_str();
  int update_hit_count();
  synTensor get_syn_new_handle(
      std::unordered_map<uint64_t, synTensor>&
          synapse_tensor_id_to_tensor_handle,
      std::unordered_map<synTensor, synTensor>& synapse_orig_to_new_handle,
      size_t ridx);
  static void update_tensor_shape(
      synTensor tensor_handle,
      PtTensorInfoShared tinfo,
      std::vector<int64_t> shape);
  inline void update_new_tensor(
      size_t ridx,
      std::unordered_map<synTensor, synTensor>& synapse_orig_to_new_handle,
      std::vector<int64_t> new_shape,
      std::optional<uint64_t> tensor_offset_opt = std::nullopt,
      std::optional<PtTensorInfoShared> tinfo_opt = std::nullopt) const;
  void update_patching_table(
      at::ArrayRef<torch::jit::IValue>& input_refs,
      std::shared_ptr<VecOfIValPtrSh>& intermediate_tensors_ptr,
      VecOfIValPtrSh& dma_inputs,
      VecOfIValPtrSh& aten_outputs,
      const habana::IdShapeMap& m_actual_shapes,
      std::optional<
          std::reference_wrapper<const std::unordered_map<int64_t, at::Tensor>>>
          tidx_to_tensor_map_opt = std::nullopt,
      const std::optional<std::vector<at::Tensor>>& allocated_outputs =
          std::nullopt,
      std::vector<std::vector<int64_t>> output_shapes = {},
      std::unordered_map<synTensor, synTensor> synapse_orig_to_new_handle = {},
      bool is_shape_agnostic_graph = false);
  void update_node_params(
      const std::unordered_map<synNodeId, synNodeId>&
          synapse_node_orig_to_new_handle,
      const std::vector<synNodeId>& syn_node_id_vec,
      const synGraphHandle duplicate_graph_handle,
      std::shared_ptr<std::vector<InferNodeParams>>& node_params_vec_ptr);
  void populate_syn_tensor_ids(
      const synapse_helpers::graph::recipe_handle& recipe);
  void patch_launch_info(
      std::vector<synLaunchTensorInfo>& syn_launch_info_vec,
      std::vector<size_t>& external_tensor_info_indexes) const;

  void create_outdup(
      size_t ti_idx,
      std::unordered_map<size_t, IValPtrShared>& parent_ivpsh_map,
      std::string map_name,
      VecOfIValPtrSh& aten_outputs,
      bool is_shape_agnostic_graph = false) const;

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

  std::string get_graph_name() const {
    return graph_name;
  }
  void set_graph_name(const std::string& name) {
    graph_name = name;
  }

  bool get_optim_output_sif_value() const {
    return enable_optim_output_sif_;
  }

  size_t get_aten_output_num() const {
    return (
        num_outputs + num_input_to_outduplicates +
        num_intermediate_to_outduplicates + num_output_to_outduplicates);
  }


  void Serialize(std::ostream& os) const;

  size_t Size() const {
    size_t size = sizeof(*this);
    size += tensor_ids_.size() * sizeof(decltype(tensor_ids_)::value_type);
    for (const auto& tensor_info : dtensorinfos) {
      size += tensor_info->Size();
    }
    return size;
  }

  std::vector<PtTensorInfoShared> dtensorinfos;
  std::shared_ptr<habana_helpers::CollectiveKernelInfos>
      collective_kernels_info;
  std::unordered_map<int64_t, PtTensorInfoShared> sif_tidx_to_tinfo_map;
  std::unordered_map<uint64_t, uint64_t> st_to_tensor_idx_map;
  std::unordered_set<uint32_t> dynamic_nodes_with_backend_STs;
  std::unordered_map<size_t, habana_helpers::DynamicSIFInfo> ds_sifinfo_map;
  std::unordered_set<std::string> disabled_jit_ir_ops_;

  size_t id{0};
  size_t iter_idx{0};

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

  size_t key{0};
  size_t graph_key{0};
  std::string opstrs;

  std::string header;
  std::string graph_name;
  std::vector<uint64_t> tensor_ids_;
  bool dynamic_graph{false};
  bool enable_optim_output_sif_{false};
  bool enable_time_scope{false};
  // is_refine becomes true if the recipe is created from the refinement thread
  bool is_refined{false};
  // is_refine_wirt becomes true if the recipe is created from the refinement
  // thread and it the runtime improvement condition for refinement is
  // satisfied. The base time is not available for the first refinement for a
  // graph, so the runtime improvement condition is not applicable for the first
  // refinement.
  bool is_refined_wirt{false};
  std::shared_ptr<torch::jit::Graph> jit_graph_{nullptr};
  std::unique_ptr<synapse_helpers::graph> shape_agnostic_synapse_graph_{
      nullptr};
  size_t curr_symval_hash_{0};

  static size_t current_id_;
  static size_t count;
  static size_t recipe_count;
  static size_t dynamic_recipe_count;
  static size_t total_recipe_ntbytes;
  static size_t compile_count;

  size_t CalculateNtensorbytes() const {
    size_t ntensorbytes = 0;
    for (auto& ti : dtensorinfos) {
      if (!ti->is_duplicate()) {
        ntensorbytes += ti->get_size();
      }
    }
    return ntensorbytes;
  }
};

struct RecipeLauncher {
  RecipeLauncher(
      const RecipeValueSpec& rvs,
      std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe = nullptr);
  RecipeLauncher(std::istream& is, const RecipeValueSpec& rvs);
  void Launch(
      synapse_helpers::hpuStream_t hpu_stream,
      const at::ArrayRef<torch::jit::IValue>& input_refs,
      std::shared_ptr<VecOfIValPtrSh>& intermediate_tensors_ptr,
      const VecOfIValPtrSh& aten_outputs,
      std::vector<synLaunchTensorInfo>& syn_launch_info,
      std::vector<size_t>& external_tensor_info_indexes,
      const VecOfIValPtrSh& dma_inputs = {});

  // TODO should not be a shared_ptr
  std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe_{nullptr};
  uint64_t workspace_size_{0};
  size_t ntensorbytes_{0};
  // Multiple recipes can be queued up, so each recipe would need
  // a dedicated time slot for itself
  std::shared_ptr<synapse_helpers::TimeSlot> time_slot_;

  void SetRecipe(
      std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe) {
    recipe_ = std::move(recipe);
    if (recipe_) {
      workspace_size_ = synapse_helpers::graph::query_workspace_size(*recipe_);
    }
  }

  size_t Size() const {
    return collective_kernels_info_->Size();
  }

  void Serialize(std::ostream& os) const;

  size_t id_ = 0;
  size_t num_inputs_ = 0;
  size_t num_outputs_ = 0;
  size_t num_input_to_outduplicates_ = 0;
  size_t num_intermediate_to_outduplicates_ = 0;
  size_t num_launches = 0;
  std::string graph_name_;
  std::shared_ptr<habana_helpers::CollectiveKernelInfos>
      collective_kernels_info_;

  friend std::ostream& operator<<(std::ostream& O, const RecipeLauncher& v);
};

struct RecipeHolder {
  RecipeHolder(
      std::shared_ptr<RecipeLauncher> rl,
      std::shared_ptr<RecipeValueSpec> rvs)
      : rl_(rl), rvs_(rvs){};
  RecipeHolder(std::istream& is);
  std::shared_ptr<RecipeLauncher> rl_;
  std::shared_ptr<RecipeValueSpec> rvs_;

  void Serialize(std::ostream& os) const;

  bool is_in_use() const {
    return rl_.use_count() > 1;
  }

  size_t Size() const {
    return rl_->Size() + rvs_->Size();
  }
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

  void refine_graph(size_t graph_key);
  size_t Size() const;
  size_t HistSize() const;
  static void DumpBucketMemoryStat();
  static void DumpHistoryMemoryStat();
  void clear();

  static void load_ds_checkpoint(std::ifstream& ds_checkpoint);
  static void save_ds_checkpoint(std::ofstream& ds_checkpoint);
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

CREATE_OSTREAM_FORMATTER(habana::RecipeArgumentSpec);
CREATE_OSTREAM_FORMATTER(habana::RecipeValueSpec);
CREATE_OSTREAM_FORMATTER(habana::RecipeLauncher);
