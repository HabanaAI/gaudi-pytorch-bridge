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

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdlib>

#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_set>

#include <ATen/Tensor.h>
#include <absl/hash/hash.h>
#include <absl/types/variant.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include "habana_bridge/kernel/hpu_habana_cache.h"
#include "habana_kernels/habana_operator.h"

using namespace habana;

using CValPtr = const torch::jit::Value*;
using tensor_or_ref = synapse_helpers::tensor_or_ref;
using SynTensorOrRefList = std::vector<tensor_or_ref>;
using SharedSynTensorOrRefListPtr = std::shared_ptr<SynTensorOrRefList>;

using IValPtrSharedToTesorInfoMap =
    std::unordered_map<IValPtrShared, PtTensorInfo>;

struct habanaTensorLayoutInfo {
  habana::LayoutFormat layout;
  habana::LayoutFormat layout_at_graph_entry;
};

habana::LayoutFormat getLayoutFromDims(const std::vector<int64_t> dims);

class HabanaLaunchOpPT {
 public:
  explicit HabanaLaunchOpPT(const torch::jit::Node* node, bool debug);
  explicit HabanaLaunchOpPT(
      std::shared_ptr<torch::jit::Graph> graph,
      bool debug);
  ~HabanaLaunchOpPT();
  void run(torch::jit::Stack& stack);

  static std::unordered_set<std::string> watchlist_;
  static size_t instance_count_;

 private:
  std::shared_ptr<torch::jit::Graph> subgraph_;
  std::string opname_;
  std::string id_str;
  size_t ref_count_ = 0;
  bool debug_;

  // We keep a vector of kernels so that the context memory
  //   for each kernel is retained till graph execution
  // This is done to enable reuse of PT and synapse tensors and their processing
  std::vector<HabanaOperatorPtr> habana_kernels;

  // A map between the abstract value containers in graph and actual Ivalues in
  // stack
  std::unordered_map<CValPtr, habanaTensorLayoutInfo> value_to_tensor_layout;

  // A map for value to persistent flag
  std::unordered_map<CValPtr, bool> value_to_persistent_flag;

  // map between PT and synapse tensors
  std::deque<synapse_helpers::tensor> meta_syn_tensors;

  std::vector<IValPtrShared> pt_stack_sh;
  std::unordered_map<CValPtr, IValPtrShared> value_to_ivalue;
  std::unordered_map<IValPtrShared, SharedSynTensorOrRefListPtr>
      pt_to_synapse_tensors;

  // TIVs for launcing the recipe
  // tiv : absl::variant<PtTensorInfo, std::vector<PtTensorInfo>> objects
  std::unordered_map<
      IValPtrShared,
      absl::variant<PtTensorInfo, std::vector<PtTensorInfo>>>
      input_tiv_map;
  std::vector<absl::variant<PtTensorInfo, std::vector<PtTensorInfo>>>
      input_tivs;
  std::vector<absl::variant<PtTensorInfo, std::vector<PtTensorInfo>>>
      duplicate_tivs;
  // tiv : absl::variant<TensorInfo, std::vector<TensorInfo>> objects
  std::unordered_map<void*, IValPtrShared> buff_to_input_ivpsh_map;
  std::unordered_map<void*, IValPtrShared> buff_to_interim_ivpsh_map;

  size_t dma_input_idx{0};
  size_t interim_idx{0};

  // For supporting operators that need inputs which are not present in the
  // stack. These inputs need to be DMA transferred during creation of the op
  // or patching a cached recipe.
  std::deque<PtTensorInfo> dma_input_tensorinfos;

  // Temp additions to enable BatchNorm..tensors created that are not in graph
  // We get this to enable correct patching
  // Right now our patching is tightly coupled to graph nodes
  // BN is exception case, we can review our patching design for this
  std::vector<PtTensorInfo> interim_tensorinfos;

  std::vector<PtTensorInfo> output_tensorinfos;
  std::vector<PtTensorInfo> duplicate_outtinfos;
  synapse_helpers::graph* syn_graph_ptr = nullptr;

  // The persistent intermediates are stored in the following two vectors.
  // aten_intermediates is used for storing intermediates which are usually
  // marked persistent by persistenceMarkingPass. aten_dma_intermediates is
  // used for storing the seed tensors needed for dropout kernel.
  std::vector<at::Tensor> aten_intermediates;
  std::vector<at::Tensor> aten_dma_intermediates;

  // caching :: begin

  size_t num_inputs{0};
  // The inputs holding data usually are of type tensor and tensorList.
  // The following member keeps track of total number of tensor and tensorList
  // inputs
  size_t num_tensor_inputs{0};

  bool use_persistent_tensors{false};
  at::ArrayRef<torch::jit::IValue> input_refs;
  torch::jit::Stack* pt_stack = nullptr;

  RecipeCacheSimple recipe_cache_simple;
  RecipeCacheSingle recipe_cache_single;

  // Making the cache eviction policy as lru as default

  PGMCachingPolicy caching_policy{PGMCachingPolicy::lru};
  IValPtrSharedToTesorInfoMap output_tensorinfo_map;
  IValPtrSharedToTesorInfoMap duplicate_input_to_outtinfo_map;
  IValPtrSharedToTesorInfoMap duplicate_interim_to_outtinfo_map;

  // caching :: end

  bool enable_caching_{true};
  bool enable_tensor_release_{false};
  bool watch_tensor_flag_{false};
  bool enable_tensor_dump_{false};
  int tensor_dump_numel_{0};

  std::string tdmp_dir_name_;
  std::string tdmp_file_name_pre_;
  std::string tdmp_file_name_;

  uint64_t htensor_wbuff{0};
  unsigned htensor_wbuff_size{0};

  size_t iteration_count_ = 0;

  std::unordered_map<torch::jit::Node*, std::vector<synNodeId>>
      jit_to_synapse_node_idx_map;
  std::vector<synNodeId> blocking_syn_nodes_vec;
  std::vector<synNodeId> blocked_syn_nodes_vec;

  // TODO add all the optimizers
  std::vector<std::string> custom_optimizer_nodestr_vec = {
      "hpu::habanaOptimizerFusedSGDMomentum",
      "hpu::habanaOptimizerFusedAdagrad",
      "hpu::habanaOptimizerAdamW"};

  habana::LayoutFormat getTensorChannelOrder(torch::jit::Value* val);
  void runMetaDataAdjustmentPasses(torch::jit::graph_node_list graph_nodes);
  void weightLayoutMarkingPass(torch::jit::graph_node_list graph_nodes);
  void persistenceMarkingPass(torch::jit::graph_node_list graph_nodes);
  void markLayoutForOriginNodes(torch::jit::Value* val);
  void preProcessInputs();
  void processInputs(
      torch::jit::Node* node,
      const HabanaOperatorPtr& habana_kernel);
  void postProcessOutputs();
  at::Tensor permuteTensor(
      torch::jit::Value* value_in,
      const at::Tensor& input,
      habana::LayoutFormat permute_order);
  torch::jit::Stack getStackForNode(torch::jit::Node* node);
  void compile();
  void clear();
  bool isInGraphInputs(torch::jit::Value* value);
  bool isInGraphOutputs(torch::jit::Value* value);
  bool isInGraphOutputs(torch::jit::Node* node, size_t index);
  std::vector<bool> nodeOutputPersistence(torch::jit::Node* node);
  void CompileAndExecuteHabanaFusedOpKernel();
  bool isBlockingNode(torch::jit::Node*, torch::jit::Node*);
  void addSynNodes(std::vector<synNodeId>&, torch::jit::Node*);
  void ProcessControlEdges();
  void PrepareBlockingNodeList(torch::jit::Node*);
  void ProcessCustomOptControlEdges(torch::jit::graph_node_list);
  void HandleMappedTensor(
      CValPtr value_in,
      const HabanaOperatorPtr& habana_op,
      SharedSynTensorOrRefListPtr& tensorList);
  void HandleUnmappedTensor(
      CValPtr value_in,
      const HabanaOperatorPtr& habana_op,
      SharedSynTensorOrRefListPtr& tensorList);
  void HandleMappedandUnmappedTensor(
      CValPtr value_in,
      const HabanaOperatorPtr& habana_op,
      SharedSynTensorOrRefListPtr& tensorList);
  void GetSynapseInputs(
      const HabanaOperatorPtr& habana_op,
      torch::jit::Node* node);
  void GetSynapseOutputs(
      const HabanaOperatorPtr& habana_op,
      torch::jit::Node* node);
  bool isChannelOrderSupported(
      torch::jit::Value* val,
      const habana::LayoutFormat& supported_channel_order);
  c10::ScalarType getNodeScalarType(torch::jit::Node* node);
  void handlePrimNodes(torch::jit::Node* node);
  void handleRestrideNode(torch::jit::Node* node);
  void handleMetaOps(torch::jit::Node* node);

  bool IsOutputToRestride(torch::jit::Value* val);
  torch::jit::Value* GetRestridedOutvalue(torch::jit::Value* val);

  torch::jit::Node* GetUnpackNodeFromTensorList(torch::jit::Value* val);

  void PrintRecipeInputs();
  void UpdateOutputs(RecipeValueSpec& rv);
  template <class T>
  void clearMember(T& m_container);

  std::shared_ptr<RecipeValueSpec> GetCachedRecipe(
      std::shared_ptr<RecipeArgumentSpec>& spec_key);
  void ReturnCachedRecipe(RecipeValueSpec& rv);

  void OrderInputs();
  void FlattenAndLinkInputTIVs(RecipeValueSpec& rv);
  void OrderOutputTinfos(RecipeValueSpec& rv);

  void DumpTensors_pre(RecipeValueSpec& rv);
  void DumpTensors(RecipeValueSpec& rv);
  void create_duplicate_syn_tensor(
      at::Tensor* tensor,
      torch::jit::Value* value_in,
      bool persistence = true);
  bool IsCustomOptimizer(std::string node_str);
};
