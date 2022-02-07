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
#include "habana_bridge/kernel/hpu_habana_meta_op_list.h"
#include "habana_bridge/kernel/hpu_shape_inference.h"

#include "habana_kernels/habana_operator.h"
#include "habana_lazy/visualize.h"
#include "pytorch_helpers/habana_helpers/compilation_statistics.h"

namespace habana {
using CValPtr = const torch::jit::Value*;
using tensor_or_ref = synapse_helpers::tensor_or_ref;
using SynTensorOrRefList = std::vector<tensor_or_ref>;
using SharedSynTensorOrRefListPtr = std::shared_ptr<SynTensorOrRefList>;
using IValPtrSharedToTesorInfoMap =
    std::unordered_map<IValPtrShared, PtTensorInfoShared>;

struct habanaTensorLayoutInfo {
  LayoutFormat layout;
  LayoutFormat layout_at_graph_entry;
};

enum ControlEdgeType {
  kCONTROL_EDGE_NONE = 0,
  kCONTROL_EDGE_,
  kCONTROL_EDGE_OTHER_,
  kCONTROL_EDGE_INPLACE
};

LayoutFormat getLayoutFromDims(const std::vector<int64_t>& dims);

struct DynamicShapeInfo {
  habana_helpers::DynamicBucketInfo::InpTensorShapes act_input_tshapes;
  habana_helpers::DynamicBucketInfo::InpTensorShapes min_input_tshapes;
  habana_helpers::DynamicBucketInfo::InpTensorShapes max_input_tshapes;
  habana_helpers::DynamicDimsPolicy min_policy;
  habana_helpers::DynamicDimsPolicy max_policy;
  uint64_t current_bucket_id;
};

class PassException : public std::exception {
 public:
  explicit PassException(
      habana::ShapeInfo::InferencePass pass,
      std::string message)
      : m_pass(pass), m_message(message) {}

  virtual ~PassException() = default;

  const char* what() const noexcept override {
    return m_message.c_str();
  }
  habana::ShapeInfo::InferencePass Pass() const {
    return m_pass;
  }

 private:
  habana::ShapeInfo::InferencePass m_pass =
      habana::ShapeInfo::InferencePass::INVALID;
  std::string m_message;
};

struct TensorMetaData {
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  c10::MemoryFormat mf;

  TensorMetaData(
      std::vector<int64_t> sz,
      std::vector<int64_t> st,
      c10::MemoryFormat f)
      : sizes(sz), strides(st), mf(f) {}
};

struct HabanaMetaDataToLowering {
  HabanaMetaDataToLowering(
      const bool& debug,
      const size_t& graphIndex,
      const std::string OpName,
      const std::string& op_strs,
      const size_t graph_key,
      bool is_optimized_lazy_eager = false);

  std::string GetOpStrs() {
    return opstrs;
  }

  size_t GetGraphKey() {
    return graphKey;
  }

  std::string& GetOpName() {
    return op_name;
  }

  size_t GetGraphIndex() {
    return graph_index;
  }

  bool GetDbgFlag() {
    return dbg;
  }

  bool GetOptimizedLazyEagerFlag() {
    return isOptimizedLazyEager;
  }

 private:
  bool dbg = false;
  size_t graph_index = 0;
  std::string op_name = std::string();
  std::string opstrs = std::string();
  size_t graphKey = 0;
  bool isOptimizedLazyEager = false;
};

class HabanaLaunchOpPT {
 public:
  explicit HabanaLaunchOpPT(
      std::shared_ptr<torch::jit::Graph> graph,
      std::shared_ptr<HabanaMetaDataToLowering> hb_meta_data_to_lowering);
  ~HabanaLaunchOpPT();

  void CompileGraphWithRange(
      torch::jit::Stack& stack,
      habana_helpers::DynamicBucketInfo::ResultShapes& input_ranges,
      habana_helpers::Bucket& new_bucket);

  void run(torch::jit::Stack& stack);

  static std::unordered_set<std::string> watchlist_;

 private:
  std::string op_name = std::string();
  std::string name = std::string();
  size_t graph_index = 0;
  std::shared_ptr<torch::jit::Graph> jit_ir_graph;
  bool debug;
  std::string id_str = std::string();
  std::string op_strs = std::string();
  size_t graph_key = 0;
  synapse_helpers::graph* syn_graph_ptr = nullptr;

  std::vector<TensorMetaData> input_tms;

  std::string DumpNode(torch::jit::Node* node);
  // We keep a vector of kernels so that the context memory
  //   for each kernel is retained till graph execution
  // This is done to enable reuse of PT and synapse tensors and their processing
  std::vector<HabanaOperatorPtr> habana_kernels;

  // map between PT and synapse tensors
  std::deque<synapse_helpers::tensor> meta_syn_tensors;

  std::vector<IValPtrShared> pt_stack_sh;
  std::unordered_map<CValPtr, IValPtrShared> value_to_ivalue;
  std::unordered_map<IValPtrShared, SharedSynTensorOrRefListPtr>
      pt_to_synapse_tensors;

  std::unordered_map<IValPtrShared, PtTensorInfoShared>
      ivalue_to_tensor_info_map;

  // A map for value to persistent flag
  std::unordered_map<CValPtr, bool> valptr_to_persistent_map;

  // TIV : absl::variant<PtTensorInfoShared, std::vector<PtTensorInfoShared>>
  // objects TIVs for launcing the recipe

  // input_tivs and output_tensorinfos are used with caching disabled
  std::vector<
      absl::variant<PtTensorInfoShared, std::vector<PtTensorInfoShared>>>
      input_tivs;
  std::vector<PtTensorInfoShared> output_tensorinfos;

  // Following tiv stores are used with caching enabled
  std::unordered_map<
      IValPtrShared,
      absl::variant<PtTensorInfoShared, std::vector<PtTensorInfoShared>>>
      input_tiv_map;
  std::vector<
      absl::variant<PtTensorInfoShared, std::vector<PtTensorInfoShared>>>
      duplicate_input_tivs;
  std::unordered_map<void*, IValPtrShared> buff_to_input_ivpsh_map;
  std::unordered_map<void*, IValPtrShared> buff_to_intermediate_ivpsh_map;
  std::unordered_map<void*, IValPtrShared> buff_to_output_ivpsh_map;
  std::unordered_map<void*, tensor_or_ref> buff_to_syn_tensor_map;
  std::vector<PtTensorInfoShared> duplicate_outtinfos;

  size_t dma_input_idx{0};
  size_t appended_index{0};
  size_t intermediate_index{0};
  size_t shape_index{0};

  // The persistent intermediates are stored in the following two vectors.
  // aten_intermediates is used for storing intermediates which are usually
  // marked persistent by persistenceMarkingPass. aten_dma_inputs is
  // used for storing the seed tensors needed for dropout kernel.
  std::vector<at::Tensor> aten_intermediates;
  // tinfos corresponding to aten_intermediates.
  std::vector<PtTensorInfoShared> intermediate_tinfos;

  // For supporting operators that need inputs which are not present in the
  // stack. These inputs need to be DMA transferred during creation of the op
  // or patching a cached recipe.
  std::vector<at::Tensor> aten_dma_inputs;
  // tinfos corresponding to aten_intermediates.
  std::deque<PtTensorInfoShared> dma_input_tensorinfos;
  // tinfos corresponding to shape tensor.
  std::vector<PtTensorInfoShared> shape_tensor_tinfos;

  // caching :: begin

  size_t num_inputs{0};
  // The inputs holding data usually are of type tensor and tensorList.
  // The following member keeps track of total number of tensor and tensorList
  // inputs
  size_t num_tensor_inputs{0};

  bool use_persistent_tensors{false};

  at::ArrayRef<torch::jit::IValue> input_refs;
  torch::jit::Stack* pt_stack{nullptr};
  uint64_t t_compile_ns{0};
  std::shared_ptr<RecipeValueSpec> cur_rvalpsh{nullptr};

  // Making the cache eviction policy as lru as default

  IValPtrSharedToTesorInfoMap output_tensorinfo_map;
  IValPtrSharedToTesorInfoMap duplicate_input_to_outtinfo_map;
  IValPtrSharedToTesorInfoMap duplicate_intermediate_to_outtinfo_map;
  IValPtrSharedToTesorInfoMap duplicate_output_to_outtinfo_map;
  std::shared_ptr<RecipeArgumentSpec> cur_rargpsh{nullptr};

  // caching :: end

  bool enable_caching_{true};
  bool watch_tensor_flag_{false};
  bool enable_tensor_dump_{false};
  bool refine_ds_enabled_{false};
  int tensor_dump_numel_{0};

  uint64_t cur_ds_token_{0};

  std::string tdmp_dir_name_;
  std::string tdmp_file_name_pre_;
  std::string tdmp_file_name_;

  uint64_t htensor_wbuff{0};
  unsigned htensor_wbuff_size{0};

  size_t iteration_count_ = 0;

  std::unordered_map<torch::jit::Node*, std::vector<synNodeId>>
      jit_to_synapse_node_idx_map;
  std::vector<torch::jit::Node*> blocking_nodes_vec;
  std::vector<synNodeId> blocking_syn_nodes_vec;
  std::vector<synNodeId> blocked_syn_nodes_vec;
  std::vector<std::pair<torch::jit::Value*, torch::jit::Node*>>
      memory_reuse_pairs;
  std::vector<std::shared_ptr<habana_helpers::collective_kernel_info>>
      collective_kernels_info;

  // TODO add all the optimizers
  std::vector<std::string> custom_optimizer_nodestr_vec = {
      "hpu::habanaOptimizerFusedSGDMomentum",
      "hpu::habanaOptimizerFusedAdagrad",
      "hpu::habanaOptimizerAdamW",
      "hpu::habanaOptimizerLambPhase1",
      "hpu::habanaOptimizerLambPhase2"};

  // TODO collect the control edge structures in a child class
  std::unordered_map<torch::jit::Node*, std::pair<size_t, size_t>>
      dfs_time_in_out_map;
  size_t dfs_cnt = 0;

  // Main function responsible for constructing a synapse graph from
  // 1. JIT IR Graph
  // 2. Input Stack
  // Currently this funciton is used for shape inference as well
  void BuildSynapseGraph(
      synapse_helpers::graph& syn_graph,
      bool is_shape_inference = false);

  LayoutFormat getTensorChannelOrder(torch::jit::Value* val);
  void runMetaDataAdjustmentPasses(torch::jit::graph_node_list graph_nodes);
  void weightLayoutMarkingPass(torch::jit::graph_node_list graph_nodes);
  void set_persistence_input(torch::jit::Node*);
  void set_persistence_output(torch::jit::Node*);
  void persistenceMarkingPass(torch::jit::graph_node_list graph_nodes);
  void markLayoutForOriginNodes(torch::jit::Value* val);
  void preProcessInputs();
  torch::jit::Stack getStackForNode(torch::jit::Node* node);
  int64_t isInGraphInputs(torch::jit::Value* value);
  bool isInGraphOutputs(torch::jit::Value* value);
  bool isInGraphOutputs(torch::jit::Node* node, size_t index);
  bool nodeOutputPersistencePerValue(
      torch::jit::Node* node,
      torch::jit::Value* value_out);
  std::vector<bool> nodeOutputPersistence(torch::jit::Node* node);
  bool isInplace(torch::jit::Node* node);
  bool isControlEdge(torch::jit::Node* node);
  void CreateValueToIvalueMapForInputs();
  void InitiateSynlaunchTimeCapture(RecipeValueSpec& rv);
  void ProcessHabanaFusedOpWithDS();
  bool IsValidNode(torch::jit::Node*);

  void addSynNodes(std::vector<synNodeId>&, torch::jit::Node*);
  void ProcessControlEdges();
  ControlEdgeType nodeRequiresControlEdge(torch::jit::Node* node);
  void PrepareBlockingNodeList(torch::jit::Node*, ControlEdgeType control_type);
  void ProcessCustomOptControlEdges(torch::jit::graph_node_list);
  void Dfs(torch::jit::Node*);
  void PreprocessControlEdges();
  bool IsControlEdgeCycle(torch::jit::Node*);
  bool IsAncestor(torch::jit::Node*, torch::jit::Node*);
  bool IsAncestorOrDescendant(torch::jit::Node*, torch::jit::Node*);
  void ProcessControlEdgesForMemoryReuse();
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
  const std::string& GetSynapseGraphName() {
    return SetAndGetSynapseGraphName(name, graph_index);
  }
  std::string& SetAndGetSynapseGraphName(
      const std::string& name,
      size_t g_index);
  void SetSynapseGraphName(const std::string& name, size_t g_index);
  void SetOpName(const std::string& name);
  void ProcessPersistentNodeOutput(
      const IValPtrShared& ivpsh,
      const ValPtr& vp,
      const synapse_helpers::tensor& out_syntensor);
  void ProcessSynapseOutputs(
      const HabanaOperatorPtr& habana_op,
      torch::jit::Node* node);
  void ProcessStridedInsertAtOutput(
      torch::jit::Node*,
      HabanaOperatorPtr,
      torch::jit::Stack&,
      synapse_helpers::graph&);
  void ProcessSynapseShapeTensors(
      const HabanaOperatorPtr& habana_op,
      torch::jit::Node* node);
  c10::ScalarType getNodeScalarType(torch::jit::Node* node);
  void handlePrimNodes(torch::jit::Node* node);
  void handleRestrideNode(torch::jit::Node* node, bool is_restride_cl);
  void handleMetaOps(torch::jit::Node* node);

  bool IsOutputToRestride(torch::jit::Value* val);
  torch::jit::Value* GetRestridedOutvalue(torch::jit::Value* val);

  bool isPermuteInGraphOutputs(torch::jit::Value* value);
  bool IsOutputToPermute(torch::jit::Value* value);
  torch::jit::Value* GetPermuteOutvalue(torch::jit::Value* val);
  bool isCollective(torch::jit::Node* node);

  torch::jit::Node* GetUnpackNodeFromTensorList(torch::jit::Value* val);

  void PrintRecipeInputs();

  std::shared_ptr<RecipeValueSpec> GetCachedRecipe(
      std::shared_ptr<RecipeArgumentSpec>& spec_key) {
    auto rvpsh{RecipeCacheLRU::get_cache().get(spec_key)};
    if (nullptr != rvpsh && nullptr == rvpsh->jit_graph_) {
      rvpsh->jit_graph_ = jit_ir_graph;
    }
    return rvpsh;
  }
  void ReturnCachedRecipe(RecipeValueSpec& rv) {
    rv.set_use_flag(false);
  }

  void create_duplicate_syn_tensor(
      at::Tensor* tensor,
      torch::jit::Value* value_in,
      bool persistence = true);
  bool IsCustomOptimizer(std::string node_str);

  // Patching related
  void AddAtenIntermediate(
      const IValPtrShared& ivpsh,
      const PtTensorInfoShared ti) {
    void* buffp = ti->get_buffer();
    intermediate_tinfos.emplace_back(ti);
    aten_intermediates.push_back(ivpsh->toTensor());
    // We might have outputs that are duplicate of
    // persistent intermediate tensors
    if (false == ti->is_ZST()) {
      buff_to_intermediate_ivpsh_map.emplace(buffp, ivpsh);
    }
  }
  void AddAtenIntermediate(
      const IValPtrShared& ivpsh,
      const std::string& syntensor_name,
      const std::string& ir_name,
      const uint64_t tensor_id) {
    const auto& pttensor = ivpsh->toTensor();
    PtTensorInfoShared ti = std::make_shared<PtTensorInfo>(
        pttensor, syntensor_name, ir_name, watch_tensor_flag_, tensor_id);
    AddAtenIntermediate(ivpsh, ti);
  }
  void AddAtenIntermediate(
      const IValPtrShared& ivpsh,
      const std::string& syntensor_name,
      const ValPtr& vp,
      const uint64_t tensor_id) {
    std::string ir_name = "%" + vp->debugName();
    AddAtenIntermediate(ivpsh, syntensor_name, ir_name, tensor_id);
  }

  // Member functions related to lowering IR to Synapse
  void Clear(bool is_shape_inference = false);
  void CopyInputStack(torch::jit::Stack& input_st);

  // TODO: Check whether the swap destruct paradigm provides any performance
  // gain
  template <class T>
  void ClearMember(T& m_container) {
    T empty;
    using std::swap;
    swap(m_container, empty);
  }

  void CompileSynapseGraph();
  void ConstructPatchingTable();
  void DumpTensors_pre(RecipeValueSpec& rv);
  void DumpTensors(RecipeValueSpec& rv);
  void ExecuteSynapseGraph();
  void FlattenAndLinkInputTIVs(RecipeValueSpec& rv);
  void OrderInputs();
  void OrderOutputTinfos(RecipeValueSpec& rv);
  void ProcessInputStack(torch::jit::Stack& input_st);
  void RestoreInputTensorMetadata();
  void UpdateOutputs();
  void UpdateOutputs(RecipeValueSpec& rv);

  // --------------------

  // Dynamic shape specific parts
  uint64_t current_bucket_id_{};
  std::shared_ptr<habana_helpers::DynamicBucketInfo> current_dbipsh_{};
  std::shared_ptr<habana_helpers::CompilationStatistics> statistics_;

  void CreateDynamicBucketInputShapes(
      habana_helpers::DynamicBucketInfo::InpTensorShapes& shape_map);

  synapse_helpers::tensor& AllocateSynapseTensor(
      const HabanaOperatorPtr& habana_op,
      at::Tensor& pt_tensor);
  habana::ShapeInfo m_map_shape;
  void run_shape_inference(
      const ShapeInfo::InferencePass& pass,
      DynamicShapeInfo& graph_input_info);
  void run_pass();
  void handle_pass_exception(
      DynamicShapeInfo& graph_input_info,
      const PassException& e);
  void CompileAndRunDynamicGraph(DynamicShapeInfo& graph_input_info);
  torch::jit::Stack CreateStack(
      const torch::jit::Stack& stack,
      habana_helpers::DynamicBucketInfo::InpTensorShapes& dynamic_shapes);
  inline void try_run_shape_inference(
      const ShapeInfo::InferencePass& pass,
      DynamicShapeInfo& graph_input_info) {
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_DYNAMIC_PASS_FALLBACK)) {
      try {
        run_shape_inference(pass, graph_input_info);
      } catch (const PassException& e) {
        handle_pass_exception(graph_input_info, e);
      }
    } else {
      run_shape_inference(pass, graph_input_info);
    }
  }
};

} // namespace habana
