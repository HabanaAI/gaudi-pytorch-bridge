/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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

#include "backend/helpers/dynamic_bucket_info.h"
#include "backend/helpers/dynamic_bucket_info_utils.h"
#include "backend/helpers/dynamic_shape_infer.h"
#include "backend/helpers/tensor_info.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/jit_graph_cache.h"
#include "backend/kernel/hpu_shape_inference.h"

namespace habana {
using IValPtrSharedToTesorInfoMap =
    std::unordered_map<IValPtrShared, PtTensorInfoShared>;

IValPtrShared GetPrimListConstructNodeOuputIValue(
    torch::jit::Node* node,
    CValuePtrToIValuePtrMap& value_to_ivalue);

using InputSymbolMap = std::unordered_map<std::string, std::shared_ptr<double>>;

// Api to create shape or H2d tensors with zero memory allocations.
// Information in shape tensor is embedded in tensor meta data
at::Tensor createDynamicTensor(const std::vector<int64_t>&, synTensorType);
at::Tensor createDynamicTensor(
    const std::vector<int64_t>& size,
    synTensorType type,
    c10::ScalarType dtype);

struct DynamicShapeInfo {
  habana_helpers::InpTensorShapes act_input_tshapes;
  habana_helpers::InpTensorShapes min_input_tshapes;
  habana_helpers::InpTensorShapes max_input_tshapes;
  habana_helpers::DynamicDimsPolicy min_policy;
  habana_helpers::DynamicDimsPolicy max_policy;
  size_t current_bucket_id{};
  // The min_fallback_seq_num holds the index of char from environment
  // variable specifying fallback sequence, the fallback char is extracted
  // from sequence string based on this index.
  uint64_t min_fallback_seq_num{};
  uint64_t max_fallback_seq_num{};
  // To go from one fallback sequence to next the value of index is incremented
  // by 2 for eg: Fallback seq = 4,3,2 -> the gap between consequtive number is
  // 2
  void set_next_min_policy() {
    min_fallback_seq_num += 2;
  };
  void set_next_max_policy() {
    max_fallback_seq_num += 2;
  };
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

class PermutationInfoSaver {
 public:
  PermutationInfoSaver(
      std::shared_ptr<habana::OptimizedJITGraphAndMetaData> jit_graph)
      : jit_graph_(jit_graph){};
  void add_permutation(
      uint64_t index,
      synapse_helpers::layouts::MemoryPermutation permutation) {
    permutation_info_.push_back({index, permutation});
  }
  ~PermutationInfoSaver() {
    jit_graph_->store_permutation_info(std::move(permutation_info_));
  }

 private:
  OptimizedJITGraphAndMetaData::PermutationInfo permutation_info_;
  std::shared_ptr<OptimizedJITGraphAndMetaData> jit_graph_;
};

struct ExecutionControl {
  std::optional<size_t> graph_key_with_perm_{};
  bool is_shape_agnostic_cache_hit_ = false;
  bool no_compile_ = false;
  void cached_task(size_t graph_key_with_perm) {
    graph_key_with_perm_ = graph_key_with_perm;
    no_compile_ = true;
  }
  void sag_cache_hit() {
    is_shape_agnostic_cache_hit_ = true;
  }
  void no_compile() {
    no_compile_ = true;
  }
};

class HabanaLaunchOpPT;

namespace HabanaLaunchOpPipeline {

class PipelineCallBase;
extern PipelineCallBase NoPipeline;

void LoweringTask(
    std::unique_ptr<HabanaLaunchOpPT>&& launch_op,
    torch::jit::Stack& stack,
    std::optional<std::vector<at::Tensor>> allocated_outputs,
    std::optional<std::vector<std::vector<int64_t>>> output_shapes = {});
} // namespace HabanaLaunchOpPipeline

namespace OpInfo {
std::string DumpOpInfo(
    const at::OperatorName& opname,
    const torch::jit::Stack& input_stack);
} // namespace OpInfo

// Forward declaration
class PersistenceMarkerPassData;

class HabanaLaunchOpPT {
 public:
  explicit HabanaLaunchOpPT(
      std::shared_ptr<habana::OptimizedJITGraphAndMetaData>
          optimized_jit_graph_and_meta_data);
  ~HabanaLaunchOpPT();

  void CompileGraphWithRange(
      torch::jit::Stack& stack,
      habana_helpers::ResultShapes& input_ranges,
      habana_helpers::Bucket& new_bucket,
      size_t& new_recipe_key,
      std::shared_ptr<habana_helpers::CompilationStatistics> statpsh,
      std::shared_ptr<habana_helpers::DynamicBucketInfo> dbipsh);

  void UpdatePatchingInformation(
      RecipeValueSpec& rv,
      bool is_graph_empty,
      // do not update output shapes
      bool is_ds_patching_update = false,
      std::optional<
          std::reference_wrapper<const std::unordered_map<int64_t, at::Tensor>>>
          tidx_to_tensor_map_opt = std::nullopt,
      const std::unordered_map<synTensor, synTensor>&
          synapse_orig_to_new_handle = {},
      const bool is_shape_agnostic_graph = false);

  void run(
      torch::jit::Stack& stack,
      std::optional<std::vector<at::Tensor>> allocated_outputs = {},
      std::optional<std::vector<std::vector<int64_t>>> output_shapes = {},
      bool dry_run = false,
      HabanaLaunchOpPipeline::PipelineCallBase& pipeline_execution =
          HabanaLaunchOpPipeline::NoPipeline);

  void ProcessIntermediateSymbolicShapes(
      std::shared_ptr<torch::jit::Graph>& jit_graph);
  void CreateORUpdateExprSymbolicTable(RecipeValueSpec* rv = nullptr);
  void CreateIValueForNodeInputs(
      torch::jit::Node* node,
      habana_helpers::DynamicSIFInfo* dsi);
  void CreateIValueForNodeOutputs(
      torch::jit::Node* node,
      habana_helpers::DynamicSIFInfo* dsi);
  void UpdateIshapeForNodeInputs(torch::jit::Node* node, RecipeValueSpec& rv);
  void UpdateIshapeForNodeOuputs(torch::jit::Node* node, RecipeValueSpec& rv);
  void CreateValueIShapeMapForNode(
      torch::jit::Node* node,
      torch::jit::Node* rv_node,
      const torch::jit::Stack& input_stack,
      OutputMetaDataVector& meta_vec);
  void CreateValueToIShapeMapForInputs(
      std::shared_ptr<torch::jit::Graph>& jit_graph);
  void UpdateValueIShapeMapForListUnpack(
      torch::jit::Node* node,
      RecipeValueSpec& rv);
  void UpdateValueToIShapeMapForInputs(
      std::shared_ptr<torch::jit::Graph>& jit_graph,
      RecipeValueSpec& rv);
  static void cleanUp();

  static std::unordered_map<size_t, habana_helpers::InpTensorShapes>&
  ref_input_shape_map();

  c10::ScalarType getNodeScalarType(torch::jit::Node* node);
  void set_lazy_front_end_info(
      std::shared_ptr<habana_lazy::HbLazyFrontEndInfoToBackend> info);
  bool is_hccl_send_mark_step();
  void CompileSynapse();
  std::shared_ptr<RecipeValueSpec> CompileSynapseGraphAndPatchTable();
  std::shared_ptr<synapse_helpers::graph::recipe_handle> CompileSynapseGraph();
  void ConstructPatchingTableAndAtenOutputs(
      RecipeValueSpec& rv,
      const std::shared_ptr<synapse_helpers::graph::recipe_handle>& recipe);
  void UpdateSynapsePermutations(
      RecipeValueSpec& rvs,
      const synapse_helpers::graph::recipe_handle& recipe);
  void ApplyOutputPermutationsFromCache();
  void StoreCompiledInformation(std::shared_ptr<RecipeValueSpec>& rvs);
  void ExecuteSynapse();
  void ExecuteSynapseGraph();
  void ExecuteSynapseCache(size_t graph_key_with_perm);
  // To clear the static variables
  void ClearStatics(bool is_shape_inference = false);

  bool get_enable_shape_agnostic_caching_() {
    return enable_shape_agnostic_caching_;
  }

  std::shared_ptr<habana::OptimizedJITGraphAndMetaData>
  get_jit_graph_and_meta_data() const {
    return jit_graph_and_meta_data_;
  }

  at::ArrayRef<torch::jit::IValue> get_input_refs() const {
    return input_refs;
  }

  std::shared_ptr<RecipeArgumentSpec> get_cur_rargpsh() const {
    return cur_rargpsh;
  }

  std::optional<std::vector<at::Tensor>> get_allocated_outputs() const {
    return allocated_outputs_;
  }

  torch::jit::Stack& get_input_stack() {
    return input_st_copy;
  }

  void set_input_stack(torch::jit::Stack stack) {
    input_st_copy = std::move(stack);
  }

  void set_symbol_values(InputSymbolMap symint_value) {
    in_symbol_value_map = std::move(symint_value);
  }

  InputSymbolMap& get_symbol_values() {
    return in_symbol_value_map;
  }

  std::shared_ptr<VecOfIValPtrSh> get_intermediate_tensors_ptrsh() const {
    return intermediate_tensors_ptr_sh_;
  }

  bool get_enable_2stage_pipeline() const {
    return enable_2stage_pipeline_;
  }

  bool get_enable_4stage_pipeline() const {
    return enable_4stage_pipeline_;
  }

  bool get_enable_optim_output_sif() const {
    return enable_optim_output_sif_;
  }

  std::shared_ptr<synapse_helpers::graph::recipe_handle> get_hpu_op_recipe()
      const {
    return hpu_op_recipe_;
  }

  bool get_is_shape_agnostic_supported() const {
    return is_shape_agnostic_supported_;
  }

  void set_require_h2d_st(bool require_h2d, bool require_st) {
    require_h2d_ = require_h2d;
    require_st_ = require_st;
  }

  bool get_require_h2d_st() const {
    return require_h2d_ || require_st_;
  }

  // A map holding the ival hash and inputidx. 1-1 map for all inputs
  std::unordered_map<int64_t, int64_t> ival_hash_to_input_index_map_ = {};

  /// Property---------------------------------///-------------Comments--------------///---Read/Write-in-Lowering-Thread---///---Read/Write-in-Compile-Thread----///--Read/Write-in-Execute-Thread
  /// hpu_stream-------------------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  /// input_refs-------------------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  /// cur_rvalpsh------------------------------///-----------------------------------///---------------Write---------------///----------------Write--------------///-----------Read
  /// jit_graph_and_meta_data_-----------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  /// input_st_copy----------------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  /// refine_ds_enabled_-----------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  /// num_inputs-------------------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  /// syn_graph_ptr_---------------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  /// current_dbipsh_--------------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Write-for-dynamic-shapes
  /// cur_rargpsh------------------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  // duplicate_intermediate_to_outtinfo_map----///-----------------------------------///---------------Write---------------///----------------Read---------------///------------NA
  // persistence_marker_pass_data_ptr_---------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // dry_run_----------------------------------///-----------------------------------///---------------Write---------------///------Read-for-Dynamic-Shapes------///-----------Read
  // node_bcast_map_---------------------------///----------------LAZY---------------///----------------NA-----------------///----------------NA-----------------///------------NA
  // op_name-----------------------------------///-----------------------------------///---------------Write---------------///----------------NA-----------------///------------NA
  // graph_index_------------------------------///-----------------------------------///---------------Write---------------///----------------NA-----------------///------------NA
  // jit_ir_graph------------------------------///-----------------------------------///---------------Write---------------///---------------Read----------------///-----------Read
  // op_strs-----------------------------------///-----------------------------------///---------------Write---------------///----------------NA-----------------///------------NA
  // graph_key---------------------------------///-----------------------------------///---------------Write---------------///---------------Read----------------///------------NA
  // out_shapes--------------------------------///-----------------------------------///---------------Write---------------///----------------NA-----------------///------------NA
  // prim_nodes_ival_counter-------------------///-----------------------------------///---------------Write---------------///----------------NA-----------------///------------NA
  // restride_node_swap_counter----------------///-----------------------------------///---------------Write---------------///----------------NA-----------------///------------NA
  // restride_node_out_val_counter-------------///-----------------------------------///---------------Write---------------///----------------NA-----------------///------------NA
  // input_tms---------------------------------///-----------------------------------///---------------Write---------------///----------------NA-----------------///------------NA
  // habana_kernels----------------------------///-----------------------------------///---------------Write---------------///----------------NA-----------------///------------NA
  // meta_syn_tensors--------------------------///-----------------------------------///---------------Write---------------///----------------NA-----------------///------------NA
  // pt_stack_sh-------------------------------///-----------------------------------///---------------Write---------------///----------------NA-----------------///------------NA
  // value_to_ivalue---------------------------///-----------------------------------///---------------Write---------------///---------------Read----------------///-----------Read
  // pt_to_synapse_tensors---------------------///-----------------------------------///---------------Write---------------///---------------Read----------------///------------NA
  // ivalue_to_tensor_info_map-----------------///-----------------------------------///---------------Write---------------///---------------Write---------------///------------NA
  // input_tivs--------------------------------///-----------------------------------///---------------Write---------------///---------------Write---------------///------------NA
  // output_tensorinfos------------------------///-----------------------------------///-----------------NA----------------///---------------Write---------------///------------NA
  // input_tiv_map-----------------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // duplicate_input_tivs----------------------///-----------------------------------///-----------------NA----------------///---------------Write---------------///------------NA
  // buff_to_input_ivpsh_map-------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // buff_to_intermediate_ivpsh_map------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // buff_to_output_ivpsh_map------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // buff_to_syn_tensor_map--------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // duplicate_outtinfos-----------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///------------NA
  // appended_index----------------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // intermediate_index------------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // shape_index-------------------------------///----------Dynamic-Shapes-----------///---------------Write---------------///-----------------NA----------------///------------NA
  // aten_intermediates------------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///-----------Read
  // intermediate_tinfos-----------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///------------NA
  // dma_input_tensorinfos---------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  // shape_tensor_tinfos-----------------------///----------Dynamic-Shapes-----------///---------------Write---------------///----------------Read---------------///------------NA
  // num_tensor_inputs-------------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///------------NA
  // use_persistent_tensors--------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // pt_stack----------------------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///-----------Write
  // output_tensorinfo_map---------------------///-----------------------------------///---------------Write---------------///----------------Write--------------///------------NA
  // duplicate_input_to_outtinfo_map-----------///-----------------------------------///---------------Write---------------///----------------Read---------------///------------NA
  // duplicate_output_to_outtinfo_map----------///-----------------------------------///---------------Write---------------///----------------Read---------------///------------NA
  // sif_tidx_to_tinfo_map---------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///------------NA
  // enable_caching_---------------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  // enable_graph_caching_---------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  // enable_eager_caching_---------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // enable_shape_agnostic_caching_------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  // enable_2stage_pipeline_-------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  // enable_4stage_pipeline_-------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  // enable_optim_output_sif_-------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  // enable_fast_shape_inf_--------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // cur_ds_token_-----------------------------///----------Dynamic-Shapes-----------///---------------Write---------------///-----------------NA----------------///------------NA
  // jit_to_synapse_node_idx_map---------------///-----------------------------------///---------------Write---------------///----------------Read---------------///------------NA
  // collective_kernels_info-------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///------------NA
  // execution_mode_---------------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // allocated_outputs_------------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///-----------Read
  // aten_outputs_-----------------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///-----------Read
  // intermediate_tensors_ptr_sh_--------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///-----------Read
  // dma_inputs_-------------------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///-----------Read
  // syn_launch_info_--------------------------///-----------------------------------///-----Write-(in-cache-hit-case)-----///-----Write-(in-cache-miss-case)----///-----------Read
  // external_tensor_info_indexes_-------------///-----------------------------------///-----Write-(in-cache-hit-case)-----///-----Write-(in-cache-miss-case)----///-----------Read
  // permutation_info_saver_-------------------///-----------------------------------///---------------Write---------------///----------------Write--------------///------------NA
  // hpu_op_recipe_----------------------------///-----------------------------------///-----------------------------------///----------------Write--------------///-----------Read
  // is_shape_agnostic_supported_--------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  std::shared_ptr<synapse_helpers::graph> syn_graph_ptr_ = nullptr;
  VecOfIValPtrSh aten_outputs_;
  static void RunHybridSif(
      std::shared_ptr<torch::jit::Graph> jit_ir_graph,
      torch::jit::Stack& inputs,
      CValPtrtoIValueMap& val_to_ival_map);

 private:
  // user stream info
  synapse_helpers::hpuStream_t hpu_stream_;
  at::ArrayRef<torch::jit::IValue> input_refs;
  std::shared_ptr<RecipeLauncher> recipe_launcher_{nullptr};
  std::shared_ptr<habana::OptimizedJITGraphAndMetaData>
      jit_graph_and_meta_data_ = nullptr;
  bool enable_user_dynamic_ranges = false;
  std::vector<habana_helpers::RangeInfo> m_range_infos;
  torch::jit::Stack input_st_copy;
  bool refine_ds_enabled_{false};
  size_t num_inputs{0};
  std::shared_ptr<habana_helpers::DynamicBucketInfo> current_dbipsh_{};
  std::shared_ptr<RecipeArgumentSpec> cur_rargpsh{nullptr};
  std::unique_ptr<PersistenceMarkerPassData> persistence_marker_pass_data_ptr_;
  std::shared_ptr<habana_lazy::HbLazyFrontEndInfoToBackend> lazy_info_ =
      nullptr;

  bool dry_run_ = false;
  std::string name_ = std::string();
  size_t graph_index_ = 0;
  std::shared_ptr<torch::jit::Graph> jit_ir_graph_;
  std::string id_str_ = std::string();
  std::string op_strs_ = std::string();
  size_t graph_key_ = 0;
  size_t graph_key_with_perm_ = 0;
  size_t graph_symint_hash_ = 0;
  size_t graph_perm_hash_ = 0;
  std::vector<std::vector<int64_t>> out_shapes{};

  size_t prim_nodes_ival_counter{0};
  size_t restride_node_swap_counter{0};
  size_t restride_node_out_val_counter{0};

  std::vector<TensorMetaData> input_tms;

  // We keep a vector of kernels so that the context memory
  //   for each kernel is retained till graph execution
  // This is done to enable reuse of PT and synapse tensors and their
  // processing
  std::vector<HabanaOperatorPtr> habana_kernels;

  // map between PT and synapse tensors
  std::deque<synapse_helpers::tensor> meta_syn_tensors;

  VecOfIValPtrSh pt_stack_sh;
  CValuePtrToIValuePtrMap value_to_ivalue;
  InputSymbolMap in_symbol_value_map;
  habana_helpers::DynamicSIFInfo ds_sif_info_;
  size_t sym_expr_hash_ = 0;
  std::unordered_set<std::string> st_backend_create_op_list;
  std::unordered_map<IValPtrShared, SharedSynTensorOrRefListPtr>
      pt_to_synapse_tensors;

  std::unordered_map<IValPtrShared, PtTensorInfoShared>
      ivalue_to_tensor_info_map;

  void update_syn_launch_info(uint64_t oldAddress, uint64_t newAdress);
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
  std::unordered_map<void*, synapse_helpers::tensor_or_ref>
      buff_to_syn_tensor_map;
  std::vector<PtTensorInfoShared> duplicate_outtinfos;

  size_t appended_index{0};
  size_t intermediate_index{0};
  size_t shape_index{0};

  // The persistent intermediates are stored in the following two vectors.
  // aten_intermediates is used for storing intermediates which are usually
  // marked persistent by persistenceMarkingPass.
  std::vector<at::Tensor> aten_intermediates;
  // tinfos corresponding to aten_intermediates.
  std::vector<PtTensorInfoShared> intermediate_tinfos;

  // tinfos corresponding to aten_intermediates.
  std::deque<PtTensorInfoShared> dma_input_tensorinfos;
  // tinfos corresponding to shape tensor.
  std::vector<PtTensorInfoShared> shape_tensor_tinfos;

  // caching :: begin

  // The inputs holding data usually are of type tensor and tensorList.
  // The following member keeps track of total number of tensor and tensorList
  // inputs
  size_t num_tensor_inputs{0};

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const bool use_persistent_tensors;

  torch::jit::Stack* pt_stack{nullptr};
  uint64_t t_compile_ns{0};

  // Making the cache eviction policy as lru as default

  IValPtrSharedToTesorInfoMap output_tensorinfo_map;
  IValPtrSharedToTesorInfoMap duplicate_input_to_outtinfo_map;
  IValPtrSharedToTesorInfoMap duplicate_intermediate_to_outtinfo_map;
  IValPtrSharedToTesorInfoMap duplicate_output_to_outtinfo_map;

  // Output shape inference map
  std::unordered_map<int64_t, PtTensorInfoShared> sif_tidx_to_tinfo_map;

  // caching :: end

  bool enable_caching_{false};
  bool enable_graph_caching_{false};
  bool enable_eager_caching_{false};
  bool enable_shape_agnostic_caching_{false};
  bool enable_2stage_pipeline_{false};
  bool enable_4stage_pipeline_{false};
  bool enable_fast_shape_inf_{false};
  bool enable_optim_output_sif_{false};

  uint64_t cur_ds_token_{0};

  std::unordered_map<torch::jit::Node*, std::vector<synNodeId>>
      jit_to_synapse_node_idx_map;
  habana_helpers::CollectiveKernelInfos collective_kernels_info;

  // Execution mode based on frontend type
  habana_helpers::HabanaFrontendTypes execution_mode_{
      habana_helpers::HabanaFrontendTypes::INVALID};

  std::optional<std::vector<at::Tensor>> allocated_outputs_;

  // Count for intermediate synapse tensors in a graph
  // intermediates syn tensors can be both persistent and non-persistent
  int64_t intermediate_syn_tensors_count_{0};

  // Count for implicit synapse tensors which are duplicate to inputs
  // and are persistent and not present in pt_to_synapse_tensors map
  int64_t implicit_syn_tensors_count_{0};

  // Count for JIT IR nodes for which meta attribute is set
  // such nodes kernel skips adding synapse node to the synapse graph
  // At this point, only StridedView op may use it in eager mode.
  int64_t meta_attribute_nodes_count_{0};

  std::shared_ptr<VecOfIValPtrSh> intermediate_tensors_ptr_sh_{nullptr};

  VecOfIValPtrSh dma_inputs_{};

  std::vector<synLaunchTensorInfo> syn_launch_info_{};
  std::vector<size_t> external_tensor_info_indexes_{};

  // This object writes permutation data to the jit graph cache
  // We only use it in normal flow, because we need to set output permutation
  // in lowering thread in order to have ability execute compile and execution
  // in another threads
  std::unique_ptr<PermutationInfoSaver> permutation_info_saver_;
  std::shared_ptr<synapse_helpers::graph::recipe_handle> hpu_op_recipe_{
      nullptr};
  bool is_shape_agnostic_supported_ = false;
  bool require_h2d_ = false;
  bool require_st_ = false;

  // Main function responsible for constructing a synapse graph from
  // 1. JIT IR Graph
  // 2. Input Stack
  // Currently this funciton is used for shape inference as well
  void BuildSynapseGraph(
      std::shared_ptr<synapse_helpers::graph>& syn_graph,
      SynBuildCache& syn_build_cache,
      bool is_shape_inference = false);

  void BuildSynapseGraphLite(
      std::shared_ptr<synapse_helpers::graph>& syn_graph,
      SynBuildCache& syn_build_cache);

  void HandleOutputSIFException(
      torch::jit::Node* node,
      const HabanaOperatorPtr& habana_op,
      RecipeValueSpec& rv,
      size_t& outputs_meta_index,
      SynBuildCache& syn_build_cache);

  void HandleViewBaseOutputShape(
      torch::jit::Node* node,
      habana_helpers::DynamicSIFInfo& dsisss);

  uint64_t HandleFrontendShapeTensorOp(torch::jit::Node* node);
  void HandleOutputExprMappedJITGraph(
      std::shared_ptr<torch::jit::Graph>& rv_jit_graph,
      RecipeValueSpec& rv,
      SynBuildCache& syn_build_cache);
  void HandleOutputExprUnMappedJITGraph(
      RecipeValueSpec& rv,
      std::shared_ptr<synapse_helpers::graph>& syn_graph,
      SynBuildCache& syn_build_cache);
  void setSynapsePermuteFlag(
      synapse_helpers::tensor& out_syntensor,
      PtTensorInfoShared& ti,
      IValPtrShared ivpsh);
  void preProcessInputs();
  torch::jit::Stack getStackForNode(torch::jit::Node* node);
  habana_helpers::IShapeList getInputIShapesForNode(
      torch::jit::Node* node,
      RecipeValueSpec& rv);
  habana_helpers::IShapeList getOutputIShapesForNode(
      torch::jit::Node* node,
      RecipeValueSpec& rv);
  bool nodeOutputPersistencePerValue(
      torch::jit::Node* node,
      torch::jit::Value* value_out);
  bool IsValueExternal(torch::jit::Value* value);
  OutputMetaDataVector nodeOutputMetaData(torch::jit::Node* node);
  void CreateValueToIvalueMapForInputs();
  void ReCreateValueToIvalueMapForInputs(
      std::shared_ptr<torch::jit::Graph>& jit_graph);
  void ResetIShapeUpdateStatus(RecipeValueSpec& rv);
  void InitiateSynlaunchTimeCapture(RecipeLauncher& rv);
  void ProcessHabanaFusedOpWithDS(
      HabanaLaunchOpPipeline::PipelineCallBase& pipeline_execution);
  void CreateFirstDynamicBucket(
      std::shared_ptr<RecipeArgumentSpec> rargpsh_graph = nullptr);
  void DumpStaticCompilationStatistics(
      size_t graph_key_with_perm,
      bool is_compile = false);

  void HandleMappedTensor(
      CValPtr value_in,
      const HabanaOperatorPtr& habana_op,
      SharedSynTensorOrRefListPtr& tensorList);
  void HandleUnmappedTensor(
      CValPtr value_in,
      const HabanaOperatorPtr& habana_op,
      SharedSynTensorOrRefListPtr& tensorList,
      std::string idx);
  void HandleMappedandUnmappedTensor(
      CValPtr value_in,
      const HabanaOperatorPtr& habana_op,
      SharedSynTensorOrRefListPtr& tensorList,
      std::string idx);
  void GetSynapseInputs(
      const HabanaOperatorPtr& habana_op,
      torch::jit::Node* node);
  const std::string& GetSynapseGraphName() {
    return SetAndGetSynapseGraphName(name_, graph_index_);
  }
  std::string& SetAndGetSynapseGraphName(
      const std::string& name,
      size_t g_index);
  void SetOpName(const std::string& name);
  PtTensorInfoShared ProcessPersistentNodeOutput(
      const IValPtrShared& ivpsh,
      const ValPtr& vp,
      const synapse_helpers::tensor& out_syntensor);
  int64_t ProcessSynapseOutputs(
      const HabanaOperatorPtr& habana_op,
      torch::jit::Node* node,
      InferOutputMetaRetType& outputs);
  void ProcessSynapseShapeTensors(
      const HabanaOperatorPtr& habana_op,
      std::vector<size_t>& intermediate_shape_tensors,
      std::vector<size_t>& inputs_shape_tensors,
      bool isRecursiveCall = false);
  void ProcessShapeTensorsCS(
      const InferOutputMetaRetType& output,
      std::vector<IdxTensorTuple>& intermediate_shape_tensor_cs);
  void handlePrimNodes(torch::jit::Node* node, SynBuildCache& syn_build_cache);
  void handlePrimConstantNode(
      torch::jit::Node* node,
      SynBuildCache& syn_build_cache);
  void handlePrimListConstructNode(torch::jit::Node* node);
  void handleRestrideNode(
      torch::jit::Node* node,
      SynBuildCache& syn_build_cache,
      bool is_restride_cl);
  void handleMetaOps(torch::jit::Node* node);

  std::shared_ptr<RecipeHolder> GetCachedRecipe(
      std::shared_ptr<RecipeArgumentSpec>& spec_key) {
    auto rh{RecipeCacheLRU::get_cache().get(spec_key)};
    if (nullptr != rh && nullptr == rh->rvs_->jit_graph_) {
      rh->rvs_->jit_graph_ = jit_ir_graph_;
    }
    return rh;
  }

  void ConstructDuplicateShapeMap(
      const std::vector<synTensorHandleMap>& tensorsMap,
      std::vector<std::pair<synTensor, std::vector<int64_t>>>&
          duplicate_tensors_shape_map);
  void ValidateInputsAndOutputsAndDisableSA(
      at::ArrayRef<torch::jit::IValue>& input_refs);
  static void MaybePrintDuplicateGraphInformation(
      const synapse_helpers::graph& graph_ptr,
      const std::vector<synTensorHandleMap>& tensors_map,
      const std::vector<synNodeHandleMap>& nodes_map,
      bool is_cache_hit);

  void create_duplicate_syn_tensor(
      at::Tensor* tensor,
      torch::jit::Value* value_in,
      bool persistence = true);

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
        pttensor, syntensor_name, ir_name, tensor_id);
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
  // To clear the non static members
  void ClearMembers(bool is_shape_inference = false);
  void CopyInputStack(torch::jit::Stack& input_st);

  // No need to allocate for lazy eager shape agnostic cache hit scenario
  // API for populating Synapse tensor info which needs to be used
  // to find constant section ID for Synapse graph inputs only
  using permuteInfo =
      std::pair<synapse_helpers::layouts::MemoryPermutation, bool>;
  permuteInfo GetPermuteInfo(StorageExtraMeta* _smeta);
  void SetPermuteInfo(
      StorageExtraMeta* _new_smeta,
      StorageExtraMeta* _smeta,
      permuteInfo _info);
  void PostCompilationStepForConstTensors(
      synapse_helpers::graph::recipe_handle& recipe);
  void UpdateTensorInfoMap(std::shared_ptr<c10::IValue> src, void* ptr);
  void HandleTensorWithZeroSize(
      at::Tensor& tensor,
      ConstantInformation::key_t key);
  void HandleChecksum(
      at::Tensor& tensor,
      size_t data_size,
      bool checksum_found,
      ConstantInformation::checksum_t checksum,
      ConstantInformation::key_t key,
      char* data_ptr,
      size_t old_size,
      int device_id);
  void HandleTensorWithNewChecksum(
      at::Tensor& tensor,
      size_t section_size,
      ConstantInformation::checksum_t checksum,
      ConstantInformation::key_t key,
      char* section_data_ptr,
      size_t old_size,
      int device_id);
  void HandleTensorWithExistingChecksumInCache(
      ConstantInformation::id_t const_id,
      ConstantInformation::checksum_t checksum,
      ConstantInformation::key_t key,
      at::Tensor& tensor);
  void HandleTensorWithChecksumOnDevice(
      ConstantInformation::id_t const_id,
      ConstantInformation::checksum_t checksum,
      ConstantInformation::key_t key);
  void SerializeConstSection(
      at::Tensor& tensor,
      size_t section_size,
      char* section_data_ptr,
      const size_t key);
  void DeserializeConstSection(at::Tensor& tensor, const size_t key);
  void EvictSynapseRecipe(size_t& dsi_bucket_id);
  void FlattenAndLinkInputTIVs(RecipeValueSpec& rv);
  void OrderInputs();
  void OrderOutputTinfos(RecipeValueSpec& rv);
  void ProcessInputStack(torch::jit::Stack& input_st);
  void RestoreInputTensorMetadata();
  void UpdateOutputs();
  void UpdateRecipeOutputs();
  void validateOutputShapeNonDynamic(
      const HabanaOperatorPtr& HabanaKernel,
      const InferOutputMetaRetType& output_shape_handle,
      const std::string& opname);
  void validateOutputShapeDynamic(
      const HabanaOperatorPtr& HabanaKernel,
      const InferOutputMetaRetType& output_shape_handle,
      const std::string& opname);
  void validateOutputShape(
      const HabanaOperatorPtr& HabanaKernel,
      const InferOutputMetaRetType& output_shape_handle,
      const synapse_helpers::graph& syn_graph,
      const std::string& opname);

  // --------------------

  // Dynamic shape specific functions
  size_t current_bucket_id_{};
  bool updatemax_graph = false;

  void FillMaxValues(
      const HabanaOperatorPtr& habana_op,
      const torch::jit::Stack& input_stack,
      std::unordered_map<int64_t, std::vector<int64_t>>& index2maxvalues);

  void UpdateMaxValues(
      const HabanaOperatorPtr& habana_op,
      const torch::jit::Stack& input_stack,
      std::unordered_map<int64_t, std::vector<int64_t>>& index2maxvalues);

  void UpdatePTStack(DynamicShapeInfo& graph_input_info);

  void RevertH2DMinMaxData();

  std::shared_ptr<habana_helpers::CompilationStatistics> statistics_;

  void CreateStaticCompilationDBI(size_t graph_key_with_perm);
  void CreateDynamicDBI(size_t graph_key_with_perm);

  void CreateDynamicBucketInputShapes(
      habana_helpers::InpTensorShapes& shape_map);

  void ProcessDynamicBucketInputShapesWithH2D(
      habana_helpers::InpTensorShapes& shape_map);

  synapse_helpers::tensor& AllocateSynapseTensor(
      const HabanaOperatorPtr& habana_op,
      at::Tensor& pt_tensor,
      std::string idx = std::string());
  habana::ShapeInfo m_map_shape;
  void run_shape_inference(
      const ShapeInfo::InferencePass& pass,
      DynamicShapeInfo& graph_input_info);
  void run_pass();
  void handle_pass_exception(
      DynamicShapeInfo& graph_input_info,
      const PassException& e);
  void CompileAndRunDynamicGraph(
      DynamicShapeInfo& graph_input_info,
      HabanaLaunchOpPipeline::PipelineCallBase& pipeline_execution);
  torch::jit::Stack CreateStack(
      const torch::jit::Stack& stack,
      habana_helpers::InpTensorShapes& dynamic_shapes);
  void SetH2DMinMaxData(
      const torch::jit::Stack& stack,
      habana_helpers::InpTensorShapes& dynamic_shapes,
      const ShapeInfo::InferencePass& pass);
  inline void try_run_shape_inference(
      const ShapeInfo::InferencePass& pass,
      DynamicShapeInfo& graph_input_info) {
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_DYNAMIC_PASS_FALLBACK)) {
      try {
        run_shape_inference(pass, graph_input_info);
      } catch (const PassException& e) {
        RestoreInputTensorMetadata();
        handle_pass_exception(graph_input_info, e);
      }
    } else {
      run_shape_inference(pass, graph_input_info);
    }
    RestoreInputTensorMetadata();
  }

  // Fast shape inference specific members and functions
  // Currently fast shape inference is realized through a pass which works in
  // hybrid mode. This hybrid shape inference pass uses OutputShapeInf
  // for JIT OPs whenever possible, otherwise falls back to
  // AllocateAndAddSynapseNode for the output shape computation.

  static std::unordered_set<std::string>& disabled_jit_ir_ops();

  torch::jit::Stack create_stack_for_node(
      const torch::jit::Node* node,
      bool& flag,
      CValPtrtoIValueMap& val_to_ival_map);

  int64_t get_output_tensors_count(
      const HabanaOperatorPtr& habana_op,
      synapse_helpers::graph& syn_graph);

  void process_outputs(
      const HabanaOperatorPtr& habana_op,
      torch::jit::Node* node,
      CValPtrtoIValueMap& val_to_ival_map,
      std::unordered_map<int64_t, at::Tensor>& tidx_to_tensor_map);

  void visit_prim_node(
      const torch::jit::Node* node,
      CValPtrtoIValueMap& val_to_ival_map);

  template <bool DynamicShapes>
  bool RunHybridSif(
      std::unordered_map<int64_t, at::Tensor>& tidx_to_tensor_map,
      std::shared_ptr<std::vector<InferNodeParams>> node_params_ptr = nullptr);
  // --------------------

  ExecutionControl execution_control_;
};
} // namespace habana
