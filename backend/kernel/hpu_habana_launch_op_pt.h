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
#include <utility>

#include <ATen/Tensor.h>
#include <absl/hash/hash.h>
#include <absl/types/variant.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include "backend/kernel/control_edges_processing.h"
#include "backend/kernel/hpu_habana_cache.h"
#include "backend/kernel/hpu_habana_meta_op_list.h"
#include "backend/kernel/hpu_shape_inference.h"

#include "backend/habana_operator.h"
#include "backend/helpers/compilation_statistics.h"
#include "backend/jit_graph_cache.h"
#include "habana_helpers/thread_pool/thread_pool.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_arg_spec.h"
#include "habana_lazy/visualize.h"

namespace habana {
using IValPtrSharedToTesorInfoMap =
    std::unordered_map<IValPtrShared, PtTensorInfoShared>;

IValPtrShared GetPrimListConstructNodeOuputIValue(
    torch::jit::Node* node,
    CValuePtrToIValuePtrMap& value_to_ivalue);

// Api to create shape or H2d tensors with zero memory allocations.
// Information in shape tensor is embedded in tensor meta data
at::Tensor createDynamicTensor(const std::vector<int64_t>&, synTensorType);

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

// Forward declaration
class PersistenceMarkerPassData;

class HabanaLaunchOpPT : public std::enable_shared_from_this<HabanaLaunchOpPT> {
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

  void run(
      torch::jit::Stack& stack,
      std::optional<std::vector<at::Tensor>> allocated_outputs = {},
      bool dry_run = false);

  HabanaLaunchOpPT& getInstance() {
    return *this;
  };
  static void cleanUp();

  static std::unordered_set<std::string> watchlist_;
  static std::unordered_map<size_t, habana_helpers::InpTensorShapes>
      ref_input_shape_map_;

  c10::ScalarType getNodeScalarType(torch::jit::Node* node);
  void set_lazy_front_end_info(
      std::shared_ptr<habana_lazy::HbLazyFrontEndInfoToBackend> info);
  bool is_hccl_send_mark_step();
  void CompileSynapseGraph(bool allocate_rval = true);
  void ConstructPatchingTable();
  void UpdateSynapsePermutations();
  void StoreShapeAgnosticGraph();
  void ExecuteSynapseGraph(synapse_helpers::hpuStream_t hpu_stream);
  static void ExecuteSynapseCache(
      synapse_helpers::hpuStream_t hpu_stream,
      size_t graph_key_with_perm,
      at::ArrayRef<torch::jit::IValue> input_refs,
      HabanaLaunchOpPT* hbLaunchOp,
      std::shared_ptr<RecipeValueSpec> cur_rvalpsh,
      std::shared_ptr<RecipeArgumentSpec> cur_rargpsh,
      std::optional<std::vector<at::Tensor>> allocated_outputs,
      bool dry_run = false);
  static void ExecuteSynapseCacheTask(
      size_t graph_key_with_perm,
      std::shared_ptr<HabanaLaunchOpPT> hbLaunchOp,
      bool dry_run = false);
  // To clear the static variables
  void ClearStatics(bool is_shape_inference = false);
  void DumpTensors(RecipeValueSpec& rv);
  void DumpTensors_pre(RecipeValueSpec& rv);
  bool get_enable_shape_agnostic_caching_() {
    return enable_shape_agnostic_caching_;
  }

  std::shared_ptr<habana::OptimizedJITGraphAndMetaData>
  get_jit_graph_and_meta_data() const {
    return jit_graph_and_meta_data_;
  }
  bool get_enable_tensor_dump_() const {
    return enable_tensor_dump_;
  }

  at::ArrayRef<torch::jit::IValue> get_input_refs_() const {
    return input_refs;
  }

  std::shared_ptr<RecipeValueSpec> get_cur_rvalpsh() const {
    return cur_rvalpsh;
  }

  std::shared_ptr<RecipeArgumentSpec> get_cur_rargpsh() const {
    return cur_rargpsh;
  }

  std::optional<std::vector<at::Tensor>> get_allocated_outputs_() const {
    return allocated_outputs_;
  }

  torch::jit::Stack& get_input_stack_() {
    return input_st_copy;
  }

  void set_input_stack_(torch::jit::Stack stack) {
    input_st_copy = stack;
  }

  void copy_input_stack_(torch::jit::Stack& stack) {
    stack = input_st_copy;
  }

  std::shared_ptr<std::vector<IValPtrShared>> get_intermediate_tensors_ptrsh()
      const {
    return intermediate_tensors_ptr_sh_;
  }

  // A map holding the ival hash and inputidx. 1-1 map for all inputs
  std::unordered_map<int64_t, int64_t> ival_hash_to_input_index_map_ = {};

  /// Property---------------------------------///-------------Comments--------------///---Read/Write-in-Lowering-Thread---///---Read/Write-in-Compile-Thread----///--Read/Write-in-Execute-Thread
  /// hpu_stream-------------------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  /// input_refs-------------------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  /// cur_rvalpsh------------------------------///-----------------------------------///---------------Write---------------///----------------Write--------------///-----------Read
  /// jit_graph_and_meta_data_-----------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  /// input_st_copy----------------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  /// enable_tensor_dump_----------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  /// refine_ds_enabled_-----------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  /// htensor_wbuff_size-----------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  /// htensor_wbuff----------------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  /// num_inputs-------------------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  /// syn_graph_ptr_---------------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  /// current_dbipsh_--------------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Write-for-dynamic-shapes
  /// cur_rargpsh------------------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  // duplicate_intermediate_to_outtinfo_map----///-----------------------------------///---------------Write---------------///----------------Read---------------///------------NA
  // persistence_marker_pass_data_ptr_---------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // lazy_info---------------------------------///----------------LAZY---------------///----------------NA-----------------///-----------------NA----------------///------------NA
  // dry_run_----------------------------------///-----------------------------------///---------------Write---------------///------Read-for-Dynamic-Shapes------///-----------Read
  // node_bcast_map_---------------------------///----------------LAZY---------------///----------------NA-----------------///----------------NA-----------------///------------NA
  // op_name-----------------------------------///-----------------------------------///---------------Write---------------///----------------NA-----------------///------------NA
  // graph_index_------------------------------///-----------------------------------///---------------Write---------------///----------------NA-----------------///------------NA
  // jit_ir_graph------------------------------///-----------------------------------///---------------Write---------------///---------------Read----------------///-----------Read
  // id_str------------------------------------///-----------------------------------///---------------Write---------------///----------------NA-----------------///------------NA
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
  // m_const_checksum_map----------------------///-----------------------------------///-----------------NA----------------///---------------Write---------------///------------NA
  // checksum_map_mtx--------------------------///-----------------------------------///-----------------NA----------------///---------------Write---------------///------------NA
  // input_tivs--------------------------------///-----------------------------------///---------------Write---------------///---------------Write---------------///------------NA
  // output_tensorinfos------------------------///-----------------------------------///-----------------NA----------------///---------------Write---------------///------------NA
  // input_tiv_map-----------------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // duplicate_input_tivs----------------------///-----------------------------------///-----------------NA----------------///---------------Write---------------///------------NA
  // buff_to_input_ivpsh_map-------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // buff_to_intermediate_ivpsh_map------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // buff_to_output_ivpsh_map------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // buff_to_syn_tensor_map--------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // duplicate_outtinfos-----------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///------------NA
  // dma_input_idx-----------------------------///-----------------------------------///-----------------NA----------------///-----------------NA----------------///------------NA
  // appended_index----------------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // intermediate_index------------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // shape_index-------------------------------///----------Dynamic-Shapes-----------///---------------Write---------------///-----------------NA----------------///------------NA
  // aten_intermediates------------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///-----------Read
  // intermediate_tinfos-----------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///------------NA
  // aten_dma_inputs---------------------------///-----------------------------------///-----------------NA----------------///-----------------NA----------------///------------NA
  // dma_input_tensorinfos---------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  // shape_tensor_tinfos-----------------------///----------Dynamic-Shapes-----------///---------------Write---------------///----------------Read---------------///------------NA
  // non_persistent_intermediate_tinfos--------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
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
  // watch_tensor_flag_------------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // enable_fast_shape_inf_--------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // tensor_dump_numel_------------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///-----------Read
  // cur_ds_token_-----------------------------///----------Dynamic-Shapes-----------///---------------Write---------------///-----------------NA----------------///------------NA
  // tdmp_dir_name_----------------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // tdmp_file_name_pre_-----------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  // tdmp_file_name_---------------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  // iteration_count_--------------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///-----------Read
  // jit_to_synapse_node_idx_map---------------///-----------------------------------///---------------Write---------------///----------------Read---------------///------------NA
  // blocking_nodes_vec------------------------///-----------------------------------///-----------------NA----------------///----------------Write--------------///------------NA
  // blocking_syn_nodes_vec--------------------///-----------------------------------///-----------------NA----------------///----------------Write--------------///------------NA
  // blocked_syn_nodes_vec---------------------///-----------------------------------///-----------------NA----------------///----------------Write--------------///------------NA
  // memory_reuse_pairs------------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///------------NA
  // collective_kernels_info-------------------///-----------------------------------///---------------Write---------------///----------------Read---------------///------------NA
  // dfs_time_in_out_map-----------------------///----------Dynamic-Shapes-----------///----------------NA-----------------///----------------Write--------------///------------NA
  // dfs_cnt-----------------------------------///----------Dynamic-Shapes-----------///----------------NA-----------------///----------------Write--------------///------------NA
  // execution_mode_---------------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // allocated_outputs_------------------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///-----------Read
  // intermediate_syn_tensors_count------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///------------NA
  // intermediate_tensors_ptr_sh_--------------///-----------------------------------///---------------Write---------------///-----------------NA----------------///-----------Read
  std::shared_ptr<synapse_helpers::graph> syn_graph_ptr_ = nullptr;

 private:
  // user stream info
  synapse_helpers::hpuStream_t hpu_stream;
  at::ArrayRef<torch::jit::IValue> input_refs;
  std::shared_ptr<RecipeValueSpec> cur_rvalpsh{nullptr};
  std::shared_ptr<habana::OptimizedJITGraphAndMetaData>
      jit_graph_and_meta_data_ = nullptr;
  torch::jit::Stack input_st_copy;
  bool enable_tensor_dump_{false};
  bool refine_ds_enabled_{false};
  unsigned htensor_wbuff_size{0};
  uint64_t htensor_wbuff{0};
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
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const bool debug_;
  std::string id_str_ = std::string();
  std::string op_strs_ = std::string();
  size_t graph_key_ = 0;
  std::vector<std::vector<int64_t>> out_shapes{};

  size_t prim_nodes_ival_counter{0};
  size_t restride_node_swap_counter{0};
  size_t restride_node_out_val_counter{0};

  std::vector<TensorMetaData> input_tms;

  // We keep a vector of kernels so that the context memory
  //   for each kernel is retained till graph execution
  // This is done to enable reuse of PT and synapse tensors and their processing
  std::vector<HabanaOperatorPtr> habana_kernels;

  // map between PT and synapse tensors
  std::deque<synapse_helpers::tensor> meta_syn_tensors;

  std::vector<IValPtrShared> pt_stack_sh;
  CValuePtrToIValuePtrMap value_to_ivalue;
  std::unordered_map<IValPtrShared, SharedSynTensorOrRefListPtr>
      pt_to_synapse_tensors;

  std::unordered_map<IValPtrShared, PtTensorInfoShared>
      ivalue_to_tensor_info_map;

  static std::unordered_map<int, size_t> m_const_checksum_map
      GUARDED_BY(checksum_map_mtx);
  static std::mutex checksum_map_mtx;
  static void insertConstantChecksum(int id, size_t checksum) {
    std::lock_guard<std::mutex> lock(checksum_map_mtx); // Acquire the lock
    m_const_checksum_map[id] =
        checksum; // Insert or replace the value in a single line
  }

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

  // tinfos corresponding to non-persistent intermediates tensors.
  // for eager shape agnostic patching for synapse graph.
  std::vector<PtTensorInfoShared> non_persistent_intermediate_tinfos;

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
  bool watch_tensor_flag_{false};
  bool enable_fast_shape_inf_{false};
  int tensor_dump_numel_{0};

  uint64_t cur_ds_token_{0};

  std::string tdmp_dir_name_;
  std::string tdmp_file_name_pre_;
  std::string tdmp_file_name_;

  size_t iteration_count_ = 0;

  std::unordered_map<torch::jit::Node*, std::vector<synNodeId>>
      jit_to_synapse_node_idx_map;
  std::vector<std::pair<torch::jit::Value*, torch::jit::Node*>>
      memory_reuse_pairs;
  std::vector<std::shared_ptr<habana_helpers::collective_kernel_info>>
      collective_kernels_info;

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

  std::shared_ptr<std::vector<IValPtrShared>> intermediate_tensors_ptr_sh_{
      nullptr};

  // Main function responsible for constructing a synapse graph from
  // 1. JIT IR Graph
  // 2. Input Stack
  // Currently this funciton is used for shape inference as well
  void BuildSynapseGraph(
      std::shared_ptr<synapse_helpers::graph>& syn_graph,
      bool is_shape_inference = false);

  void setSynapsePermuteFlag(
      synapse_helpers::tensor& out_syntensor,
      PtTensorInfoShared& ti,
      IValPtrShared ivpsh);
  void preProcessInputs();
  torch::jit::Stack getStackForNode(torch::jit::Node* node);
  bool nodeOutputPersistencePerValue(
      torch::jit::Node* node,
      torch::jit::Value* value_out);
  bool IsValueExternal(torch::jit::Value* value);
  OutputMetaDataVector nodeOutputMetaData(torch::jit::Node* node);
  void CreateValueToIvalueMapForInputs();
  void InitiateSynlaunchTimeCapture(RecipeValueSpec& rv);
  void ProcessHabanaFusedOpWithDS();
  void CreateFirstDynamicBucket();
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
      std::vector<IdxTensorTup>& intermediate_shape_tensor_cs);
  void handlePrimNodes(torch::jit::Node* node);
  void handlePrimConstantNode(torch::jit::Node* node);
  void handlePrimListConstructNode(torch::jit::Node* node);
  void handleRestrideNode(torch::jit::Node* node, bool is_restride_cl);
  void handleMetaOps(torch::jit::Node* node);

  std::shared_ptr<RecipeValueSpec> GetCachedRecipe(
      std::shared_ptr<RecipeArgumentSpec>& spec_key) {
    auto rvpsh{RecipeCacheLRU::get_cache().get(spec_key)};
    if (nullptr != rvpsh && nullptr == rvpsh->jit_graph_) {
      rvpsh->jit_graph_ = jit_ir_graph_;
    }
    return rvpsh;
  }
  void ReturnCachedRecipe(RecipeValueSpec& rv);
  void DuplicateSynapseGraph();
  void ValidateInputsAndOutputsAndDisableSA(
      at::ArrayRef<torch::jit::IValue>& input_refs);
  void MaybePrintDuplicateGraphInformation(
      const std::shared_ptr<synapse_helpers::graph>& graph_ptr,
      std::vector<synTensorHandleMap>& tensors_map,
      std::vector<synNodeHandleMap>& nodes_map [[maybe_unused]],
      std::string cache_hit_or_miss);

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
  // To clear the non static members
  void ClearMembers(bool is_shape_inference = false);
  void CopyInputStack(torch::jit::Stack& input_st);

  // No need to allocate for lazy eager shape agnostic cache hit scenario
  // API for populating Synapse tensor info which needs to be used
  // to find constant section ID for Synapse graph inputs only
  void PostCompilationStepForConstTensors(RecipeValueSpec& rv);

  void EvictSynapseRecipe(size_t& dsi_bucket_id);
  void FlattenAndLinkInputTIVs(RecipeValueSpec& rv);
  void OrderInputs();
  void OrderOutputTinfos(RecipeValueSpec& rv);
  void ProcessInputStack(torch::jit::Stack& input_st);
  void RestoreInputTensorMetadata();
  void UpdateOutputs();
  void UpdateOutputs(RecipeValueSpec& rv);
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
  bool update_max = false;

  void FillMaxValues(
      const HabanaOperatorPtr& habana_op,
      const torch::jit::Stack& input_stack,
      std::unordered_map<int64_t, std::vector<int64_t>>& index2maxvalues);

  void UpdateMaxValues(
      const HabanaOperatorPtr& habana_op,
      const torch::jit::Stack& input_stack,
      std::unordered_map<int64_t, std::vector<int64_t>>& index2maxvalues);

  void UpdatePTStack(DynamicShapeInfo& graph_input_info);

  std::shared_ptr<habana_helpers::CompilationStatistics> statistics_;

  void CreateStaticCompilationDBI(size_t graph_key_with_perm);

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
  void CompileAndRunDynamicGraph(DynamicShapeInfo& graph_input_info);
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

  static std::unordered_set<std::string> disabled_jit_ir_ops_;

  torch::jit::Stack create_stack_for_node(
      const torch::jit::Node* node,
      bool& flag,
      std::unordered_map<CValPtr, torch::jit::IValue>& val_to_ival_map);

  int64_t get_output_tensors_count(
      const HabanaOperatorPtr& habana_op,
      synapse_helpers::graph& syn_graph);

  void process_outputs(
      const HabanaOperatorPtr& habana_op,
      torch::jit::Node* node,
      std::unordered_map<CValPtr, torch::jit::IValue>& val_to_ival_map,
      std::unordered_map<int64_t, at::Tensor>& tidx_to_tensor_map);

  void visit_prim_node(
      const torch::jit::Node* node,
      std::unordered_map<CValPtr, torch::jit::IValue>& val_to_ival_map);

  template <bool DynamicShapes>
  void RunHybridSif(
      std::unordered_map<int64_t, at::Tensor>& tidx_to_tensor_map);
  // --------------------
};
} // namespace habana
