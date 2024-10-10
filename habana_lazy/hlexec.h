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
#include <iostream>
#include <string>
#include <vector>
#include "backend/habana_device/HPUStream.h"
#include "backend/helpers/runtime_config.h"
#include "backend/jit_graph_cache.h"
#include "hpu_lazy_tensors.h"
#include "ir.h"
#include "lazy_executor.h"
#include "torch/csrc/jit/ir/ir.h"

namespace habana_lazy {
namespace exec {

using Graph = torch::jit::Graph;
using JitValue = torch::jit::Value;
using HabanaLazyValue = habana_lazy::ir::Value;
using HabanaLazyOutput = habana_lazy::ir::Output;
using JitIValue = torch::jit::IValue;
using GraphPtr = std::shared_ptr<Graph>;
using OptimizedJITGraphAndMetaDataPtr =
    std::shared_ptr<habana::OptimizedJITGraphAndMetaData>;
using JitValuePtr = std::shared_ptr<JitValue>;
using ScopePtr = torch::jit::ScopePtr;
using HabanaLazyTensorPtr = habana_lazy::HbLazyTensor*;
using HabanaLazyTensorPtrList = std::vector<HabanaLazyTensorPtr>;
using LazyOutputToJitValueMap = std::unordered_map<
    HabanaLazyOutput,
    JitValue*,
    habana_lazy::ir::OutputHash,
    habana_lazy::ir::OutputEqual>;

/**
 * Define Singleton class to select/deselect the optimization passes
 */
class OptPassCfg {
 private:
  static OptPassCfg* p_instance_;

  OptPassCfg() {
    SetDefaultOptFlags();
  }

 public:
  OptPassCfg(const OptPassCfg&) = delete;
  OptPassCfg& operator=(const OptPassCfg&) = delete;

 public:
  static OptPassCfg* GetInstance() {
    if (p_instance_ == nullptr) {
      p_instance_ = new OptPassCfg();
    }

    return p_instance_;
  }

  void SetDeadCodeElimination(const bool flag) {
    pass.enable_eliminate_dead_code = flag;
  }
  void SetCSEElimination(const bool flag) {
    pass.enable_eliminate_common_subexpression = flag;
  }
  void SetConstPooling(const bool flag) {
    pass.enable_constant_pooling = flag;
  }
  void SetPeepholeOpt(const bool flag) {
    pass.enable_peephole_optimization = flag;
  }
  void SetSubgraphRewrite(const bool flag) {
    pass.enable_subgraph_rewrite = flag;
  }
  void SetFuseTMM(const bool flag) {
    pass.enable_fuse_t_mm_optimization = flag;
  }
  void SetFuseBnRelu(const bool flag) {
    pass.enable_fuse_bn_relu_optimization = flag;
  }
  void SetPermutePass(const bool flag) {
    pass.enable_permute_pass = flag;
  }
  void SetReplaceInplaceOps(const bool flag) {
    pass.enable_replace_inplace_ops = flag;
  }
  void SetReplaceViews(const bool flag) {
    pass.enable_replace_views = flag;
  }
  void SetBnParamRecalc(const bool flag) {
    pass.enable_bn_param_recalculation = flag;
  }
  void SetFuseConvBn(const bool flag) {
    pass.enable_fuse_conv_bn_optimization = flag;
  }

  bool IsEnabledDeadCodeElimination() const {
    return pass.enable_eliminate_dead_code;
  }
  bool IsEnabledCSEElimination() const {
    return pass.enable_eliminate_common_subexpression;
  }
  bool IsEnabledConstPooling() const {
    return pass.enable_constant_pooling;
  }
  bool IsEnabledPeepholeOpt() const {
    return pass.enable_peephole_optimization;
  }
  bool IsEnabledSubgraphRewrite() const {
    return pass.enable_subgraph_rewrite;
  }
  bool IsEnabledFuseTMM() const {
    return pass.enable_fuse_t_mm_optimization;
  }
  bool IsEnabledFuseBnRelu() const {
    return pass.enable_fuse_bn_relu_optimization;
  }
  bool IsEnabledPermutePass() const {
    return false;
  }
  bool IsEnabledReplaceInplaceOps() const {
    return pass.enable_replace_inplace_ops;
  }
  bool IsEnabledReplaceViews() const {
    return pass.enable_replace_views;
  }
  bool IsEnabledBnParamRecalc() const {
    bool flag =
        habana_helpers::IsInferenceMode() || pass.enable_bn_param_recalculation;
    return flag;
  }
  bool IsEnabledFuseConvBn() const {
    bool flag = habana_helpers::IsInferenceMode() &&
        pass.enable_fuse_conv_bn_optimization;
    return flag;
  }

  void SetDefaultOptFlags() {
    pass.enable_eliminate_dead_code = true;
    pass.enable_eliminate_common_subexpression = true;
    pass.enable_constant_pooling = true;
    pass.enable_peephole_optimization = true;
    pass.enable_subgraph_rewrite = true;
    pass.enable_fuse_t_mm_optimization = true;
    pass.enable_fuse_bn_relu_optimization = true;
    pass.enable_permute_pass = true;
    pass.enable_replace_inplace_ops = true;
    pass.enable_replace_views = true;
    pass.enable_bn_param_recalculation = false;
    pass.enable_fuse_conv_bn_optimization = false;
  }

  void BkupAndDisableAndAllOptPass() {
    // Create a backup of the currently enabled passes and disable all
    // optimization passes
    if (!backup_available) {
      pass_cfg_backup = pass;
      backup_available = true;

      // Disable the passes
      pass.enable_eliminate_dead_code = false;
      pass.enable_eliminate_common_subexpression = false;
      pass.enable_constant_pooling = false;
      pass.enable_peephole_optimization = false;
      pass.enable_subgraph_rewrite = false;
      pass.enable_fuse_t_mm_optimization = false;
      pass.enable_fuse_bn_relu_optimization = false;
      pass.enable_permute_pass = false;
      pass.enable_replace_inplace_ops = false;
      pass.enable_replace_views = false;
      pass.enable_bn_param_recalculation = false;
      pass.enable_fuse_conv_bn_optimization = false;
    }
  }

  void RestoreOptPass() {
    if (backup_available) {
      pass = pass_cfg_backup;
      backup_available = false;
    }
  }

 private:
  struct PassCfg {
    bool enable_eliminate_dead_code;
    bool enable_eliminate_common_subexpression;
    bool enable_constant_pooling;
    bool enable_peephole_optimization;
    bool enable_subgraph_rewrite;
    bool enable_fuse_t_mm_optimization;
    bool enable_fuse_bn_relu_optimization;
    bool enable_permute_pass;
    bool enable_replace_inplace_ops;
    bool enable_replace_views;
    bool enable_bn_param_recalculation;
    bool enable_fuse_conv_bn_optimization;
  };

  struct PassCfg pass;
  struct PassCfg pass_cfg_backup;
  bool backup_available = false;
};

/**
 * This is the lazy execution JIT Graph creator class. An object of this class
 * will manage creation of pytorch JIT graph. It will also manage mapping or
 * binding of Habana Lazy tensor (hltensor) with torch::jit::Value
 */
class HlExec {
 public:
  HlExec();
  HlExec(ScopePtr scope);

  virtual ~HlExec() {}

  /**
   * This method finds the duplicate inputs in the stack
   */
  void FindDuplicateInStack(
      const ir::PostOrderData& po_data,
      torch::jit::Stack& stack,
      std::vector<size_t>& parent_vec,
      std::vector<bool>& is_duplicate_vec);

  /**
   * This method prunes the duplicate inputs from the stack
   */
  void PruneDuplicateStackInputs(
      torch::jit::Stack& stack,
      std::vector<bool>& is_duplicate_vec);

  /**
   * This method prunes the duplicate inputs from the JIT IR Graph
   */
  void PruneDuplicateGraphInputs(
      std::vector<size_t>& parent_vec,
      std::vector<bool>& is_duplicate_vec);

  /**
   * This method prunes the redundant inputs from the input stack in lazy cache
   * hit case
   */
  void deleteRedundantInputsFromInputStack(torch::jit::Stack& stack);

  /**
   * This method prunes the redundant inputs from the JIT IR Graph and the stack
   */
  void SearchAndDeleteRedundantInputs(
      ir::PostOrderData& po_data,
      torch::jit::Stack& stack,
      std::vector<torch::jit::Value*>& redundant_inputs);

  /**
   * This method gets an optimized JIT IR graph from cache
   * or creates the JIT IR Graph
   * Inputs:
   *   nodes: Vector of Lazy IR nodes
   *   stack: Stack for the inputs
   *   inputs: Lazy value pointers representing input tensors
   *   outputs: Lazy value pointers representing output tensors
   *   str: post order graph string
   */
  void GetOrCreate(ir::PostOrderData& po_data, torch::jit::Stack& stack);

  /**
   * This method calls torch::jit optimizer passes.
   * Optionally, habana specific optimzers can be added.
   */
  void Optimize(
      torch::jit::Stack& stack,
      std::vector<torch::jit::Value*>& redundant_inputs);

  /**
   * This method calls the Habana Graph Lowering kernel
   */
  void Launch(
      torch::jit::Stack& stack,
      const c10::hpu::HPUStream& stream,
      bool dry_run = false);

  GraphPtr get_graph() {
    return mp_g_;
  }

  static size_t GetGraphIndex(
      size_t hash,
      at::ArrayRef<torch::jit::IValue> input_refs);
  size_t GetGraphHash() {
    return m_g_hash_;
  }

  void set_graph(GraphPtr p_g) {
    mp_g_ = p_g;
    mp_g_and_meta_data_ =
        std::make_shared<habana::OptimizedJITGraphAndMetaData>();
    mp_g_and_meta_data_->set_cached_graph(mp_g_);
  }

  void set_hash(size_t p_h) {
    m_g_hash_ = p_h;
  }

  void set_fwd_graph_hash(size_t p_h) {
    m_fwd_graph_hash_ = p_h;
  }

  size_t get_fwd_graph_hash() {
    return m_fwd_graph_hash_;
  }

  void set_fwd_graph_stack_map(std::vector<uint64_t> graph_input_stack_map) {
    m_fwd_graph_stack_map_ = graph_input_stack_map;
  }

  void set_graph_key(size_t graphKey) {
    mp_g_and_meta_data_->set_cached_graph_key(graphKey);
  }

  size_t get_graph_key() {
    return mp_g_and_meta_data_->get_cached_graph_key();
  }

  std::string get_opstrs() {
    return mp_g_and_meta_data_->get_cached_opstrs();
  }

  void set_opstrs(std::string opStrs) {
    mp_g_and_meta_data_->set_cached_opstrs(opStrs);
  }

  void set_lazy_front_end_info(
      std::shared_ptr<HbLazyFrontEndInfoToBackend> info) {
    lazyInfo = info;
  }

  OptimizedJITGraphAndMetaDataPtr GetJITGraphMetaDataPtr() {
    return mp_g_and_meta_data_;
  }

  static void Serialize(std::ostream& os);
  static void Deserialize(std::istream& is);
  static void LoadDSCheckpoint(const std::string& path);
  static void SaveDSCheckpoint(const std::string& path);

 private:
  /**
   * This method creates the JIT IR Graph
   * Inputs:
   *   nodes: Vector of Lazy IR nodes
   *   inputs: Lazy value pointers representing input tensors
   *   outputs: Lazy value pointers representing output tensors
   */
  void Create(
      const ir::NodePtrList& nodes,
      const ir::ValueList& inputs,
      const ir::ValueList& outputs,
      torch::jit::Stack& stack,
      std::vector<torch::jit::Value*>& redundant_inputs);

  void CreateNodeBcastMap(const ir::NodePtrList& nodes);
  void IdentifyAndSetGraphNodes(const ir::NodePtrList& nodes);
  GraphPtr mp_g_;
  OptimizedJITGraphAndMetaDataPtr mp_g_and_meta_data_{nullptr};
  size_t m_g_hash_;
  size_t m_fwd_graph_hash_ = 0;
  std::vector<uint64_t> m_fwd_graph_stack_map_;
  std::shared_ptr<HbLazyFrontEndInfoToBackend> lazyInfo = nullptr;
  std::vector<bool> node_bcast_map_;
  static std::unordered_map<size_t, size_t> s_graphIndexMap;
  static size_t s_graphIndex;
};

}; // namespace exec
}; // namespace habana_lazy
