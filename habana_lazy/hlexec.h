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
#include <iostream>
#include <string>
#include <vector>
#include "habana_lazy/hpu_lazy_cache.h"
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
    std::shared_ptr<habana_lazy::OptimizedJITGraphAndMetaData>;
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
  void SetWeightPermutePass(const bool flag) {
    pass.enable_weight_permute_pass = flag;
  }
  void SetReplaceInplaceOps(const bool flag) {
    pass.enable_replace_inplace_ops = flag;
  }
  void SetReplaceViews(const bool flag) {
    pass.enable_replace_views = flag;
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
    return pass.enable_permute_pass;
  }
  bool IsEnabledWeightPermutePass() const {
    return pass.enable_weight_permute_pass;
  }
  bool IsEnabledReplaceInplaceOps() const {
    return pass.enable_replace_inplace_ops;
  }
  bool IsEnabledReplaceViews() const {
    return pass.enable_replace_views;
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
    pass.enable_weight_permute_pass = false;
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
      pass.enable_weight_permute_pass = false;
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
    bool enable_weight_permute_pass;
    bool enable_replace_inplace_ops;
    bool enable_replace_views;
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
   * This method gets an optimized JIT IR graph from cache
   * or creates the JIT IR Graph
   * Inputs:
   *   nodes: Vector of Lazy IR nodes
   *   stack: Stack for the inputs
   *   inputs: Lazy value pointers representing input tensors
   *   outputs: Lazy value pointers representing output tensors
   *   str: post order graph string
   */
  void GetOrCreate(const ir::PostOrderData& po_data, torch::jit::Stack& stack);

  /**
   * This method calls torch::jit optimizer passes.
   * Optionally, habana specific optimzers can be added.
   */
  void Optimize(torch::jit::Stack& stack);

  /**
   * This method calls the Habana Graph Lowering kernel
   */
  void Launch(torch::jit::Stack& stack);

  std::string DumpGraph() {
    std::stringstream strbuff;
    std::streambuf* oldbuff = std::cout.rdbuf(strbuff.rdbuf());
    std::cout << "JIT IR graph\n";
    mp_g_->dump();
    std::string str = strbuff.str();
    std::cout.rdbuf(oldbuff);
    return str;
  }

  GraphPtr get_graph() {
    return mp_g_;
  }

  void set_graph(GraphPtr p_g) {
    mp_g_ = p_g;
  }

  void set_lazy_front_end_info(
      std::shared_ptr<HbLazyFrontEndInfoToBackend> info) {
    lazyInfo = info;
  }

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
      torch::jit::Stack& stack);

  GraphPtr mp_g_;
  OptimizedJITGraphAndMetaDataPtr mp_g_and_meta_data_;
  size_t m_g_hash_;
  std::shared_ptr<HbLazyFrontEndInfoToBackend> lazyInfo = nullptr;
};

}; // namespace exec
}; // namespace habana_lazy
