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
    enable_eliminate_dead_code = true;
    enable_eliminate_common_subexpression = true;
    enable_constant_pooling = true;
    enable_peephole_optimization = true;
    enable_subgraph_rewrite = false;
    enable_fuse_t_mm_optimization = false;
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

  bool enable_eliminate_dead_code;
  bool enable_eliminate_common_subexpression;
  bool enable_constant_pooling;
  bool enable_peephole_optimization;
  bool enable_subgraph_rewrite;
  bool enable_fuse_t_mm_optimization;
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
  void Optimize();

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

 private:
  /**
   * This method creates the JIT IR Graph
   * Inputs:
   *   nodes: Vector of Lazy IR nodes
   *   inputs: Lazy value pointers representing input tensors
   *   outputs: Lazy value pointers representing output tensors
   */
  void Create(
      const ir::NodePtrList nodes,
      const ir::ValueList inputs,
      const ir::ValueList outputs);

  GraphPtr mp_g_;
  std::map<HabanaLazyTensorPtr, JitValuePtr> m_tensorbind_;
};

}; // namespace exec
}; // namespace habana_lazy
