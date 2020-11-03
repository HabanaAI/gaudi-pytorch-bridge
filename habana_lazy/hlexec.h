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
#include "torch/csrc/jit/ir/ir.h"

namespace habana_lazy {
namespace exec {

using Graph = torch::jit::Graph;
using JitValue = torch::jit::Value;
using HabanaLazyValue = habana_lazy::ir::Value;
using JitIValue = torch::jit::IValue;
using GraphPtr = std::shared_ptr<Graph>;
using JitValuePtr = std::shared_ptr<JitValue>;
using ScopePtr = torch::jit::ScopePtr;

using HabanaLazyTensorPtr = habana_lazy::HbLazyTensor*;
using HabanaLazyTensorPtrList = std::vector<HabanaLazyTensorPtr>;

using LazyValueToJitValueMap = std::unordered_map<HabanaLazyValue,
                                                  JitValue*,
                                                  habana_lazy::ir::ValueHash,
                                                  habana_lazy::ir::ValueEqual>;
/**
 * This is the lazy execution JIT Graph creator class. An object of this class will
 * manage creation of pytorch JIT graph. It will also manage mapping
 * or binding of Habana Lazy tensor (hltensor) with torch::jit::Value
 */
class HlExec {
 public:
  HlExec();
  HlExec(ScopePtr scope);

  virtual ~HlExec() {}

  /**
   * This method binds the habana ir nodes to JIT
   * Value pointers
   */
  void Bind(const HabanaLazyTensorPtrList& inputs);

  /**
   * This method creates the JIT IR Graph
   * Inputs:
   *   nodes: Vector of Lazy IR nodes
   *   inputs: Lazy value pointers representing input tensors
   *   outputs: Lazy value pointers representing output tensors
   * Returns:
   *   Tuple containing -
   *     map : Lazy input value pointer -> JIT IR input value pointers
   *     map : Lazy output value pointer -> JIT IR output value pointers
   */
  std::tuple<LazyValueToJitValueMap, LazyValueToJitValueMap>
    Create(const ir::NodePtrList nodes,
           const ir::ValueList inputs,
           const ir::ValueList outputs);

  /**
   * This method calls torch::jit optimzer passes.
   * Optionally, habana specific optimzers can be added.
   * TBD: Add optimzer levels and take in a mask from caller
   * to control optimization passes applied on the graph.
   */
  void Optimize(); //opt_level_mask=0x0);

  /**
  * This method calls the Habana Graph Lowering kernel
  */
  void Launch(torch::jit::Stack& stack);

  void DumpGraph() {
    std::stringstream strbuff;
    std::streambuf * oldbuff = std::cout.rdbuf(strbuff.rdbuf());
    std::cout << "JIT IR graph\n";
    mp_g_->dump();
    std::string str = strbuff.str();
    std::cout.rdbuf(oldbuff);
    PT_LAZY_DEBUG(str);
  }

  GraphPtr get_graph() {
    return mp_g_;
  }

 private:
  GraphPtr mp_g_;
  std::map<HabanaLazyTensorPtr, JitValuePtr> m_tensorbind_;
};

}; // namespace exec
}; // namespace habana
