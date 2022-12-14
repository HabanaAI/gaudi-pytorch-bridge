/*******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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

#include <memory>
#include "habana_lazy/hpu_lazy_cache.h"
#include "program.h"

namespace habana {
namespace program {

/*
 * Executor for clustered program.
 */
class Executor {
 public:
  struct Frame {
    std::vector<at::IValue> inputs;
  };
  Executor(ClusteredProgramSPtr program);

  /*
   * Run program according to program's schedule.
   * Semantics is similar to HabanaLaunchOpPT::run, inputs should be
   * on the given stack and outputs will be stored there.
   */
  void Run(torch::jit::Stack& stack);

 private:
  void RunCluster(Cluster* cluster);
  ClusteredProgramSPtr program_;

  std::unordered_map<Cluster::Id, Frame> frames_;
};

/*
 * Creates executor for given lazy_jit_graph.
 * The program associated with executor is looked up from cache.
 *
 * TODO: is it enough to use cached_graph_key() to identify program?
 */
std::unique_ptr<Executor> CreateExecutor(
    const std::shared_ptr<LazyJitGraph>& lazy_jit_graph);

} // namespace program
} // namespace habana