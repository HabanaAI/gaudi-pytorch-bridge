/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
#include "backend/habana_device/HPUStream.h"
#include "backend/jit_graph_cache.h"
#include "program.h"

namespace habana {
namespace program {

struct Environment {
  synapse_helpers::hpuStream_t stream;
  std::string op_name;
  std::size_t graph_hash;
  std::size_t graph_key;
  std::size_t graph_index;
};

/*
 * Executor for clustered program.
 */
class Executor {
 public:
  struct Frame {
    torch::jit::Stack stack_;

    void Assign(std::size_t index, at::IValue& value);
  };
  Executor(ClusteredProgramSPtr program, Environment& env);

  /*
   * Run program according to program's schedule.
   * Semantics is similar to HabanaLaunchOpPT::run, inputs should be
   * on the given stack and outputs will be stored there.
   */
  void Run(torch::jit::Stack& stack);

 private:
  void RunCluster(Cluster& cluster);
  void PrepareFrames();
  void PropagateClusterOutputs(Frame& frame, Cluster& cluster);

  void ForwardProgramInputs(torch::jit::Stack& stack);
  void ForwardProgramOutputs(torch::jit::Stack& stack);
  ClusteredProgramSPtr program_;
  Environment env_;

  std::unordered_map<Cluster::Id, Frame> frames_;
};

} // namespace program
} // namespace habana
