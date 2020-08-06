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

#include <ATen/Tensor.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/interpreter.h>

class HbCompiler {
 public:
  explicit HbCompiler(const torch::jit::Node* node, bool debug);
  void run(torch::jit::Stack& stack);

 private:
  std::shared_ptr<torch::jit::Graph> subgraph_;
  bool debug_;
};