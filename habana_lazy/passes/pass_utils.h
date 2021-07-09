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

#include <torch/csrc/jit/ir/ir.h>
#include "habana_kernels/habana_operator.h"

namespace habana_lazy {

struct habanaTensorLayoutInfo {
  habana::LayoutFormat layout;
  habana::LayoutFormat layout_at_graph_entry;
};

at::IntArrayRef getDimsForLayout(
    habana::LayoutFormat channel_order,
    habana::LayoutFormat current_order);

class WeightIdentificationPass {
 public:
  WeightIdentificationPass() {}

  void markWeightTensors(
      std::shared_ptr<torch::jit::Graph>& graph,
      bool mark_out_varients = true);

  std::unordered_set<const torch::jit::Value*> getWeightTensors() const {
    return weightTensors;
  }

  const std::unordered_map<std::string, size_t> getConvKernelInWeights() const {
    return kernelWeightIdx;
  }

  const std::unordered_map<std::string, size_t> getConvKernelOutWeights()
      const {
    return kernelOutWeightIdx;
  }

  const std::unordered_map<std::string, std::vector<size_t>>
  getCustomOptimizerWeights() const {
    return customOptimizerWeightIdx;
  }

  void weightMarker(const torch::jit::Value* weight_value) {
    weightTensors.insert(weight_value);
  }

 private:
  bool is_mark_out_varients = true;
  std::unordered_set<const torch::jit::Value*> weightTensors;
  const std::unordered_map<std::string, size_t> kernelWeightIdx = {
      {"aten::convolution_overrideable", 1},
      {"aten::convolution_backward_overrideable", 2}};
  const std::unordered_map<std::string, size_t> kernelOutWeightIdx = {
      {"aten::convolution_backward_overrideable", 1}};
  const std::unordered_map<std::string, size_t> kernelOutVariantIdx = {
      {"hpu::habana_d2d_memcpy_other", 1}};

  const std::unordered_map<std::string, std::vector<size_t>>
      customOptimizerWeightIdx = {
          {"hpu::habanaOptimizerFusedSGDMomentum", {0, 1, 2}},
          {"hpu::habanaOptimizerFusedAdagrad", {0, 1, 2}},
          {"hpu::habanaOptimizerAdamW", {0, 1}},
          {"hpu::habanaOptimizerLambPhase1", {0, 1}},
          {"hpu::habanaOptimizerLambPhase1", {0, 1}}};

  void markWeights(const torch::jit::Value* value);
  bool isTensor(const torch::jit::Value* value);
  void markInputs(const torch::jit::Value* value);
  void markOutputs(const torch::jit::Value* value);
};
}; // namespace habana_lazy
