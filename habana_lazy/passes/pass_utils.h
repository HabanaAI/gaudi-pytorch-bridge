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
#include "backend/habana_operator.h"

namespace habana_lazy {

struct habanaTensorLayoutInfo {
  habana::LayoutFormat layout;
  habana::LayoutFormat layout_at_graph_entry;
};

int64_t getLayoutDim5d(habana::LayoutFormat layout, int64_t dim);

int64_t getLayoutDim(habana::LayoutFormat layout, int64_t dim);

at::IntArrayRef getDimsForLayout5d(
    habana::LayoutFormat channel_order,
    habana::LayoutFormat current_order);

at::IntArrayRef getDimsForLayout(
    habana::LayoutFormat channel_order,
    habana::LayoutFormat current_order);

bool isRetunrOut(
    std::shared_ptr<torch::jit::Graph>& graph,
    const torch::jit::Value* value_out);

bool is_4d_5d_value(const torch::jit::Value* value_in);

class WeightIdentificationPass {
 public:
  WeightIdentificationPass() {}

  void markWeightTensors(
      std::shared_ptr<torch::jit::Graph>& graph,
      bool mark_out_varients = true);

  void markWeightInTensors(std::shared_ptr<torch::jit::Graph>& graph);

  std::unordered_set<const torch::jit::Value*> getWeightTensors() const {
    return weightTensors;
  }

  bool isMarkedAsweight(const torch::jit::Value* value_in) {
    if (weightTensors.find(value_in) != weightTensors.end()) {
      if (*value_in->type()->cast<torch::jit::TensorType>()->dim() == 4 ||
          *value_in->type()->cast<torch::jit::TensorType>()->dim() == 5) {
        return true;
      }
    }
    return false;
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
    markWeights(weight_value);
  }

 private:
  bool is_mark_out_varients = true;
  std::unordered_set<const torch::jit::Value*> weightTensors;
  const std::unordered_map<std::string, size_t> kernelWeightIdx = {
      {"aten::convolution_overrideable", 1},
      {"aten::convolution_backward_overrideable", 2},
      {"hpu::permuted_weight", 0},
      {"hpu::permuted_weight_restride", 0}};
  const std::unordered_map<std::string, size_t> kernelOutWeightIdx = {
      {"aten::convolution_backward_overrideable", 1}};
  const std::unordered_map<std::string, size_t> kernelOutVariantIdx = {
      {"hpu::habana_d2d_memcpy_other", 1},
      {"hpu::mul_out", 0}};
  const std::set<std::string> StridedKernels = {
      "hpu::as_strided_lazy",
      "hpu::strided_insert",
      "hpu::strided_insert_cl",
      "hpu::strided_insert_ds",
      "hpu::strided_insert_cl_ds",
      "hpu::strided_view",
      "hpu::strided_view_cl",
      "hpu::strided_view_ds",
      "hpu::strided_view_cl_ds"};

  const std::unordered_map<std::string, std::vector<size_t>>
      customOptimizerWeightIdx = {
          {"hpu::habanaOptimizerFusedSGDMomentum", {0, 1, 2}},
          {"hpu::habanaOptimizerFusedAdagrad", {0, 1, 2}},
          {"hpu::habanaOptimizerAdamW", {0, 1, 2, 3}},
          {"hpu::habanaOptimizerLambPhase1", {0, 1}},
          {"hpu::habanaOptimizerLambPhase1", {0, 1}}};

  void markWeights(const torch::jit::Value* value);
  bool isTensor(const torch::jit::Value* value);
  void markInputs(const torch::jit::Value* value);
  void markOutputs(const torch::jit::Value* value);
  void markInOutputs(const torch::jit::Value* value);
};
}; // namespace habana_lazy
