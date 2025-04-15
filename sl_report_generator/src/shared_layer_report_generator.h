/**
 * Copyright (c) 2024-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "executor.h"
#include "generated/slrg/registry.h"
#include "utils/shared_structures.h"

#pragma once
namespace slrg {
class ISharedLayerReportGenerator {
 public:
  virtual void register_auto_generated_executors() = 0;
  virtual void register_exceptions() = 0;
  virtual void register_op(
      OperatorDescriptor operator_descriptor,
      SharedLayerExecutor<>* const executor) = 0;
  virtual void validateAll() = 0;
  virtual FinalReport getFinalReport() const = 0;
  virtual ~ISharedLayerReportGenerator() = default;
};

class SharedLayerReportGenerator : public ISharedLayerReportGenerator {
 public:
  SharedLayerReportGenerator() = default;
  virtual ~SharedLayerReportGenerator() = default;
  void register_auto_generated_executors() override {
    slrg::register_auto_generated_executors(this);
  }
  void register_exceptions();
  void register_op(
      OperatorDescriptor operator_descriptor,
      SharedLayerExecutor<>* const executor) override {
    executors.push_back({operator_descriptor, executor});
  }
  void validateAll() override {
    for (const auto& [_, executor] : executors)
      executor->validate();
  }
  FinalReport getFinalReport() const override {
    FinalReport finalReport;
    for (const auto& [operator_descriptor, executor] : executors)
      finalReport[operator_descriptor.op_name].push_back(
          {operator_descriptor, executor->getReport()});
    return finalReport;
  }

 private:
  void register__adaptive_avg_pool2d_exception();
  void register_abs__exception();
  void register_bmm_exception();
  void register_bmm_out_exception();
  void register_channel_shuffle_exception();
  void register_clamp_exception();
  void register_ctc_loss_exception();
  void register_ctc_loss_tensor_exception();
  void register_grid_sample_exception();
  void register_im2col_exception();
  void register_im2col_out_exception();
  void register_index_reduce__exception();
  void register_linear_exception();
  void register_masked_fill_exception();
  void register_masked_scatter_exception();
  void register_max_pool2d_exception();
  void register_max_pool3d_exception();
  void register_mm_exception();
  void register_mm_out_exception();
  void register_multi_margin_loss_exception();
  void register_multi_margin_loss_out_exception();
  void register_multilabel_margin_loss_exception();
  void register_nll_loss_forward_exception();
  void register_nll_loss_forward_output_exception();
  void register_reflection_pad_exception();
  void register_replication_pad_exception();
  void register_scatter_add__exception();
  void register_scatter_exception();
  void register_scatter_out_exception();
  void register_searchsorted_exception();
  void register_max_unpool2d_exception();
  void register_max_unpool2d_out_exception();
  void register_max_unpool3d_exception();
  void register_max_unpool3d_out_exception();
  void register_upsample_exception();
  void register_upsample_out_exception();
  void register_where_exception();
  void register_where_out_exception();
  void register_static_exceptions();
  std::vector<std::pair<OperatorDescriptor, SharedLayerExecutor<>* const>>
      executors;
  // below attributes are required for lifetime management of custom generators
  // and executors
  std::vector<std::unique_ptr<IStackGenerator>> custom_stack_generators;
  std::vector<std::unique_ptr<SharedLayerExecutor<>>> custom_executors;
};

} // namespace slrg
