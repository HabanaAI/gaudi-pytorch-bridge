/**
* Copyright (c) 2021-2024 Intel Corporation
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

#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "backend/habana_device/hpu_cached_devices.h"
#include "generated/lazy/wrap_kernels_declarations.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"
#include "habana_lazy/lazy_executor.h"

using namespace habana_lazy;
using namespace torch;

class DryRunTest : public habana_lazy_test::LazyTest {};

TEST_F(DryRunTest, Test) {
  int out_features = 4096;
  int in_features = 2048;

  auto in = torch::randn({in_features});
  auto hin = in.to(torch::kHPU);
  auto wt = torch::randn({out_features, in_features}); // ckhw
  auto hwt = wt.to(torch::kHPU);
  auto bias = torch::randn({out_features});
  auto hbias = bias.to(torch::kHPU);

  auto out_hpu = torch::linear(hin, hwt);

  auto& device = habana::HPUDeviceContext::get_device();
  habana_lazy::HbExecutionContext* context =
      habana_lazy::get_device_lazy_execution_context();

  synapse_helpers::MemoryStats stats;
  device.get_device_memory().get_memory_stats(&stats);
  auto peak_bytes_in_use_before = stats.peak_bytes_in_use;
  if (context != nullptr) {
    context->setDryRun(true);
  }
  HbLazyTensor::StepMarker({});
  auto add_hpu = torch::add(out_hpu, hbias);
  HbLazyTensor::StepMarker({});
  device.get_device_memory().get_memory_stats(&stats);
  auto peak_bytes_in_use_after = stats.peak_bytes_in_use;

  if (context != nullptr) {
    context->setDryRun(false);
  }

  EXPECT_EQ(peak_bytes_in_use_before == peak_bytes_in_use_after, true);
}
