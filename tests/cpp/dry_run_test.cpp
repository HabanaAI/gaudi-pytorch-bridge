/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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

  auto& device = habana::HPURegistrar::get_device();
  habana_lazy::HbExecutionContext* context =
      habana_lazy::get_device_lazy_execution_context(device.id());

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
