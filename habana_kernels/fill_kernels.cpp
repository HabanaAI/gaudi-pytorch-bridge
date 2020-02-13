/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"

using namespace torch;

Tensor& fill_hpu(Tensor& self, Scalar value) {
  LOG_FUNC_BEGIN;
  TORCH_WARN("fill_hpu executes CPU kernel internally");

  auto hpu = self.device();
  auto self_ = self.to(DeviceType::CPU);

  auto result = at::native::fill_(self_, value);
  self = self_.to(hpu);
  LOG_FUNC_END;
  return self;
}

static auto registry = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema(
            "aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)")
        .impl_unboxedOnlyKernel<decltype(fill_hpu), &fill_hpu>(
            TensorTypeId::HABANATensorId)
        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
