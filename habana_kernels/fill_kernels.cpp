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

template <typename T, typename U>
void synapse_fill(const Tensor& output, const T val, U memset_function) {
  auto& device =
      synapse_helpers::HPURegistrar::get_device(output.device().index());

  TORCH_HABANA_CHECK(memset_function(
      reinterpret_cast<uint64_t>(output.data_ptr()),
      val,
      output.numel(),
      device.get_host_to_device_stream()));

  TORCH_HABANA_CHECK(
      synStreamSynchronize(device.get_host_to_device_stream()),
      "Stream synchronization failed");
}

Tensor& fill_hpu_(Tensor& self, Scalar value) {
  LOG_FUNC_BEGIN;
  auto dtype = habana_helpers::scalar_type(value);
  if (self.scalar_type() != dtype)
    TORCH_WARN(
        "Self tensor's type: ",
        self.scalar_type(),
        ". Value type: ",
        dtype,
        "\nfill_hpu will use cast provided value");
  TORCH_CHECK(dtype != c10::ScalarType::Bool);

  switch (self.element_size()) {
    case 1: {
      TORCH_CHECK(value.isIntegral(false));
      auto memset_val = value.to<unsigned char>();
      synapse_fill(self, memset_val, synMemsetD8Async);
    } break;
    case 2: {
      TORCH_CHECK(value.isFloatingPoint() || value.isIntegral(false));
      if (value.isFloatingPoint()) {
        TORCH_CHECK(
            0, "HPU is unable to differentatiate between fp16 and bf16");
      } else {
        auto memset_val = value.to<uint16_t>();
        synapse_fill(self, memset_val, synMemsetD16Async);
      }
    } break;
    case 4: {
      uint32_t memset_val;
      TORCH_CHECK(value.isFloatingPoint() || value.isIntegral(false));
      if (value.isIntegral(false) && self.scalar_type() == dtype) {
        memset_val = value.to<uint32_t>();
      } else {
        auto float_val = value.to<float>();
        memcpy(&memset_val, &float_val, sizeof(float_val));
      }
      synapse_fill(self, memset_val, synMemsetD32Async);
    } break;
    default:
      TORCH_CHECK(
          self.element_size() < 4,
          "HPU doesn't support data types bigger than 4 bytes. Unsupported type: ",
          self.scalar_type());
  }
  LOG_FUNC_END;
  return self;
}

static auto registry = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema(
            "aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)")
        .impl_unboxedOnlyKernel<decltype(fill_hpu_), &fill_hpu_>(
            TensorTypeId::HABANATensorId)
        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
