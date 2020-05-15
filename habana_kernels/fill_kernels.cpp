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
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"

using namespace torch;

template <typename T>
void synapse_fill(const Tensor& output, const T val) {
  // Using below approach of filling a buffer on HOST and then copying
  // to Device memory instead of doing a synMemSetD[]Async due to SW-11757
  // TODO revert to synMemSet once SW-11757 is resolved
  auto size = output.numel() * output.element_size();
  std::vector<T> buffer(size, val);

  habana_helpers::copy_data_to_device(buffer.data(), output, size);
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
      synapse_fill(self, memset_val);
    } break;
    case 2: {
      TORCH_CHECK(value.isFloatingPoint() || value.isIntegral(false));
      if (value.isFloatingPoint()) {
        TORCH_CHECK(
            0, "HPU is unable to differentatiate between fp16 and bf16");
      } else {
        auto memset_val = value.to<uint16_t>();
        synapse_fill(self, memset_val);
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
      synapse_fill(self, memset_val);
    } break;
    case 8: {
      // Even though HPU doesnt support long/double. Intermediate tensors in
      // embedding_bag used by PyT needs this fill functionality
      if (value.isIntegral(true)) {
        uint64_t memset_val = value.to<long>();
        synapse_fill(self, memset_val);
      } else {
        // double
        double memset_val = value.to<double>();
        synapse_fill(self, memset_val);
      }
    } break;
    default:
      TORCH_WARN("Unsupported data type used in fill");
  }
  LOG_FUNC_END;
  return self;
}

/** @brief Function implementing torch.Tensor.masked_fill_(mask, value)
 * @param self: (fp32/bf16, 1-4D) Input tensor
 * @param mask: (BoolTensor) the boolean mask
 * @param value: (floatTensor, 0D) the value to fill with
 */
Tensor& masked_fill_hpu_(
    Tensor& self,
    const Tensor& mask,
    const Tensor& value) {
  LOG_FUNC_BEGIN;

  TORCH_CHECK(
      value.dim() == 0, "value supports only 0D tensor to match CPU behavior");

  auto mask_expand = mask;
  if (self.sizes() != mask.sizes()) {
    // this explicit broadcast can be removed when
    // binary kernels start supporting broadcase
    mask_expand = mask.expand(self.sizes());
  }

  TORCH_CHECK(
      self.sizes() == mask_expand.sizes(),
      "input & mask tensor shapes not matching");

  // mask (datatype "bool") needs to be casted because TPC kernels support
  // fp32/bf16 only
  auto new_mask = habana_helpers::hpu_cast_tensor(mask_expand, self.dtype());
  // create a inverted mask
  auto zero_tensor = at::zeros_like(new_mask, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto inv_mask = habana_helpers::hpu_cast_tensor(
      at::eq(new_mask, zero_tensor), self.dtype());

  // broadcast value to same shape as input tensor
  // this explicit broadcast can be removed when
  // binary kernels start supporting broadcase
  auto value_expand = value.expand(self.sizes());

  // mask_fill computation
  self.mul_(inv_mask);
  self.add_(new_mask * value_expand);

  LOG_FUNC_END;
  return self;
}

/** @brief Function implementing torch.Tensor.masked_fill_(mask, value)
 * @param self: (fp32/bf16, 1-4D) Input tensor
 * @param mask: (BoolTensor) the boolean mask
 * @param value: (float) the value to fill with
 */
Tensor& masked_fill_scalar_hpu_(
    Tensor& self,
    const Tensor& mask,
    Scalar value) {
  // convert scalar fill value to device tensor
  auto value_tensor = habana_helpers::scalar_to_device_tensor(
      value.to<float>(), self.options(), 0);

  return masked_fill_hpu_(self, mask, value_tensor);
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(fill_hpu_), &fill_hpu_>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::masked_fill_.Tensor(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(masked_fill_hpu_),
                    &masked_fill_hpu_>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(masked_fill_scalar_hpu_),
                    &masked_fill_scalar_hpu_>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
