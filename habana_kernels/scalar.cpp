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

#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/logging.h"

using namespace torch;

namespace at {
namespace native {

Scalar _local_scalar_dense_hpu(const Tensor& self) {
  LOG_FUNC_BEGIN;
  Scalar r;
  // Note: this macro expands to more types than HPU supports, but this is not
  // an issue
  // Note: this kernel intentionally doesn't check if numel == 1, dunno why,
  // it's just pytorch
  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "_local_scalar_dense_hpu",
      [&] {
        scalar_t value;
        synapse_helpers::HPURegistrar::get_device(self.device().index())
            .copy_data_to_host(
                reinterpret_cast<synapse_helpers::device_ptr>(self.data_ptr()),
                &value,
                self.nbytes());
        r = Scalar(value);
      });
  LOG_FUNC_END;
  return r;
}

} // namespace native
} // namespace at

static auto registry = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema("aten::_local_scalar_dense(Tensor self) -> Scalar")
        .impl_unboxedOnlyKernel<
            decltype(at::native::_local_scalar_dense_hpu),
            &at::native::_local_scalar_dense_hpu>(TensorTypeId::HABANATensorId)
        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));