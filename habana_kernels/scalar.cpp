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
#include "habana_helpers/tensor_utils.h"

using namespace torch;

namespace at {
namespace native {

Scalar _local_scalar_dense_hpu(const Tensor& self) {
  LOG_FUNC_BEGIN;
  Scalar r;

  // Note: this macro expands to more types than HPU supports, but this is not
  // an issue
  // Note: Pytorch uses this function to check a specific emement of a tensor
  // eg. embedding_bag validates the first value offsets using this function
  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "_local_scalar_dense_hpu",
      [&] {
        habana_helpers::copy_data_to_host(self, &r, sizeof(self.dtype()));
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
