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
  PT_KERNEL_BEGIN;
  Scalar r;

  // Note: this macro expands to more types than HPU supports, but this is not
  // an issue
  // Note: Pytorch uses this function to check a specific emement of a tensor
  // eg. embedding_bag validates the first value offsets to be 0 using this
  // function

  if (at::ScalarType::Bool == self.scalar_type()) {
    AT_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Bool,
        self.scalar_type(),
        "_local_scalar_dense_hpu",
        [&] {
          scalar_t val;
          habana_helpers::copy_scalar_to_host(self, &val, sizeof(self.dtype()));
          r = Scalar(val);
        });
  } else {
    AT_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "_local_scalar_dense_hpu",
        [&] {
          scalar_t val;
          habana_helpers::copy_scalar_to_host(self, &val, sizeof(self.dtype()));
          r = Scalar(val);
        });
  }
  PT_KERNEL_END;

  return r;
}

} // namespace native
} // namespace at

static auto registry = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema("aten::_local_scalar_dense(Tensor self) -> Scalar")
        .impl_unboxedOnlyKernel<
            decltype(at::native::_local_scalar_dense_hpu),
            &at::native::_local_scalar_dense_hpu>(DispatchKey::HABANATensorId)
        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
