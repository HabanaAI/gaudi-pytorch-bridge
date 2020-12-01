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

  // Note:
  // 1. This macro expands to more types than HPU supports,
  //   but that should not be an issue issue.
  // 2. Pytorch uses this function to check a specific emement of a tensor
  //   eg. embedding_bag validates the first value offsets to be 0 using this
  //   function
  // 3. A TORCH_CHECK is added to ensure that the size at source
  //   matches with the destination.

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "_local_scalar_dense_hpu",
      [&] {
        scalar_t val;
        TORCH_CHECK(
            elementSize(self.scalar_type()) == sizeof(val),
            " source and destination size mismatch");
        habana_helpers::copy_scalar_to_host(self, &val, sizeof(val));
        r = Scalar(val);
      });

  PT_KERNEL_END;

  return r;
}

} // namespace native
} // namespace at
