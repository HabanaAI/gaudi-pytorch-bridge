/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include "local_scalar_dense.h"
#include <ATen/Dispatch.h>
#include <ATen/core/TensorBody.h>
#include <pybind11/pybind11.h>
#include "common/utils.h"
#include "habana_eager/eager_context.h"
#include "habana_eager/helpers.h"
#include "habana_helpers/frontend_utils.h"

namespace habana {
namespace eager {
at::Scalar _local_scalar_dense_hpu(const at::Tensor& self) {
  c10::Scalar r;
  // To Do - join pending not required here once copy d2h
  // also comes through pipeline SW-126657
  habana::eager::JoinPendingPipelineThreads();
  // Note:
  // 1. This macro expands to more types than HPU supports,
  //   but that should not be an issue issue.
  // 2. Pytorch uses this function to check a specific emement of a tensor
  //   eg. embedding_bag validates the first value offsets to be 0 using this
  //   function
  // 3. A TORCH_CHECK is added to ensure that the size at source
  //   matches with the destination.

  gil_scoped_release_if_held release;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
// Due to runtime check cases detected by compiler won't appear.
// (clang does not have this check at all)
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Warray-bounds"

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self.scalar_type(),
      "_local_scalar_dense",
      [&] {
        scalar_t val;
        TORCH_CHECK(
            elementSize(self.scalar_type()) == sizeof(val),
            " source and destination size mismatch");
        habana_helpers::copy_scalar_to_host(self, &val, sizeof(val));
        // copy_from_ operator is doing implicit down/upcasting
        // for Long and Double. _local_scalar_dense_hpu needs to preserve it
        if (self.scalar_type() == c10::ScalarType::Long &&
            !common::IsInt64Supported()) {
          val = *reinterpret_cast<int32_t*>(&val);
        } else if (self.scalar_type() == c10::ScalarType::Double) {
          val = *reinterpret_cast<float*>(&val);
        }
        r = c10::Scalar(val);
      });

#pragma GCC diagnostic pop
  return r;
}
} // namespace eager
} // namespace habana
