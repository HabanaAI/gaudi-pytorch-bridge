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

Scalar _local_scalar_dense_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;
  Scalar r = habana_helpers::_local_scalar_dense_internal(self);
  PT_KERNEL_END;

  return r;
}
