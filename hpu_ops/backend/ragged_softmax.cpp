/******************************************************************************
 * Copyright (C) 2023 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 *******************************************************************************/

#include "generated/backend/_softmax.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

struct RaggedSoftmax : OpBackend {
  RaggedSoftmax(int device_id, c10::ScalarType scalar_type)
      : OpBackend(device_id, "softmax_fwd_", scalar_type, {0}, {}, {}, false) {
    SetFillParams(FillSoftmaxForwardParams);
  }
};

} // namespace habana

static auto& KernelRegistry = habana::KernelRegistry().add(
    "hpu::ragged_softmax",
    KERNEL_FN(RaggedSoftmax));