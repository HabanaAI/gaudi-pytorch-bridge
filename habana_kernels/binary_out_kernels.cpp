/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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
#include <ATen/ExpandUtils.h>
#include <torch/script.h>
#include <memory>

#include "backend/helpers/create_tensor.h"
#include "backend/helpers/graph.h"
#include "backend/helpers/tensor_utils.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "hpu_ops/op_backend.h"

// Using autogen's OpBackend for mul_out as it has fixes for type promotion
namespace habana {
using namespace torch;
struct mul_out : OpBackend {
  mul_out(int device_id, c10::ScalarType scalar_type)
      : OpBackend(device_id, MULT_GUID, scalar_type, {}, {}, {}, true) {}
};
} // namespace habana

static auto& BinaryOutKernelsKernelRegistry =
    habana::KernelRegistry().add("aten::mul.out", KERNEL_FN(mul_out));
