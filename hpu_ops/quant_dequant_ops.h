/******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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

#pragma once

#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/op_backend.h"

namespace habana {

struct QuantizePerTensor : OpBackend {
  QuantizePerTensor(int device_id, c10::ScalarType scalar_type);
};

FILL_PARAMS_DECL(FillQuantizePerTensorParams)
OUTMETA_DECL(QuantizePerTensorMeta)

struct DequantizePerTensor : OpBackend {
  DequantizePerTensor(int device_id, c10::ScalarType scalar_type);
  void CustomHandler(synapse_helpers::graph&, at::Stack&) override;
};

FILL_PARAMS_DECL(FillDequantizePerTensorParams)

struct QuantizePerChannel : OpBackend {
  QuantizePerChannel(int device_id, c10::ScalarType scalar_type);
};

FILL_PARAMS_DECL(FillQuantizePerChannelParams)
OUTMETA_DECL(QuantizePerChannelMeta)

struct DequantizePerChannel : OpBackend {
  DequantizePerChannel(int device_id, c10::ScalarType scalar_type);
  void CustomHandler(synapse_helpers::graph&, at::Stack&) override;
};

FILL_PARAMS_DECL(FillDequantizePerChannelParams)

} // namespace habana
