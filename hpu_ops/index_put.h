/******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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

struct IndexPutEager : OpBackend {
  IndexPutEager(int device_id, c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

struct IndexPutBoolEager : OpBackend {
  IndexPutBoolEager(int device_id, c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

struct IndexPutCompile : OpBackend {
  IndexPutCompile(int device_id, c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

} // namespace habana
