/******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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

struct Int4BaseOp : OpBackend {
  Int4BaseOp(
      int device_id,
      c10::ScalarType scalar_type,
      const std::string& guid);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

#define DEFINE_OP(op)                               \
  struct op : Int4BaseOp {                          \
    op(int device_id, c10::ScalarType scalar_type); \
  };

DEFINE_OP(ConvertFromInt4)
DEFINE_OP(ConvertFromUint4)

OUTMETA_DECL(ConvertFromInt4Meta)

} // namespace habana