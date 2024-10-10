/*******************************************************************************
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
#include "hpu_ops/op_backend.h"
#include "include/habanalabs/hpu_custom_op_pt2.h"

namespace habana {

// A class that represents user's torch custom op
// All user's torch ops with a single user tpc kernel will pass here
// when lowered to synapse graph.
class UserCustomOpBackend : public OpBackend {
 public:
  UserCustomOpBackend(
      int device_id,
      const habana::custom_op::UserCustomOpDescriptor& desc)
      : OpBackend(
            device_id,
            desc.getGuid(),
            c10::ScalarType::Undefined, // This kernel will be called without
                                        // dtype suffix.
            {0},
            {},
            {},
            false) {
    SetOutputMetaFn(desc.getOutputMetaFn());
    SetFillParams(desc.getFillParamsFn());
  }
};

} // namespace habana
