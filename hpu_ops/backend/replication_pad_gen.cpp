/******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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

#include <cmath>
#include "hpu_ops/common/replication_pad.h"

namespace habana {
OutputMetaDataVector ReplicationPad1DMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  OutputMetaData meta;
  meta.shape = ComputePadOutputShape(stack, pad1D)[0];
  meta.dtype = self.scalar_type();
  return {meta};
}

OutputMetaDataVector ReplicationPad2DMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  OutputMetaData meta;
  meta.shape = ComputePadOutputShape(stack, pad2D)[0];
  meta.dtype = self.scalar_type();
  return {meta};
}

OutputMetaDataVector ReplicationPad3DMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  OutputMetaData meta;
  meta.shape = ComputePadOutputShape(stack, pad3D)[0];
  meta.dtype = self.scalar_type();
  return {meta};
}

std::shared_ptr<void> FillReplicationPad1dFwdParams(
    const at::Stack& stack,
    size_t& size) {
  return FillPadFwdBwdParams(stack, pad1D, size, false);
}

std::shared_ptr<void> FillReplicationPad2dFwdParams(
    const at::Stack& stack,
    size_t& size) {
  return FillPadFwdBwdParams(stack, pad2D, size, false);
}

std::shared_ptr<void> FillReplicationPad3dFwdParams(
    const at::Stack& stack,
    size_t& size) {
  return FillPadFwdBwdParams(stack, pad3D, size, false);
}

} // namespace habana
