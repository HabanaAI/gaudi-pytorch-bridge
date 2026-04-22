/**
 * Copyright (c) 2021-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cmath>
#include "hpu_ops/common/replication_pad.h"

namespace habana {
OutputMetaDataVector ReplicationPad1DMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  TORCH_CHECK_NOT_IMPLEMENTED(
      self.scalar_type() != torch::kBool,
      "\"replication_pad1d\" not implemented for 'Bool'");
  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.shape = ComputePadOutputShape(stack, pad1D)[0];
  meta.dtype = self.scalar_type();
  return metaVec;
}

OutputMetaDataVector ReplicationPad2DMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  TORCH_CHECK_NOT_IMPLEMENTED(
      self.scalar_type() != torch::kBool,
      "\"replication_pad2d\" not implemented for 'Bool'");
  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.shape = ComputePadOutputShape(stack, pad2D)[0];
  meta.dtype = self.scalar_type();
  return metaVec;
}

OutputMetaDataVector ReplicationPad3DMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  TORCH_CHECK_NOT_IMPLEMENTED(
      self.scalar_type() != torch::kBool,
      "\"replication_pad3d\" not implemented for 'Bool'");
  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.shape = ComputePadOutputShape(stack, pad3D)[0];
  meta.dtype = self.scalar_type();
  return metaVec;
}

FillParamsT FillReplicationPad1dFwdParams(const at::Stack& stack) {
  return FillPadFwdBwdParams(stack, pad1D, false);
}

FillParamsT FillReplicationPad2dFwdParams(const at::Stack& stack) {
  return FillPadFwdBwdParams(stack, pad2D, false);
}

FillParamsT FillReplicationPad3dFwdParams(const at::Stack& stack) {
  return FillPadFwdBwdParams(stack, pad3D, false);
}

} // namespace habana
