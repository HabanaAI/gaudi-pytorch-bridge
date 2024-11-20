/**
* Copyright (c) 2021-2024 Intel Corporation
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
