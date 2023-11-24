/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <cmath>
#include "generated/backend/reflection_pad1d.h"
#include "generated/backend/reflection_pad2d.h"
#include "generated/backend/reflection_pad3d.h"

namespace habana {

enum PadType : int64_t { pad1D = 0, pad2D, pad3D };

sizes_vec ComputeOutputShape(const at::Stack& stack, PadType padType) {
  auto self = stack.at(0).toTensor();
  auto padding = stack.at(1).toIntVector();
  std::vector<int64_t> outputsize = self.sizes().vec();
  TORCH_CHECK(
      padding.size() == 1 || padding.size() % 2 == 0,
      "Padding length must be divisible by 2");
  TORCH_CHECK(
      floor(padding.size() / 2) <= self.dim(), "Padding length too large");
  TORCH_CHECK(
      (padding.size() == 1 || padding.size() == 2 || padding.size() == 4 ||
       padding.size() == 6) &&
          (self.dim() == 2 || self.dim() == 3 || self.dim() == 4 ||
           self.dim() == 5),
      "Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now");

  if (padding.size() == 1) {
    for (auto i = 0; i < self.dim() - 1; i++) {
      outputsize.rbegin()[i] += 2 * padding.at(0);
      if (i == padType) {
        break;
      }
    }
  } else {
    for (auto i = 0; i < self.dim() - 1; i++) {
      outputsize.rbegin()[i] += padding.at(i * 2) + padding.at(i * 2 + 1);
      if (i == padType) {
        break;
      }
    }
  }

  return {outputsize};
}

std::shared_ptr<void> FillPadParams(
    const at::Stack& stack,
    PadType padType,
    size_t& size) {
  PARAMS_STUB(ns_PadKernelEx::Params);
  auto self = stack.at(0).toTensor();
  auto padding = stack.at(1).toIntVector();
  params->mode = PadMode_t::PAD_MODE_EDGE;

  if (padding.size() == 1) {
    for (auto i = 0; i < self.dim() - 1; i++) {
      params->pads[i] = padding.at(0);
      params->pads[i + self.dim()] = padding.at(0);
      if (i == padType) {
        break;
      }
    }
  } else {
    for (auto i = 0; i < self.dim() - 1; i += 1) {
      params->pads[i] = padding.at(i * 2);
      params->pads[i + self.dim()] = padding.at(i * 2 + 1);
      if (i == padType) {
        break;
      }
    }
  }

  return params;
}

OutputMetaDataVector ReplicationPad1DMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  OutputMetaData meta;
  meta.shape = ComputeOutputShape(stack, pad1D)[0];
  meta.dtype = self.scalar_type();
  return {meta};
}

OutputMetaDataVector ReplicationPad2DMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  OutputMetaData meta;
  meta.shape = ComputeOutputShape(stack, pad2D)[0];
  meta.dtype = self.scalar_type();
  return {meta};
}

OutputMetaDataVector ReplicationPad3DMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  OutputMetaData meta;
  meta.shape = ComputeOutputShape(stack, pad3D)[0];
  meta.dtype = self.scalar_type();
  return {meta};
}

std::shared_ptr<void> FillReplicationPad1dParams(
    const at::Stack& stack,
    size_t& size) {
  return FillPadParams(stack, pad1D, size);
}

std::shared_ptr<void> FillReplicationPad2dParams(
    const at::Stack& stack,
    size_t& size) {
  return FillPadParams(stack, pad2D, size);
}

std::shared_ptr<void> FillReplicationPad3dParams(
    const at::Stack& stack,
    size_t& size) {
  return FillPadParams(stack, pad3D, size);
}

} // namespace habana
