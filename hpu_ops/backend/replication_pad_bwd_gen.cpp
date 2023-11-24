/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/replication_pad1d_backward.h"
#include "generated/backend/replication_pad3d_backward.h"

namespace habana {

enum ReplicationPad { Pad1d = 1, Pad2d, Pad3d };

std::shared_ptr<void> FillPadBwdParams(
    const at::Stack& stack,
    ReplicationPad pad,
    size_t& size) {
  PARAMS_STUB(ns_PadKernelEx::Params);
  auto self = stack.at(1).toTensor();
  auto padding = stack.at(2).toIntVector();
  params->mode = PadMode_t::PAD_MODE_EDGE;

  if (padding.size() == 1) {
    for (auto i = 0; i < self.dim() - 1; i++) {
      params->pads[i] = padding.at(0);
      params->pads[i + self.dim()] = padding.at(0);
      if ((i == 0 && pad == Pad1d) || (i == 1 && pad == Pad2d) ||
          (i == 2 && pad == Pad3d)) {
        break;
      }
    }
  } else {
    for (auto i = 0; i < self.dim() - 1; i += 1) {
      params->pads[i] = padding.at(i * 2);
      params->pads[i + self.dim()] = padding.at(i * 2 + 1);
      if ((i == 0 && pad == Pad1d) || (i == 1 && pad == Pad2d) ||
          (i == 2 && pad == Pad3d)) {
        break;
      }
    }
  }

  return params;
}

OutputMetaDataVector ReplicationPadBwdMeta(const at::Stack& stack) {
  auto self = stack.at(1).toTensor();
  OutputMetaData meta;
  meta.shape = self.sizes().vec();
  meta.dtype = self.scalar_type();
  return {meta};
}

std::shared_ptr<void> FillReplicationPad1dBwdParams(
    const at::Stack& stack,
    size_t& size) {
  return FillPadBwdParams(stack, Pad1d, size);
}

std::shared_ptr<void> FillReplicationPad2dBwdParams(
    const at::Stack& stack,
    size_t& size) {
  return FillPadBwdParams(stack, Pad2d, size);
}

std::shared_ptr<void> FillReplicationPad3dBwdParams(
    const at::Stack& stack,
    size_t& size) {
  return FillPadBwdParams(stack, Pad3d, size);
}

void ReplicationPadBwdOp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  // This AddNode function was added, because for backward pass
  // we are having 2 tensor arguments(Grad-In & Self),
  // but the kernel expects Grad-In tensor alone.
  std::vector<synapse_helpers::tensor> pad_bwd_out;
  auto meta = ReplicationPadBwdMeta(stack)[0];
  size_t size = 0;
  auto params = FillParams(stack, size);

  pad_bwd_out = BuildOp(
      graph,
      get_guid_with_precision("pad_bwd", ScalarType()),
      {syn_in(0)},
      {{meta.shape, meta.dtype, 0}},
      params.get(),
      size);

  syn_out(0) = std::move(pad_bwd_out[0]);
}
} // namespace habana
