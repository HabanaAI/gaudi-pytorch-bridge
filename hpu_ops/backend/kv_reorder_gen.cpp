/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include "hpu_ops/kv_reorder.h"

namespace habana {

struct KvReorder : KvReorderCommon {
  KvReorder(int device_id, c10::ScalarType scalar_type)
      : KvReorderCommon(
            device_id,
            "kv_reorder",
            scalar_type,
            {0},
            {},
            {},
            false) {}
};

struct KvReorder_ : KvReorderCommon {
  KvReorder_(int device_id, c10::ScalarType scalar_type)
      : KvReorderCommon(
            device_id,
            "kv_reorder_",
            scalar_type,
            {},
            {0},
            {},
            false) {}
};

void KvReorderCommon::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 4, "KvReorder must have 4 input arguments");

  StackGetter stackGetter(stack, "KvReorder::AddNode");
  auto self = getNextInput<TensorsPair>(stackGetter);
  auto start = getNextInput<TensorsPair>(stackGetter);
  auto end = getNextInput<TensorsPair>(stackGetter);
  auto beam_idx = getNextInput<TensorsPair>(stackGetter);

  TORCH_CHECK(
      start.pt_t.dtype() == c10::ScalarType::Int,
      "Start tensor must be of type Int32");
  TORCH_CHECK(
      end.pt_t.dtype() == c10::ScalarType::Int,
      "End tensor must be of type Int32");
  TORCH_CHECK(
      beam_idx.pt_t.dtype() == c10::ScalarType::Byte,
      "Beam_idx tensor must be of type UInt8");
  TORCH_CHECK(start.pt_t.dim() == 1, "Start tensor must have dimensions 1");
  TORCH_CHECK(end.pt_t.dim() == 1, "End tensor must have dimensions 1");
  TORCH_CHECK(
      beam_idx.pt_t.dim() == 1, "Beam_idx tensor must have dimensions 1");

  auto shape = self.pt_t.sizes().vec();
  auto selective_gather = BuildNode(
      this,
      graph,
      {get_guid_with_precision("selective_gather_fwd", ScalarType()),
       {self.syn_t, start.syn_t, end.syn_t, beam_idx.syn_t},
       {{shape, ScalarType(), 0}}});

  syn_out(0) = std::move(selective_gather[0]);
}

} // namespace habana

static auto& KvReorderKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::kv_reorder_", KERNEL_FN(KvReorder_))
        .add("hpu::kv_reorder", KERNEL_FN(KvReorder));