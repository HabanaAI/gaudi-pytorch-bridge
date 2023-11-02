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

#include "hpu_ops/masked_batch_gemm.h"

namespace habana {

sizes_vec MaskedBatchGemmOutputShape(const at::Stack& stack) {
  auto a = stack_tensor(stack, 0);
  auto b = stack_tensor(stack, 1);
  bool trans_a = stack[4].toBool();
  bool trans_b = stack[5].toBool();

  std::vector<int64_t> a_shape = a.sizes().vec();
  std::vector<int64_t> b_shape = b.sizes().vec();
  std::vector<int64_t> out_shape{a_shape[0], a_shape[1]};
  int a_dim = 2 + (trans_a ? 1 : 0);
  int b_dim = 2 + (trans_b ? 0 : 1);
  out_shape.push_back(a_shape[a_dim]);
  out_shape.push_back(b_shape[b_dim]);

  return {out_shape};
}

MaskedBatchGemm::MaskedBatchGemm(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "masked_batch_gemm",
          scalar_type,
          {0},
          {},
          {},
          false) {
  SetComputeOutputShapes(MaskedBatchGemmOutputShape);
}

void MaskedBatchGemm::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "MaskedBatchGemm::AddNode");
  auto a = getNextInput<TensorsPair>(stackGetter);
  auto b = getNextInput<TensorsPair>(stackGetter);
  auto mask_a = getNextInput<TensorsPair>(stackGetter);
  auto mask_b = getNextInput<TensorsPair>(stackGetter);
  auto trans_a = getNextInput<bool>(stackGetter);
  auto trans_b = getNextInput<bool>(stackGetter);

  TORCH_CHECK(
      a.pt_t.dim() == 4 && b.pt_t.dim() == 4 && mask_a.pt_t.dim() == 4 &&
          mask_b.pt_t.dim() == 4,
      "All inputs must be 4D, but got: a = ",
      a.pt_t.dim(),
      "D, b = ",
      b.pt_t.dim(),
      "D, mask_a = ",
      mask_a.pt_t.dim(),
      "D, mask_b = ",
      mask_b.pt_t.dim(),
      "D");

  std::vector<synTensor> syn_inputs = {
      a.syn_t, b.syn_t, mask_a.syn_t, mask_b.syn_t};
  synGEMMParams params{trans_a, trans_b};
  auto out_shapes = MaskedBatchGemmOutputShape(stack);

  auto output = OpBackend::BuildNode(
      this,
      graph,
      {GetGuid(),
       syn_inputs,
       {{out_shapes[0], ScalarType(), 0}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(output[0]);
}

} // namespace habana

static const auto& MaskedBatchGemmKernelRegistry = habana::KernelRegistry().add(
    "hpu::masked_batch_gemm",
    KERNEL_FN_GLOBAL(habana::MaskedBatchGemm));
