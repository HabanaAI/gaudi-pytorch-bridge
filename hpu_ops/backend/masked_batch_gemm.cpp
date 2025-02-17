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

#include "hpu_ops/masked_batch_gemm.h"
#include "hpu_ops/custom_op_outshape.h"

namespace habana {

template <class DimT>
sizes_vec_template<DimT> MaskedBatchGemmOutputShapeInternal(
    c10::ArrayRef<DimT> a_shape,
    c10::ArrayRef<DimT> b_shape,
    const bool trans_a,
    const bool trans_b) {
  std::vector<DimT> out_shape{a_shape[0], a_shape[1]};
  int a_dim = 2 + (trans_a ? 1 : 0);
  int b_dim = 2 + (trans_b ? 0 : 1);
  out_shape.push_back(a_shape[a_dim]);
  out_shape.push_back(b_shape[b_dim]);

  return {out_shape};
}

sizes_vec MaskedBatchGemmOutputShape(const at::Stack& stack) {
  auto a = stack_tensor(stack, 0);
  auto b = stack_tensor(stack, 1);
  bool trans_a = stack[4].toBool();
  bool trans_b = stack[5].toBool();

  return MaskedBatchGemmOutputShapeInternal(
      a.sizes(), b.sizes(), trans_a, trans_b);
}

sym_sizes_vec masked_batch_gemm_out_shape(
    const std::vector<at::Tensor>& inputs,
    const std::vector<int64_t>& params) {
  TORCH_CHECK(inputs.size() == 2);
  TORCH_CHECK(params.size() == 2);
  return MaskedBatchGemmOutputShapeInternal(
      inputs[0].sym_sizes(),
      inputs[1].sym_sizes(),
      static_cast<bool>(params[0]),
      static_cast<bool>(params[1]));
}

REGISTER_CUSTOM_OP_OUTSHAPE_FUN(masked_batch_gemm, masked_batch_gemm_out_shape);

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
  StackGetter stackGetter(this, stack, "MaskedBatchGemm::AddNode");
  auto a = stackGetter.getNextInput<TensorsPair>();
  auto b = stackGetter.getNextInput<TensorsPair>();
  auto mask_a = stackGetter.getNextInput<TensorsPair>();
  auto mask_b = stackGetter.getNextInput<TensorsPair>();
  auto trans_a = stackGetter.getNextInput<bool>();
  auto trans_b = stackGetter.getNextInput<bool>();

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
