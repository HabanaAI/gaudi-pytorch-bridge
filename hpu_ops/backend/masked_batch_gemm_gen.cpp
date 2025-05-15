/**
 * Copyright (c) 2023-2025 Intel Corporation
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

#include "generated/backend/masked_batch_gemm.h"
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

sym_sizes_vec masked_batch_gemm_out_shape(
    const std::vector<at::Tensor>& inputs,
    const std::vector<int64_t>& params) {
  HABANA_ASSERT(inputs.size() == 2);
  HABANA_ASSERT(params.size() == 2);
  return MaskedBatchGemmOutputShapeInternal(
      inputs[0].sym_sizes(),
      inputs[1].sym_sizes(),
      static_cast<bool>(params[0]),
      static_cast<bool>(params[1]));
}

REGISTER_CUSTOM_OP_OUTSHAPE_FUN(masked_batch_gemm, masked_batch_gemm_out_shape);

OutputMetaDataVector MaskedBatchGemmMeta(const at::Stack& stack) {
  HABANA_ASSERT(
      HPUDeviceContext::get_device().type() == synDeviceGaudi2,
      "masked_batch_gemm is supported only on Gaudi2.");

  auto a = stack_tensor(stack, 0);
  auto b = stack_tensor(stack, 1);
  auto mask_a = stack_tensor(stack, 2);
  auto mask_b = stack_tensor(stack, 3);
  bool trans_a = stack.at(4).toBool();
  bool trans_b = stack.at(5).toBool();
  auto out_shapes = MaskedBatchGemmOutputShapeInternal(
      a.sizes(), b.sizes(), trans_a, trans_b);

  HABANA_ASSERT(
      a.dim() == 4 && b.dim() == 4 && mask_a.dim() == 4 && mask_b.dim() == 4,
      "All inputs must be 4D, but got: a = ",
      a.dim(),
      "D, b = ",
      b.dim(),
      "D, mask_a = ",
      mask_a.dim(),
      "D, mask_b = ",
      mask_b.dim(),
      "D");

  OutputMetaData meta;
  meta.shape = out_shapes[0];
  meta.dtype = a.scalar_type();
  return {meta};
}

FillParamsT FillMaskedBatchGemmParams(const at::Stack& stack) {
  PARAMS_STUB(synGEMMParams);
  params->transpose_a = stack.at(4).toBool();
  params->transpose_b = stack.at(5).toBool();
  return paramsT;
}

} // namespace habana
