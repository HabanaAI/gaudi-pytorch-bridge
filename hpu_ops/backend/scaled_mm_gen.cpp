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

#include "generated/backend/_scaled_mm.h"

namespace sh = synapse_helpers;

namespace habana {

OutputMetaDataVector ScaledMmMeta(const at::Stack& stack) {
  OutputMetaData output_meta;
  const auto& mat1 = stack[0].toTensor();
  const auto mat2_shape = stack[1].toTensor().sizes();
  output_meta.shape = {mat1.sizes()[0], mat2_shape[1]};
  output_meta.dtype =
      stack[3].toOptional<c10::ScalarType>().value_or(mat1.scalar_type());

  OutputMetaData amax_meta{at::ScalarType::Float, {}};
  return {output_meta, amax_meta};
}

SharedMetaDataVector ScaledMmSharedMeta(const at::Stack& stack) {
  const auto mat1_dtype = stack_tensor(stack, 0).scalar_type();
  const auto mat2_dtype = stack_tensor(stack, 1).scalar_type();
  const auto bias = stack[2].toOptional<at::Tensor>();
  const auto out_dtype =
      stack[3].toOptional<c10::ScalarType>().value_or(mat1_dtype);
  const auto scale_a = stack[4].toOptional<at::Tensor>();
  const auto scale_b = stack[5].toOptional<at::Tensor>();
  const auto scale_result = stack[6].toOptional<at::Tensor>();

  const auto optional_meta = createOptionalNotPresentSharedMetaTensor();
  SharedMetaTensor scale_a_meta = scale_a
      ? SharedMetaTensor(scale_a->dim(), scale_a->scalar_type())
      : optional_meta;
  SharedMetaTensor scale_b_meta = scale_b
      ? SharedMetaTensor(scale_b->dim(), scale_b->scalar_type())
      : optional_meta;
  SharedMetaTensor scale_result_meta = scale_result
      ? SharedMetaTensor(scale_result->dim(), scale_result->scalar_type())
      : optional_meta;

  SharedMetaData meta_gemm{"fp8_gemm"};
  meta_gemm.inputs_data = {
      {2, mat1_dtype},
      {2, mat2_dtype},
      scale_a_meta,
      scale_b_meta,
      optional_meta,
      scale_result_meta};
  meta_gemm.outputs_data = {{2, out_dtype}};

  if (out_dtype != at::ScalarType::Float8_e5m2 and
      out_dtype != at::ScalarType::Float8_e4m3fn) {
    return {meta_gemm};
  }

  SharedMetaData meta_abs{"abs_fwd"};
  meta_abs.inputs_data = {{2, at::ScalarType::Float}};
  meta_abs.outputs_data = meta_abs.inputs_data;

  SharedMetaData meta_reduce{"reduce_max_multi_dim_fwd"};
  meta_reduce.inputs_data = meta_abs.inputs_data;
  meta_reduce.outputs_data = {{1, at::ScalarType::Float}};

  return {meta_gemm, meta_abs, meta_reduce};
}

void ScaledMm::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(stack, "ScaledMm::AddNode");
  auto mat1 = getNextInput<TensorsPair>(stackGetter);
  auto mat2 = getNextInput<TensorsPair>(stackGetter);
  auto bias = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto out_dtype = getNextInput<c10::optional<c10::ScalarType>>(stackGetter)
                       .value_or(mat1.pt_t.scalar_type());
  auto scale_a = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto scale_b = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto scale_result = getNextInput<c10::optional<TensorsPair>>(stackGetter);

  const auto mat1_shape = mat1.pt_t.sizes();
  const auto mat2_shape = mat2.pt_t.sizes();
  const auto mat1_dtype = mat1.pt_t.scalar_type();
  const auto mat2_dtype = mat2.pt_t.scalar_type();

  TORCH_CHECK(mat1_shape.size() == 2, "mat1 must be a matrix");
  TORCH_CHECK(mat2_shape.size() == 2, "mat2 must be a matrix");
  TORCH_CHECK(
      mat1_shape[1] == mat2_shape[0],
      "mat1 and mat2 shapes cannot be multiplied (",
      mat1_shape[0],
      "x",
      mat1_shape[1],
      " and ",
      mat2_shape[0],
      "x",
      mat2_shape[1],
      ")");
  TORCH_CHECK(
      !bias || bias->pt_t.numel() == mat2_shape[1],
      "Bias must be size ",
      mat2_shape[1],
      " but got ",
      bias->pt_t.numel());
  TORCH_CHECK(
      mat1_dtype == at::ScalarType::Float8_e5m2 ||
          mat1_dtype == at::ScalarType::Float8_e4m3fn,
      "Expected mat1 to be Float8_e5m2 or Float8_e4m3fn matrix got ",
      mat1_dtype);
  TORCH_CHECK(
      mat2_dtype == at::ScalarType::Float8_e5m2 ||
          mat2_dtype == at::ScalarType::Float8_e4m3fn,
      "Expected mat2 to be Float8_e5m2 or Float8_e4m3fn matrix got ",
      mat2_dtype);

  std::string guid = get_guid_with_precision("fp8_gemm", out_dtype);
  synTensor scale_a_syn = scale_a ? scale_a->syn_t : nullptr;
  synTensor scale_b_syn = scale_b ? scale_b->syn_t : nullptr;
  synTensor bias_syn = bias ? bias->syn_t : nullptr;
  synTensor scale_result_syn = scale_result ? scale_result->syn_t : nullptr;

  std::vector<synTensor> syn_inputs = {
      mat1.syn_t,
      mat2.syn_t,
      scale_a_syn,
      scale_b_syn,
      bias_syn,
      /* accumulate = */ nullptr,
      scale_result_syn};

  synGEMMParams gemm_params{false, false};

  const auto output_shape = ScaledMmMeta(stack)[0].shape;

  auto gemm = OpBackend::BuildNode(
      this,
      graph,
      {guid,
       std::move(syn_inputs),
       {{output_shape, out_dtype, 0}},
       &gemm_params,
       sizeof(gemm_params)});

  if (out_dtype != at::ScalarType::Float8_e5m2 and
      out_dtype != at::ScalarType::Float8_e4m3fn) {
    syn_out(0) = std::move(gemm[0]);
    return;
  }

  std::vector<sh::tensor> maybe_casted_sh;
  maybe_casted_sh.push_back(std::move(gemm[0]));

  if (out_dtype != at::ScalarType::Float) {
    maybe_casted_sh.push_back(BuildCast(
        this,
        graph,
        maybe_casted_sh[0].get(),
        output_shape,
        out_dtype,
        at::ScalarType::Float));
  }

  auto abs = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("abs_fwd", at::ScalarType::Float),
       {maybe_casted_sh.back().get()},
       {{output_shape, at::ScalarType::Float}}});

  ns_Reduction::ParamsV2 amax_params;
  amax_params.reductionDimensionMask = 0;
  amax_params.keepDim = false;

  auto amax = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision(
           "reduce_max_multi_dim_fwd", at::ScalarType::Float),
       {abs.back().get()},
       {{{}, at::ScalarType::Float, 1}},
       &amax_params,
       sizeof(amax_params)});

  syn_out(0) = std::move(maybe_casted_sh[0]);
  syn_out(1) = std::move(amax[0]);
}

} // namespace habana
