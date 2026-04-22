/**
 * Copyright (c) 2025 Intel Corporation
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

#include "generated/backend/block_softmax_adjustment.h"
#include "hpu_ops/custom_op_outshape.h"

#define CEIL_TO_VEC_SIZE(num, roundup) \
  ((((num) + (roundup) - 1) / (roundup)) * (roundup))

namespace habana {

/* ======= block_softmax ======= */

template <class DimT>
std::tuple<std::vector<DimT>, std::vector<DimT>, std::vector<DimT>>
BlockSoftmaxOutputSize(
    c10::ArrayRef<DimT> block_maxes_shape,
    const at::ScalarType block_maxes_dtype) {
  auto common_dim =
      block_maxes_shape[1] * block_maxes_shape[2] * block_maxes_shape[3];

  if (habana::HPUDeviceContext::get_device().type() == synDeviceGaudi3) {
    const int64_t vec_size =
        (block_maxes_dtype == c10::ScalarType::Float) ? 64 : 128;
    common_dim = CEIL_TO_VEC_SIZE(common_dim, vec_size);
  }
  const std::vector<DimT> out_shape_maxes{block_maxes_shape[0], common_dim};
  return {block_maxes_shape.vec(), out_shape_maxes, out_shape_maxes};
}

sym_sizes_vec block_softmax_out_shape(
    const std::vector<at::Tensor>& inputs,
    const std::vector<int64_t>& params) {
  HABANA_ASSERT(inputs.size() == 1);
  HABANA_ASSERT(params.empty());
  const auto [out_shape_attn, out_shape_maxes, out_shape_sums] =
      BlockSoftmaxOutputSize(inputs[0].sym_sizes(), inputs[0].scalar_type());
  return {out_shape_attn, out_shape_maxes, out_shape_sums};
}

REGISTER_CUSTOM_OP_OUTSHAPE_FUN(block_softmax, block_softmax_out_shape);

// Meta function for block_softmax
OutputMetaDataVector BlockSoftmaxMeta(const at::Stack& stack) {
  // Calculate output shapes
  const auto& attn = stack_tensor(stack, 0);
  const auto input_dtype = attn.scalar_type();
  const auto attn_shape = attn.sizes();
  const auto block_bias_shape = stack_tensor(stack, 1).sizes();
  const auto block_indicators_shape = stack_tensor(stack, 2).sizes();

  static constexpr int input_dim = 5;
  const auto out_dtype =
      stack.at(4).toOptional<c10::ScalarType>().value_or(input_dtype);

  if (out_dtype == at::ScalarType::Float8_e4m3fn) {
    TORCH_CHECK(
        attn.scalar_type() == at::ScalarType::BFloat16,
        "When output_dtype is float8_e4m3fn, input_dtype must be bfloat16, got ",
        attn.scalar_type());
  }

  TORCH_CHECK(
      attn_shape.size() == input_dim,
      "attn tensor must be ",
      input_dim,
      "-dimensional, got ",
      attn_shape.size());

  for (const auto dim : {1, 2}) {
    TORCH_CHECK(
        block_bias_shape[dim] == 1,
        "block_bias tensor must have size 1 at dimension ",
        dim,
        ", got ",
        block_bias_shape[dim]);
  }

  for (const auto dim : {0, 3, 4}) {
    TORCH_CHECK(
        attn_shape[dim] == block_bias_shape[dim],
        "attn tensor and block_bias tensor must have the same size at dimension ",
        dim,
        ", got ",
        attn_shape[dim],
        " and ",
        block_bias_shape[dim]);
  }

  TORCH_CHECK(
      block_indicators_shape.size() == 1,
      "block_indicators tensor must be 1D tensor",
      ", got ",
      block_indicators_shape.size());

  TORCH_CHECK(
      block_indicators_shape[0] == attn_shape[0],
      "block_indicators tensor must be 1D tensor of size ",
      attn_shape[0],
      ", got ",
      block_indicators_shape[0]);

  const auto [out_shape_attn, out_shape_maxes, out_shape_sums] =
      BlockSoftmaxOutputSize(attn_shape, input_dtype);

  OutputMetaDataVector metaVec = {
      {out_dtype, out_shape_attn},
      {input_dtype, out_shape_maxes},
      {input_dtype, out_shape_sums}};

  return metaVec;
}

FillParamsT BlockSoftmaxParams(const at::Stack& stack) {
  PARAMS_STUB(ns_BlockSoftmax::Params);

  const auto out_dtype = stack.at(4).toOptional<c10::ScalarType>().value_or(
      stack_tensor(stack, 0).scalar_type());
  unsigned mode = BLOCK_SOFTMAX_MODE_DEFAULT;
  if (out_dtype == at::ScalarType::Float8_e4m3fn) {
    mode |= BLOCK_SOFTMAX_MODE_HF8_OUT;
    const auto output_scale = stack.at(3).toDouble();
    if (stack.at(5).toBool()) {
      TORCH_CHECK(
          output_scale == 1.0,
          "When use_hf8_exp_lut is true, output_scale must be 1.0, got ",
          output_scale);
      mode |= BLOCK_SOFTMAX_MODE_USE_HF8_EXP_LUT;
    } else {
      params->outputScale = output_scale;
    }
  }

  params->mode = static_cast<BlockSoftmaxMode_t>(mode);
  return paramsT;
}

/* ======= block_softmax_adjustment ======= */

OutputMetaDataVector BlockSoftmaxAdjustmentMeta(const at::Stack& stack) {
  const auto& block_maxes = stack_tensor(stack, 0);
  const auto block_maxes_shape = block_maxes.sizes();
  const auto& block_sums = stack_tensor(stack, 1);
  const auto block_sums_shape = block_sums.sizes();
  const auto& block_groups = stack_tensor(stack, 2);
  const auto block_groups_shape = block_groups.sizes();
  const auto out_shape = stack.at(4).toOptional<std::vector<int64_t>>();
  const auto is_fused_mult = stack.at(5).isTensor();
  const auto out_dtype = stack.at(7).toOptional<c10::ScalarType>().value_or(
      block_maxes.scalar_type());
  const bool is_staged = stack.at(8).toBool();
  static constexpr int input_dim = 5;
  std::vector<int64_t> adjustment_out_shape;

  if (out_dtype == at::ScalarType::Float8_e4m3fn) {
    TORCH_CHECK(
        block_maxes.scalar_type() == at::ScalarType::BFloat16,
        "When output_dtype is float8_e4m3fn, input_dtype must be bfloat16, got ",
        block_maxes.scalar_type());
  }

  if (is_staged) {
    TORCH_CHECK(
        out_shape.has_value() or is_fused_mult,
        "In staged mode, either out_shape or fused_mult must be provided.");
    const int staged_input_dim = 2;
    TORCH_CHECK(
        block_maxes_shape.size() == staged_input_dim,
        "In staged mode block_maxes tensor must be ",
        staged_input_dim,
        "-dimensional, got ",
        block_maxes_shape.size());
    TORCH_CHECK(
        block_maxes_shape[1] == block_sums_shape[1],
        "In staged mode block_maxes and block_sums tensors must have the same size of dim 1, got ",
        block_maxes_shape[1],
        " and ",
        block_sums_shape[1]);
    TORCH_CHECK(
        block_sums_shape[0] == stack.at(3).toInt(),
        "In staged mode block_sums tensor dim 0 size must be equal to batch_size, got ",
        block_sums_shape[0],
        " and ",
        stack.at(3).toInt());
  } else if (not out_shape.has_value()) {
    TORCH_CHECK(
        block_maxes_shape.size() == input_dim,
        "If no out_shape provided, block_maxes tensor must be ",
        input_dim,
        "-dimensional, got ",
        block_maxes_shape.size());
    TORCH_CHECK(
        block_maxes_shape == block_sums_shape,
        "block_maxes and block_sums tensors must have the same shape, got ",
        block_maxes_shape,
        " and ",
        block_sums_shape);
  }

  TORCH_CHECK(
      block_groups_shape.size() == 1,
      "block_groups tensor must be 1D tensor",
      ", got ",
      block_groups_shape.size());

  TORCH_CHECK(
      block_groups_shape[0] == block_maxes_shape[0],
      "block_groups tensor must be 1D tensor of size ",
      block_maxes_shape[0],
      ", got ",
      block_groups_shape[0]);

  if (is_fused_mult) {
    const auto& fused_mult = stack_tensor(stack, 5);
    adjustment_out_shape = fused_mult.sizes().vec();
    TORCH_CHECK(
        adjustment_out_shape.size() == input_dim,
        "fused_mult_attn tensor must be ",
        input_dim,
        "-dimensional, got ",
        adjustment_out_shape.size());
  } else if (out_shape.has_value()) {
    adjustment_out_shape = out_shape.value();
  } else {
    adjustment_out_shape = block_maxes_shape.vec();
  }

  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.shape = adjustment_out_shape;
  meta.dtype = out_dtype;

  return metaVec;
}

FillParamsT BlockSoftmaxAdjustmentParams(const at::Stack& stack) {
  const auto input_dtype = stack_tensor(stack, 0).scalar_type();
  const auto is_fused_mult_attn = not stack.at(5).isNone();
  const auto out_dtype = stack.at(7).toOptional<c10::ScalarType>().value_or(
      stack_tensor(stack, 0).scalar_type());
  const bool is_staged = stack.at(8).toBool();

  PARAMS_STUB(ns_BlockSoftmaxAdjustment::ParamsV2);
  params->batchSize = stack.at(3).toInt();

  unsigned mode = BLOCK_SOFTMAX_ADJUSTMENT_MODE_DEFAULT_MODE;
  if (out_dtype == at::ScalarType::Float8_e4m3fn) {
    TORCH_CHECK(
        input_dtype == at::ScalarType::BFloat16,
        "When output_dtype is float8_e4m3fn, input_dtype must be bfloat16, got ",
        input_dtype);
    mode |= BLOCK_SOFTMAX_ADJUSTMENT_MODE_HF8_OUT;
    params->outputScale = stack.at(6).toDouble();
  }
  if (is_fused_mult_attn) {
    mode |= BLOCK_SOFTMAX_ADJUSTMENT_MODE_FUSED_MULT_ATTN;
  }
  if (is_staged) {
    mode |= BLOCK_SOFTMAX_ADJUSTMENT_MODE_ADJUSTMENT_STAGE;
  }

  params->mode = static_cast<BlockSoftmaxAdjustmentMode_t>(mode);
  return paramsT;
}

/* ======= block_softmax_staged_sum_max ======= */

template <class DimT>
std::tuple<std::vector<DimT>, std::vector<DimT>>
BlockSoftmaxStagedSumMaxOutputSize(
    c10::ArrayRef<DimT> block_maxes_shape,
    const at::ScalarType block_maxes_dtype,
    const int64_t batch_size) {
  DimT common_dim{};
  if (block_maxes_shape.size() == 5) {
    common_dim =
        block_maxes_shape[1] * block_maxes_shape[2] * block_maxes_shape[3];

    if (habana::HPUDeviceContext::get_device().type() == synDeviceGaudi3) {
      const int64_t vec_size =
          (block_maxes_dtype == c10::ScalarType::Float) ? 64 : 128;
      common_dim = CEIL_TO_VEC_SIZE(common_dim, vec_size);
    }
  } else if (block_maxes_shape.size() == 2) {
    common_dim = block_maxes_shape[1];
  } else {
    TORCH_CHECK(
        false,
        "block_maxes tensor must be either 5D or 2D tensor, got ",
        block_maxes_shape.size(),
        "D tensor");
  }

  const std::vector<DimT> out_shape_maxes{block_maxes_shape[0], common_dim};
  const std::vector<DimT> out_shape_sums{batch_size, common_dim};
  return {out_shape_maxes, out_shape_sums};
}

sym_sizes_vec block_softmax_staged_sum_max_out_shape(
    const std::vector<at::Tensor>& inputs,
    const std::vector<int64_t>& params) {
  HABANA_ASSERT(inputs.size() == 1);
  HABANA_ASSERT(params.size() == 1);
  const auto [out_shape_maxes, out_shape_sums] =
      BlockSoftmaxStagedSumMaxOutputSize(
          inputs[0].sym_sizes(), inputs[0].scalar_type(), params[0]);
  return {out_shape_maxes, out_shape_sums};
}

REGISTER_CUSTOM_OP_OUTSHAPE_FUN(
    block_softmax_staged_sum_max,
    block_softmax_staged_sum_max_out_shape);

OutputMetaDataVector BlockSoftmaxStagedSumMaxMeta(const at::Stack& stack) {
  const auto& block_maxes = stack_tensor(stack, 0);
  const auto [out_shape_maxes, out_shape_sums] =
      BlockSoftmaxStagedSumMaxOutputSize(
          block_maxes.sizes(), block_maxes.scalar_type(), stack.at(3).toInt());
  const auto out_dtype = block_maxes.scalar_type();

  OutputMetaData meta_maxes;
  meta_maxes.shape = out_shape_maxes;
  meta_maxes.dtype = out_dtype;

  OutputMetaData meta_sums;
  meta_sums.shape = out_shape_sums;
  meta_sums.dtype = out_dtype;

  return {meta_maxes, meta_sums};
}

FillParamsT BlockSoftmaxStagedSumMaxParams(const at::Stack& stack) {
  PARAMS_STUB(ns_BlockSoftmaxAdjustment::ParamsV2);
  params->batchSize = stack.at(3).toInt();
  params->mode = static_cast<BlockSoftmaxAdjustmentMode_t>(
      BLOCK_SOFTMAX_ADJUSTMENT_MODE_DEFAULT_MODE |
      BLOCK_SOFTMAX_ADJUSTMENT_MODE_MAX_SUM_STAGES);
  return paramsT;
}

} // namespace habana
