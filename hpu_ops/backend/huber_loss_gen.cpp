/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/huber_loss.h"
#include "generated/backend/huber_loss_backward.h"

namespace habana {

std::shared_ptr<void> FillHuberLossFwdParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_HuberLossKernel::Params);

  double delta = stack.at(3).toScalar().to<double>();
  params->delta = delta;

  auto mode = stack.at(2).toInt();
  if (mode == at::Reduction::Reduction::Mean)
    params->mode = LossMode_t::LOSS_REDUCTION_MODE_MEAN;
  else if (mode == at::Reduction::Reduction::Sum)
    params->mode = LossMode_t::LOSS_REDUCTION_MODE_SUM;
  else
    params->mode = LossMode_t::LOSS_REDUCTION_MODE_NONE;
  return params;
}

OutputMetaDataVector HuberLossMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  int64_t reduction = stack.at(2).toInt();
  double delta = stack.at(3).toScalar().to<double>();
  TORCH_CHECK(
      delta >= 0, "huber_loss does not support negative values for delta.")

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = {};
  if (reduction == at::Reduction::Reduction::None)
    meta.shape = self.sizes().vec();

  return {meta};
}

OutputMetaDataVector HuberLossBackwardMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 1);
  OutputMetaData meta;
  meta.shape = self.sizes().vec();
  meta.dtype = self.scalar_type();
  return {meta};
}

SharedMetaDataVector HuberLossBackwardSharedMeta(const at::Stack& stack) {
  auto grad = stack_tensor(stack, 0);
  auto gradRank = grad.dim();
  auto self = stack_tensor(stack, 1);
  auto selfRank = self.dim();
  auto target = stack_tensor(stack, 2);
  auto targetRank = target.dim();
  auto dtype = self.scalar_type();

  SharedMetaDataVector metaVec;
  metaVec.reserve(7);
  SharedMetaTensor commonSharedTensor{selfRank, dtype};
  SharedMetaTensor gradSharedTensor{gradRank, dtype};
  SharedMetaTensor targetSharedTensor{targetRank, dtype};
  SharedMetaVector commonBinarySharedInput{
      commonSharedTensor, commonSharedTensor};
  SharedMetaVector commonUnarySharedInput = {commonSharedTensor};
  auto& commonSharedOutput = commonUnarySharedInput;

  SharedMetaData subSharedMeta{"sub"};
  subSharedMeta.inputs_data = {commonSharedTensor, targetSharedTensor};
  subSharedMeta.outputs_data = commonSharedOutput;
  metaVec.push_back(subSharedMeta);

  SharedMetaData multGradNormSharedMeta{"mult"};
  multGradNormSharedMeta.inputs_data = {gradSharedTensor, commonSharedTensor};
  multGradNormSharedMeta.outputs_data = commonSharedOutput;
  metaVec.push_back(multGradNormSharedMeta);

  SharedMetaData signSharedMeta{"sign_fwd"};
  signSharedMeta.inputs_data = commonUnarySharedInput;
  signSharedMeta.outputs_data = commonSharedOutput;
  metaVec.push_back(signSharedMeta);

  SharedMetaData multCommonSharedMeta{"mult"};
  multCommonSharedMeta.inputs_data = commonBinarySharedInput;
  multCommonSharedMeta.outputs_data = commonSharedOutput;
  metaVec.push_back(multCommonSharedMeta);

  SharedMetaData absSharedMeta{"abs_fwd"};
  absSharedMeta.inputs_data = commonUnarySharedInput;
  absSharedMeta.outputs_data = commonSharedOutput;
  metaVec.push_back(absSharedMeta);

  SharedMetaData lessSharedMeta{"less_fwd"};
  lessSharedMeta.inputs_data = commonBinarySharedInput;
  lessSharedMeta.outputs_data.emplace_back(selfRank, c10::ScalarType::Bool);
  metaVec.push_back(lessSharedMeta);

  SharedMetaData whereSharedMeta{"where_fwd"};
  whereSharedMeta.inputs_data = {
      lessSharedMeta.outputs_data[0], commonSharedTensor, commonSharedTensor};
  whereSharedMeta.outputs_data = commonSharedOutput;
  metaVec.push_back(whereSharedMeta);

  return metaVec;
}

void HuberLossBwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = HuberLossBackwardMeta(stack)[0];

  float delta = stack.at(4).toScalar().to<float>();
  TORCH_CHECK(
      delta >= 0,
      "huber_loss_backward does not support negative values for delta.")
  auto mode = stack.at(3).toInt();
  float norm_factor = (mode == at::Reduction::Reduction::Mean)
      ? 1 / static_cast<float>(stack_tensor(stack, 1).numel())
      : 1;

  auto norm = ConstantHelper(graph, norm_factor, meta.dtype, meta.shape);

  auto delta_const = ConstantHelper(graph, delta, meta.dtype, meta.shape);

  auto t_diff = BuildOp(
      graph,
      get_guid_with_precision("sub", meta.dtype),
      {syn_in(1), syn_in(2)},
      {{meta.shape, meta.dtype}});

  auto t_mul = BuildOp(
      graph,
      get_guid_with_precision("mult", meta.dtype),
      {syn_in(0), norm.get()},
      {{meta.shape, meta.dtype}});

  auto t_sign = BuildOp(
      graph,
      get_guid_with_precision("sign_fwd", meta.dtype),
      {t_diff.at(0).get()},
      {{meta.shape, meta.dtype}});

  auto t_0 = BuildOp(
      graph,
      get_guid_with_precision("mult", meta.dtype),
      {t_mul.at(0).get(), delta_const.get()},
      {{meta.shape, meta.dtype}});

  auto t_1 = BuildOp(
      graph,
      get_guid_with_precision("mult", meta.dtype),
      {t_0.at(0).get(), t_sign.at(0).get()},
      {{meta.shape, meta.dtype}});

  auto t_2 = BuildOp(
      graph,
      get_guid_with_precision("mult", meta.dtype),
      {t_diff.at(0).get(), t_mul.at(0).get()},
      {{meta.shape, meta.dtype}});

  auto t_abs = BuildOp(
      graph,
      get_guid_with_precision("abs_fwd", meta.dtype),
      {t_diff.at(0).get()},
      {{meta.shape, meta.dtype}});

  auto mask_bwd = BuildOp(
      graph,
      get_guid_with_precision("less_fwd", meta.dtype),
      {t_abs.at(0).get(), delta_const.get()},
      {{meta.shape, at::kBool}});

  auto grad_in = BuildOp(
      graph,
      get_guid_with_precision("where_fwd", meta.dtype),
      {mask_bwd.at(0).get(), t_2.at(0).get(), t_1.at(0).get()},
      {{meta.shape, meta.dtype, 0}});

  syn_out(0) = std::move(grad_in.at(0));
}
} // namespace habana
