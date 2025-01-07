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

#include "generated/backend/smooth_l1_loss.h"
#include "generated/backend/smooth_l1_loss_backward.h"

namespace habana {

std::shared_ptr<void> FillSmoothL1LossFwdParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_SmoothL1LossKernel::Params);
  auto mode = stack.at(2).toInt();
  if (mode == at::Reduction::Reduction::Mean)
    params->mode = LossMode_t::LOSS_REDUCTION_MODE_MEAN;
  else if (mode == at::Reduction::Reduction::Sum)
    params->mode = LossMode_t::LOSS_REDUCTION_MODE_SUM;
  else
    params->mode = LossMode_t::LOSS_REDUCTION_MODE_NONE;
  params->beta = stack.at(3).toScalar().to<float>();
  return params;
}

OutputMetaDataVector SmoothL1LossMeta(const at::Stack& stack) {
  float beta = stack.at(3).toScalar().to<float>();
  TORCH_CHECK(
      beta >= 0, "smooth_l1_loss does not support negative values for beta.")
  const torch::Tensor& self = stack_tensor(stack, 0);
  int64_t reduction = stack.at(2).toInt();

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = (reduction == at::Reduction::Reduction::None)
      ? self.sizes().vec()
      : std::vector<int64_t>{};
  return {meta};
}

OutputMetaDataVector SmoothL1LossBackwardMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 1);

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = self.sizes().vec();
  return {meta};
}

SharedMetaDataVector SmoothL1LossBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& grad = stack_tensor(stack, 0);
  const auto& self = stack_tensor(stack, 1);
  const auto& target = stack_tensor(stack, 2);
  const float beta = stack.at(4).toScalar().to<float>();
  const auto rank = self.dim();
  const auto dtype = self.scalar_type();
  const auto constantPresent = rank > 1 ? 1 : 0;
  const auto kernels = (beta != 0 ? 6 : 4) + constantPresent;
  SharedMetaDataVector metaVec;
  metaVec.reserve(kernels);

  SharedMetaTensor commonTensor = {rank, dtype};
  if (constantPresent) {
    SharedMetaData constantSharedMeta{"constant"};
    constantSharedMeta.outputs_data = {commonTensor};
    metaVec.push_back(constantSharedMeta);
  }

  SharedMetaData subSharedMeta{"sub"};
  subSharedMeta.inputs_data = {commonTensor, commonTensor};
  subSharedMeta.outputs_data = {commonTensor};
  metaVec.push_back(subSharedMeta);

  SharedMetaData signSharedMeta{"sign_fwd"};
  signSharedMeta.inputs_data = {commonTensor};
  signSharedMeta.outputs_data = {commonTensor};
  metaVec.push_back(signSharedMeta);

  SharedMetaData multSharedMeta{"mult"};
  multSharedMeta.inputs_data = {commonTensor, commonTensor};
  multSharedMeta.outputs_data = {commonTensor};
  metaVec.push_back(multSharedMeta);

  if (beta == 0)
    return metaVec;

  SharedMetaData absSharedMeta{"abs_fwd"};
  absSharedMeta.inputs_data = {commonTensor};
  absSharedMeta.outputs_data = {commonTensor};
  metaVec.push_back(absSharedMeta);

  SharedMetaData lessSharedMeta{"less_fwd"};
  lessSharedMeta.inputs_data = {commonTensor, commonTensor};
  lessSharedMeta.outputs_data.emplace_back(rank, c10::ScalarType::Bool);
  metaVec.push_back(lessSharedMeta);

  SharedMetaData whereSharedMeta{"where_fwd"};
  whereSharedMeta.inputs_data = {
      lessSharedMeta.outputs_data[0], commonTensor, commonTensor};
  whereSharedMeta.outputs_data = {commonTensor};
  metaVec.push_back(whereSharedMeta);

  return metaVec;
}

void SmoothL1LossBwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = SmoothL1LossBackwardMeta(stack)[0];

  float beta = stack.at(4).toScalar().to<float>();
  TORCH_CHECK(
      beta >= 0,
      "smooth_l1_loss_backward does not support negative values for beta.")
  auto mode = stack.at(3).toInt();
  float norm_factor = (mode == at::Reduction::Reduction::Mean)
      ? 1 / static_cast<float>(stack_tensor(stack, 1).numel())
      : 1;

  std::vector<synapse_helpers::tensor> t_l0;

  auto t_diff = BuildOp(
      graph,
      get_guid_with_precision("sub", meta.dtype),
      {syn_in(1), syn_in(2)},
      {{meta.shape, meta.dtype}});

  if (mode == at::Reduction::Reduction::Mean) {
    auto t_norm_factor =
        ConstantHelper(graph, norm_factor, meta.dtype, meta.shape);

    auto t_mul = BuildOp(
        graph,
        get_guid_with_precision("mult", meta.dtype),
        {syn_in(0), t_norm_factor.get()},
        {{meta.shape, meta.dtype}});

    auto t_sign = BuildOp(
        graph,
        get_guid_with_precision("sign_fwd", meta.dtype),
        {t_diff.at(0).get()},
        {{meta.shape, meta.dtype}});

    if (beta == 0) {
      t_l0 = BuildOp(
          graph,
          get_guid_with_precision("mult", meta.dtype),
          {t_mul.at(0).get(), t_sign.at(0).get()},
          {{meta.shape, meta.dtype, 0}});

      syn_out(0) = std::move(t_l0.at(0));
      return;
    }

    t_l0 = BuildOp(
        graph,
        get_guid_with_precision("mult", meta.dtype),
        {t_mul.at(0).get(), t_sign.at(0).get()},
        {{meta.shape, meta.dtype}});

  } else {
    auto t_sign = BuildOp(
        graph,
        get_guid_with_precision("sign_fwd", meta.dtype),
        {t_diff.at(0).get()},
        {{meta.shape, meta.dtype}});

    if (beta == 0) {
      t_l0 = BuildOp(
          graph,
          get_guid_with_precision("mult", meta.dtype),
          {syn_in(0), t_sign.at(0).get()},
          {{meta.shape, meta.dtype, 0}});

      syn_out(0) = std::move(t_l0.at(0));
      return;
    }

    t_l0 = BuildOp(
        graph,
        get_guid_with_precision("mult", meta.dtype),
        {syn_in(0), t_sign.at(0).get()},
        {{meta.shape, meta.dtype}});
  }

  auto t_mulfactor =
      ConstantHelper(graph, norm_factor / beta, meta.dtype, meta.shape);

  auto t_l2_temp = BuildOp(
      graph,
      get_guid_with_precision("mult", meta.dtype),
      {syn_in(0), t_mulfactor.get()},
      {{meta.shape, meta.dtype}});

  auto t_l2 = BuildOp(
      graph,
      get_guid_with_precision("mult", meta.dtype),
      {t_diff.at(0).get(), t_l2_temp.at(0).get()},
      {{meta.shape, meta.dtype}});

  auto t_mask_const = ConstantHelper(graph, beta, meta.dtype, meta.shape);

  auto t_abs = BuildOp(
      graph,
      get_guid_with_precision("abs_fwd", meta.dtype),
      {t_diff.at(0).get()},
      {{meta.shape, meta.dtype}});

  auto mask = BuildOp(
      graph,
      get_guid_with_precision("less_fwd", meta.dtype),
      {t_abs.at(0).get(), t_mask_const.get()},
      {{meta.shape, at::kBool}});

  auto grad_in = BuildOp(
      graph,
      get_guid_with_precision("where_fwd", meta.dtype),
      {mask.at(0).get(), t_l2.at(0).get(), t_l0.at(0).get()},
      {{meta.shape, meta.dtype, 0}});

  syn_out(0) = std::move(grad_in.at(0));
}
} // namespace habana
