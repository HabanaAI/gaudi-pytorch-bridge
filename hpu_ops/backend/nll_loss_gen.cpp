/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include "backend/helpers/create_tensor.h"
#include "backend/helpers/tensor_utils.h"
#include "generated/backend/nll_loss2d_backward.h"
#include "generated/backend/nll_loss2d_forward.h"
#include "generated/backend/nll_loss_backward.h"
#include "generated/backend/nll_loss_forward.h"

namespace habana {

OutputMetaDataVector NllLossFwdMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  const torch::Tensor& target = stack_tensor(stack, 1);
  int64_t reduction = stack.at(3).toInt();
  OutputMetaDataVector meta(2);
  for (int i = 0; i < 2; ++i) {
    meta.at(i).dtype = self.scalar_type();
    meta.at(i).shape = {};
  }
  if (reduction == at::Reduction::Reduction::None) {
    meta.at(0).shape = target.sizes().vec();
  }
  return meta;
}

OutputMetaDataVector NllLossBwdMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  const torch::Tensor& target = stack_tensor(stack, 1);
  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = target.sizes().vec();
  return {meta};
}

sizes_vec NllLossBwdShapeTnsrShape(const at::Stack& stack) {
  // Return correct Shape Tensor size
  const torch::Tensor& target = stack_tensor(stack, 2);
  const torch::Tensor& input = stack_tensor(stack, 1);
  auto target_vec = target.sizes().vec();
  target_vec.push_back(input.sizes().at(1));
  return {target_vec};
}

static FillParamsT FillNllLossParams(int64_t reduction, int64_t ignore_index) {
  PARAMS_STUB(ns_NLLLossKernel::ParamsOptionalIgnoreIndex);
  switch (reduction) {
    case at::Reduction::Reduction::None:
      params->mode = NLLLossMode_t::NLL_LOSS_MODE_NONE;
      break;
    case at::Reduction::Reduction::Mean:
      params->mode = NLLLossMode_t::NLL_LOSS_MODE_MEAN;
      break;
    case at::Reduction::Reduction::Sum:
      params->mode = NLLLossMode_t::NLL_LOSS_MODE_SUM;
      break;
    default:
      HABANA_ASSERT(false, "Unsupported reduction in nll_loss: ", reduction);
  }
  params->ignoreIndexValue = ignore_index;
  return paramsT;
}

FillParamsT FillNllLossFwdParams(const at::Stack& stack) {
  auto ignore = stack.at(3).toInt();
  auto reduction = stack.at(4).toInt();
  return FillNllLossParams(ignore, reduction);
}

FillParamsT FillNllLossBwdParams(const at::Stack& stack) {
  auto ignore = stack.at(4).toInt();
  auto reduction = stack.at(5).toInt();
  return FillNllLossParams(ignore, reduction);
}
enum modes { Fwd2D, Bwd2D };

static std::vector<synapse_helpers::tensor> NllLoss(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input,
    const OutputMetaData& meta,
    const FillParamsT& params,
    std::optional<int> final_index = std::nullopt) {
  return OpBackend::BuildNode(
      op,
      graph,
      {op->GetGuid(),
       std::move(input),
       {{meta.shape, meta.dtype, final_index}},
       params.ptr(),
       params.size()});
}

bool NllLossDSSTMeta(
    habana_helpers::IShapeList& inputs,
    habana_helpers::IShapeList& outputs) {
  PT_BRIDGE_DEBUG("NllLossDSSTMeta called ");

  // If the 3rd input is a scalar, then the weight is optional and None
  if (inputs.at(3).isScalar()) {
    auto reduction_mode =
        inputs.at(3).getScalar().toInt(); // get reduction type
    if (reduction_mode == at::Reduction::Reduction::Mean) {
      std::vector<int64_t> const_shape = {1};
      habana_helpers::UpdateSTShapeInfo(
          const_shape); // for the constant created in nll_loss_bwd mean backend
    }
    std::vector<int64_t> out_shape = outputs[0].getTensorShape();
    PT_BRIDGE_DEBUG("NllLossDSSTMeta output shape ", out_shape);
    habana_helpers::UpdateSTShapeInfo(out_shape);
    return true;
  }
  return false;
}

static std::vector<synapse_helpers::tensor> NllLossBwdFunc(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input,
    const OutputMetaData& meta,
    const FillParamsT& params,
    std::optional<int> final_index = std::nullopt,
    at::IntArrayRef shapeTnsrSize = {}) {
  // This helper function is used only when weight is none
  op->CreateShapeTensorInput(graph, meta.dtype, shapeTnsrSize, input);
  return NllLoss(op, graph, input, meta, params, final_index);
}

using namespace std::literals;

static std::vector<synapse_helpers::tensor> ComputeWeightsSum(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Stack& stack,
    const OutputMetaData& meta,
    std::vector<synTensor> inputs,
    std::optional<int> final_index = std::nullopt) {
  constexpr auto synTargetIdx = 0;
  constexpr auto synWeightIdx = 1;

  const auto target = stack_tensor(stack, 1);
  const auto weights = stack_tensor(stack, 2);

  const auto targetSizes = target.sizes().vec();
  const auto targetFlattenSize = std::accumulate(
      targetSizes.cbegin(), targetSizes.cend(), 1, std::multiplies<int>{});

  auto flattenTarget = OpBackend::BuildFlatten(
      op,
      graph,
      std::move(inputs[synTargetIdx]),
      {targetFlattenSize},
      c10::ScalarType::Int); // using Int by default for gaudi3 support

  ns_GatherElementsKernel::ParamsV2 gatherParams{};
  gatherParams.axis = 0;
  gatherParams.isSort = false;

  auto targetMappedToWeights = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("gather_elements_fwd"sv, weights.scalar_type()),
       std::vector<synTensor>{
           std::move(inputs[synWeightIdx]), std::move(flattenTarget.get())},
       {{targetFlattenSize, weights.scalar_type()}},
       &gatherParams,
       sizeof(gatherParams)});

  ns_Reduction::Params reduceParams{.reductionDimension = 0};

  return OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("reduce_sum_fwd"sv, meta.dtype),
       std::vector<synTensor>{std::move(targetMappedToWeights[0].get())},
       {{1, meta.dtype, final_index}},
       &reduceParams,
       sizeof(reduceParams)});
}

SharedMetaDataVector NllLoss2DFwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& self = stack_tensor(stack, 0);
  const auto& target = stack_tensor(stack, 1);
  const auto& weight =
      stack.at(2).toOptional<torch::Tensor>().value_or(at::Tensor());
  const int64_t reduction = stack.at(3).toInt();
  const auto outputRank =
      reduction == at::Reduction::Reduction::None ? target.dim() : 1;
  const auto dtype = self.scalar_type();

  SharedMetaDataVector metaVec;
  SharedMetaData nllLossFwdSharedMeta{"nll_loss_fwd"};
  nllLossFwdSharedMeta.outputs_data.emplace_back(outputRank, dtype);
  nllLossFwdSharedMeta.inputs_data = {
      {self.dim(), dtype}, {target.dim(), target.scalar_type()}};
  if (weight.defined()) {
    nllLossFwdSharedMeta.inputs_data.emplace_back(weight.dim(), dtype);
    nllLossFwdSharedMeta.inputs_data.emplace_back(1, dtype);

    const auto weightDtype = weight.scalar_type();
    SharedMetaData gatherElementsSharedMeta{"gather_elements_fwd"};
    gatherElementsSharedMeta.inputs_data = {
        {weight.dim(), weightDtype}, {1, target.scalar_type()}};
    gatherElementsSharedMeta.outputs_data.emplace_back(1, weightDtype);
    metaVec.push_back(gatherElementsSharedMeta);

    SharedMetaData reduceWeightSharedMeta{"reduce_sum_fwd"};
    reduceWeightSharedMeta.inputs_data.emplace_back(weight.dim(), dtype);
    reduceWeightSharedMeta.outputs_data.emplace_back(1, dtype);
    metaVec.push_back(reduceWeightSharedMeta);
  }

  metaVec.push_back(nllLossFwdSharedMeta);
  return metaVec;
}

void NllLoss2DFwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& params = FillParams(stack);
  const auto meta = OutputMeta(stack)[0];

  std::vector<synapse_helpers::tensor> nll_loss;
  const auto self = stack_tensor(stack, 0);
  int64_t reduction = stack.at(3).toInt();

  // from
  // https://github.com/pytorch/pytorch/blob/4015166e5d51bc39d5a81aa59ad49720ec2a23fe/aten/src/ATen/native/LossNLL2d.cpp#L183C17-L183C24
  auto self_sizes = self.sizes().vec();
  int64_t N = self_sizes[0];
  int64_t HW = self_sizes[2] * self_sizes[3];
  int64_t NHW_multiplier = N * HW;

  if (reduction != at::Reduction::Reduction::None) {
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});
  }

  if (stack.at(2).isNone()) { // weight is none
    auto total_weight = ConstantHelper(
        graph, static_cast<float>(NHW_multiplier), meta.dtype, {1}, 1);

    nll_loss = NllLoss(this, graph, {syn_in(0), syn_in(1)}, meta, params, 0);

    syn_out(1) = std::move(total_weight);
  } else { // weight is not none

    auto weight_sum =
        ComputeWeightsSum(this, graph, stack, meta, {syn_in(1), syn_in(2)}, 1);

    nll_loss = NllLoss(
        this,
        graph,
        {syn_in(0), syn_in(1), syn_in(2), weight_sum[0].get()},
        meta,
        params,
        0);

    syn_out(1) = std::move(weight_sum[0]);
  }

  syn_out(0) = std::move(nll_loss[0]);
}

SharedMetaDataVector NllLossBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& grad = stack_tensor(stack, 0);
  const auto& self = stack_tensor(stack, 1);
  const auto& target = stack_tensor(stack, 2);
  const auto& weight =
      stack.at(3).toOptional<torch::Tensor>().value_or(at::Tensor());
  const auto& totalWeight = stack_tensor(stack, 6);
  auto rank = self.dim();
  auto dtype = grad.scalar_type();
  const auto isWeightTensor = weight.defined();
  const int64_t reduction = stack.at(4).toInt();
  const std::string guid = isWeightTensor ? "cnll_loss_bwd" : "nll_loss_bwd";

  SharedMetaDataVector metaVec;
  SharedMetaData nllLossBwdSharedMeta{guid};
  nllLossBwdSharedMeta.inputs_data.emplace_back(grad.dim(), dtype);
  if (isWeightTensor)
    nllLossBwdSharedMeta.inputs_data.emplace_back(rank, dtype);
  nllLossBwdSharedMeta.inputs_data.emplace_back(
      target.dim(), target.scalar_type());
  if (isWeightTensor) {
    nllLossBwdSharedMeta.inputs_data.emplace_back(weight.dim(), dtype);
    nllLossBwdSharedMeta.inputs_data.emplace_back(totalWeight.dim(), dtype);
  }

  nllLossBwdSharedMeta.outputs_data.emplace_back(rank, dtype);

  if (!isWeightTensor && reduction == at::Reduction::Reduction::Mean) {
    SharedMetaData divSharedMeta{"div_fwd"};
    divSharedMeta.inputs_data.emplace_back(1, dtype);
    divSharedMeta.inputs_data.emplace_back(1, dtype);
    divSharedMeta.outputs_data.emplace_back(1, dtype);

    SharedMetaData mulSharedMeta{"mult_fwd"};
    mulSharedMeta.inputs_data.emplace_back(rank, dtype);
    mulSharedMeta.inputs_data.emplace_back(1, dtype);
    mulSharedMeta.outputs_data.emplace_back(rank, dtype);
    metaVec.push_back(divSharedMeta);
    metaVec.push_back(mulSharedMeta);
  }
  metaVec.push_back(nllLossBwdSharedMeta);
  return metaVec;
}

void NllLossBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& params = FillParams(stack);
  const auto meta = OutputMeta(stack)[0];
  const auto shapeTnsrSize = NllLossBwdShapeTnsrShape(stack)[0];
  auto is_weight_none = stack.at(3).isNone();
  int64_t reduction = stack.at(4).toInt();
  int64_t scalar_shape[] = {1};

  // To divide the nll_loss_bwd output with the correct total_weight,
  // first mul with the assumed batch size in tpc and then divide
  // by the correct total_weight from nll_loss_fwd. The first if block is
  // required only to fix the total_weights assumption done by tpc

  if (is_weight_none && reduction == at::Reduction::Reduction::Mean) {
    const auto& self = stack_tensor(stack, 1); // self tensor
    auto batch_size =
        self.sizes().vec().at(0); // batch size used to correct the total_weight
                                  // result from tpc_kernel
    auto batch_size_constant = ConstantHelper(
        graph, static_cast<float>(batch_size), meta.dtype, scalar_shape);

    // mul_factor = batch_size / total_weight
    auto mul_factor = BuildOp(
        graph,
        get_guid_with_precision("div_fwd"sv, meta.dtype),
        {batch_size_constant.get(), syn_in(3)},
        {{1, meta.dtype}});

    auto nll_loss = NllLossBwdFunc(
        this,
        graph,
        {syn_in(0), syn_in(2)},
        meta,
        params,
        std::nullopt,
        shapeTnsrSize);

    auto result = BuildOp(
        graph,
        get_guid_with_precision("mult_fwd"sv, meta.dtype),
        {nll_loss.at(0).get(), mul_factor.at(0).get()},
        {{meta.shape, meta.dtype, 0}});

    syn_out(0) = std::move(result[0]);
  } else if (is_weight_none) {
    auto nll_loss = NllLossBwdFunc(
        this, graph, {syn_in(0), syn_in(2)}, meta, params, 0, shapeTnsrSize);
    syn_out(0) = std::move(nll_loss[0]);
  } else { // weight is not none
    auto nll_loss = NllLoss(
        this,
        graph,
        {syn_in(0), syn_in(2), syn_in(3), syn_in(4)},
        meta,
        params,
        0);

    syn_out(0) = std::move(nll_loss[0]);
  }
}

SharedMetaDataVector NllLoss2DBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& grad = stack_tensor(stack, 0);
  const auto& self = stack_tensor(stack, 1);
  const auto& target = stack_tensor(stack, 2);
  const auto& weight =
      stack.at(3).toOptional<torch::Tensor>().value_or(at::Tensor());
  const int64_t reduction = stack.at(4).toInt();
  auto rank = self.dim();
  auto dtype = grad.scalar_type();
  const auto isWeightTensor = weight.defined();
  const std::string guid = "nll_loss_bwd";

  SharedMetaDataVector metaVec;
  SharedMetaData nllLossBwdSharedMeta{"nll_loss_bwd"};
  nllLossBwdSharedMeta.outputs_data.emplace_back(rank, dtype);
  nllLossBwdSharedMeta.inputs_data = {
      {grad.dim(), dtype}, {target.dim(), target.scalar_type()}};
  if (isWeightTensor) {
    nllLossBwdSharedMeta.inputs_data.emplace_back(weight.dim(), dtype);
    nllLossBwdSharedMeta.inputs_data.emplace_back(1, dtype);
  }

  if (!isWeightTensor && reduction == at::Reduction::Reduction::Mean) {
    SharedMetaData divSharedMeta{"div_fwd"};
    divSharedMeta.inputs_data.emplace_back(1, dtype);
    divSharedMeta.inputs_data.emplace_back(1, dtype);
    divSharedMeta.outputs_data.emplace_back(1, dtype);

    SharedMetaData mulSharedMeta{"mult_fwd"};
    mulSharedMeta.inputs_data.emplace_back(rank, dtype);
    mulSharedMeta.inputs_data.emplace_back(1, dtype);
    mulSharedMeta.outputs_data.emplace_back(rank, dtype);
    metaVec.push_back(divSharedMeta);
    metaVec.push_back(mulSharedMeta);
  }

  metaVec.push_back(nllLossBwdSharedMeta);
  return metaVec;
}

void NllLoss2DBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& params = FillParams(stack);
  const auto meta = OutputMeta(stack)[0];
  const auto shapeTnsrSize = NllLossBwdShapeTnsrShape(stack)[0];
  int64_t reduction = stack.at(4).toInt();
  const auto& self = stack_tensor(stack, 1); // self tensor
  int64_t scalar_shape[] = {1};

  // from
  // https://github.com/pytorch/pytorch/blob/4015166e5d51bc39d5a81aa59ad49720ec2a23fe/aten/src/ATen/native/LossNLL2d.cpp#L183C17-L183C24
  auto self_sizes = self.sizes().vec();
  int64_t N = self_sizes[0];
  int64_t HW = self_sizes[2] * self_sizes[3];
  int64_t NHW_multiplier = N * HW;

  // A JIRA is created for self input tensor not used
  // https://jira.habana-labs.com/browse/SW-73878

  std::vector<synapse_helpers::tensor> output;
  if (stack.at(3).isNone()) { // weight is none
    if (reduction == at::Reduction::Reduction::Mean) {
      auto NHW_multiplier_constant = ConstantHelper(
          graph, static_cast<float>(NHW_multiplier), meta.dtype, scalar_shape);

      // mul_factor = batch_size / total_weight
      auto mul_factor = BuildOp(
          graph,
          get_guid_with_precision("div_fwd"sv, meta.dtype),
          {NHW_multiplier_constant.get(), syn_in(3)},
          {{1, meta.dtype}});

      auto nll_loss = NllLossBwdFunc(
          this,
          graph,
          {syn_in(0), syn_in(2)},
          meta,
          params,
          std::nullopt,
          shapeTnsrSize);

      output = BuildOp(
          graph,
          get_guid_with_precision("mult_fwd"sv, meta.dtype),
          {nll_loss.at(0).get(), mul_factor.at(0).get()},
          {{meta.shape, meta.dtype, 0}});
    } else {
      output = NllLossBwdFunc(
          this, graph, {syn_in(0), syn_in(2)}, meta, params, 0, shapeTnsrSize);
    }
  } else { // weight is not none
    output = NllLoss(
        this,
        graph,
        {syn_in(0), syn_in(2), syn_in(3), syn_in(4)},
        meta,
        params,
        0);
  }
  syn_out(0) = std::move(output[0]);
}

} // namespace habana
