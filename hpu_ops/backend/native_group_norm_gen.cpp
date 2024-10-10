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
#include <perf_lib_layer_params.h>
#include "generated/backend/native_group_norm.h"
#include "generated/backend/native_group_norm_backward.h"
#include "hpu_ops/backend/reduction_template.h"
namespace habana {

namespace sh = synapse_helpers;

sizes_vec NativeGroupNormFwdOutputShape(const at::Stack& stack) {
  const auto input_size = stack[0].toTensor().sizes().vec();
  const int N = stack[3].toInt();
  const int G = stack[6].toInt();

  return {input_size, {N, G}, {N, G}};
}

OutputMetaDataVector GroupNormFwdMeta(const at::Stack& stack) {
  constexpr unsigned OUTPUTS_NUMBER = 3;
  auto input = stack_tensor(stack, 0);
  auto shapes = NativeGroupNormFwdOutputShape(stack);
  OutputMetaDataVector metaVec(OUTPUTS_NUMBER);

  for (unsigned i = 0; i < OUTPUTS_NUMBER; ++i) {
    metaVec[i].shape = shapes[i];
    metaVec[i].dtype = input.scalar_type();
  }

  return metaVec;
}

std::shared_ptr<void> FillNativeGroupNormParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_NativeGroupNorm::Params);
  params->N = stack[3].toInt();
  params->G = stack[6].toInt();
  params->epsilon = stack[7].toDouble();

  return params;
}

sizes_vec NativeGroupNormBwdOutputShape(const at::Stack& stack) {
  auto input = stack[1].toTensor();
  auto input_size = input.sizes().vec();

  int weight_size = stack[6].toInt();

  return {input_size, {weight_size}, {weight_size}};
}

OutputMetaDataVector GroupNormBwdMeta(const at::Stack& stack) {
  constexpr unsigned OUTPUTS_NUMBER = 3;
  auto input = stack_tensor(stack, 1);
  auto shapes = NativeGroupNormBwdOutputShape(stack);
  OutputMetaDataVector metaVec(OUTPUTS_NUMBER);

  for (unsigned i = 0; i < OUTPUTS_NUMBER; ++i) {
    metaVec[i].shape = shapes[i];
    metaVec[i].dtype = input.scalar_type();
  }

  return metaVec;
}

static std::shared_ptr<void> FillReduceSumMultiDimParamsGroupNorm(
    size_t ndims,
    size_t& size) {
  PARAMS_STUB(ns_Reduction::ParamsV2);
  params->reductionDimensionMask = getMaskWithBitPosOutInTpcOrder(1, ndims);
  params->keepDim = false;
  return params;
}

std::vector<sh::tensor> GetBnFwdOutGroupNorm(
    OpBackend* op,
    sh::graph& graph,
    synTensor input,
    const std::vector<int64_t>& outputShape,
    const at::ScalarType outputDType,
    const std::vector<int64_t>& constantShape) {
  auto bn_weight_running_var = OpBackend::BuildConstant(
      op, graph, 1.0f, at::ScalarType::Float, constantShape);
  auto bn_bias_running_mean = OpBackend::BuildConstant(
      op, graph, 0.0f, at::ScalarType::Float, constantShape);

  PARAMS_STUB_VARS(ns_BatchNormKernel::ParamsV2, params, size);
  params->momentum = 0.0;
  params->epsilon = 1e-5;
  params->isTraining = true;

  return OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("batch_norm_fwd", outputDType),
       {input,
        bn_bias_running_mean.get(),
        bn_weight_running_var.get(),
        bn_bias_running_mean.get(),
        bn_weight_running_var.get()},
       {NodeAttr::NodeOutputAttr{{outputShape}, outputDType},
        NodeAttr::NodeOutputAttr{constantShape, at::ScalarType::Float},
        NodeAttr::NodeOutputAttr{constantShape, at::ScalarType::Float},
        NodeAttr::NodeOutputAttr{constantShape, at::ScalarType::Float},
        NodeAttr::NodeOutputAttr{constantShape, at::ScalarType::Float}},
       params.get(),
       size});
}

sh::tensor PreprocessMeanRstdInputGroupNorm(
    OpBackend* op,
    sh::graph& graph,
    const at::Tensor& ptTensor,
    synTensor synTensor) {
  const int64_t numel = ptTensor.numel();
  auto tensor_reshaped = OpBackend::BuildReshape(
      op, graph, synTensor, numel, ptTensor.scalar_type());
  return ptTensor.scalar_type() != at::ScalarType::Float
      ? OpBackend::BuildCast(
            op,
            graph,
            tensor_reshaped.get(),
            numel,
            ptTensor.scalar_type(),
            at::ScalarType::Float)
      : std::move(tensor_reshaped);
}

std::vector<int64_t> GetBnInputShapeGroupNorm(
    const at::Tensor& tensor,
    const int64_t Nmod) {
  const int64_t M = tensor.numel() / Nmod;
  const auto dims = tensor.dim();

  std::vector<int64_t> bnInputShape(dims, 1);
  bnInputShape[1] = Nmod;
  bnInputShape[dims - 1] = M;
  return bnInputShape;
}

std::vector<int64_t> GetBnMeanWeightShapeGroupNorm(
    const int64_t dims,
    const int64_t numel) {
  std::vector<int64_t> bn_fwd_mean_shape(dims, 1);
  bn_fwd_mean_shape[1] = numel;
  return bn_fwd_mean_shape;
}

std::vector<int64_t> GetBnMeanShapeGroupNorm(
    const int64_t dims,
    const int64_t Nmod) {
  return GetBnMeanWeightShapeGroupNorm(dims, Nmod);
}

std::vector<int64_t> GetBnWeightShapeGroupNorm(
    const int64_t dims,
    const int64_t C) {
  return GetBnMeanWeightShapeGroupNorm(dims, C);
}

void NativeGroupNormBwdHabanaOperator::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "NativeGroupNormBwdHabanaOperator::AddNode");
  auto grad_out = getNextInput<TensorsPair>(stackGetter);
  auto input = getNextInput<TensorsPair>(stackGetter);
  auto mean = getNextInput<TensorsPair>(stackGetter);
  auto rstd = getNextInput<TensorsPair>(stackGetter);
  auto weightOpt = getNextInput<at::optional<TensorsPair>>(stackGetter);
  auto N = getNextInput<int>(stackGetter);
  auto C = getNextInput<int>(stackGetter);
  auto HxW = getNextInput<int>(stackGetter);
  auto num_groups = getNextInput<int>(stackGetter);
  const auto Nmod = N * num_groups;

  TORCH_CHECK(grad_out.pt_t.numel() == N * C * HxW);
  TORCH_CHECK(input.pt_t.numel() == N * C * HxW);
  TORCH_CHECK(mean.pt_t.numel() == Nmod);
  TORCH_CHECK(rstd.pt_t.numel() == Nmod);

  const bool use_bn_fwd_in_gn_bwd =
      GET_ENV_FLAG_NEW(PT_HPU_USE_BN_FWD_IN_GN_BWD);

  const auto metas = OutputMeta(stack);
  if (input.pt_t.numel() == 0) {
    for (unsigned i = 0; i < metas.size(); i++) {
      auto output =
          BuildOp(graph, "memset", {}, {{metas[i].shape, metas[i].dtype, i}});
      syn_out(i) = std::move(output[0]);
    }
    return;
  }

  const auto bn_input_shape = GetBnInputShapeGroupNorm(input.pt_t, Nmod);
  const auto weight_shape = GetBnWeightShapeGroupNorm(input.pt_t.dim(), C);
  const auto mean_shape = GetBnMeanShapeGroupNorm(input.pt_t.dim(), Nmod);
  const std::vector<int64_t> mean_shape_squeezed{Nmod};

  auto bn_fwd_input =
      ReshapeHelper(graph, input.syn_t, bn_input_shape, metas[0].dtype);

  std::vector<sh::tensor> bn_fwd_out;
  if (use_bn_fwd_in_gn_bwd) {
    bn_fwd_out = GetBnFwdOutGroupNorm(
        this,
        graph,
        bn_fwd_input.get(),
        bn_input_shape,
        metas[0].dtype,
        mean_shape_squeezed);

  } else {
    auto mean_for_bn_fwd =
        ReshapeHelper(graph, mean.syn_t, mean_shape, metas[0].dtype);
    auto rstd_for_bn_fwd =
        ReshapeHelper(graph, rstd.syn_t, mean_shape, metas[0].dtype);

    auto subOp = BuildOp(
        graph,
        get_guid_with_precision("sub_fwd", metas[0].dtype),
        {bn_fwd_input.get(), mean_for_bn_fwd.get()},
        {{bn_input_shape, metas[0].dtype}});
    bn_fwd_out = BuildOp(
        graph,
        get_guid_with_precision("mult_fwd", metas[0].dtype),
        {subOp[0].get(), rstd_for_bn_fwd.get()},
        {{bn_input_shape, metas[0].dtype}});
  }

  const auto bn_fwd_out_reshaped = ReshapeHelper(
      graph, bn_fwd_out[0].get(), input.pt_t.sizes().vec(), metas[0].dtype);

  const auto weightReshaped = weightOpt
      ? ReshapeHelper(graph, weightOpt->syn_t, weight_shape, metas[1].dtype)
      : ConstantHelper(graph, 1.0f, metas[1].dtype, weight_shape);

  auto grad_weight_mult = BuildOp(
      graph,
      get_guid_with_precision("mult_fwd", metas[0].dtype),
      {grad_out.syn_t, weightReshaped.get()},
      {{metas[0].shape, metas[0].dtype}});

  auto grad_weight_mult_reshape = ReshapeHelper(
      graph, grad_weight_mult[0].get(), bn_input_shape, metas[0].dtype);

  const auto bn_bwd_grad =
      grad_out.pt_t.scalar_type() != input.pt_t.scalar_type()
      ? BuildCast(
            this,
            graph,
            grad_weight_mult_reshape.get(),
            bn_input_shape,
            grad_out.pt_t.scalar_type(),
            input.pt_t.scalar_type())
      : std::move(grad_weight_mult_reshape);

  auto bn_bwd_in =
      ReshapeHelper(graph, input.syn_t, bn_input_shape, metas[0].dtype);
  auto bn_gamma =
      ConstantHelper(graph, 1.0f, at::ScalarType::Float, mean_shape_squeezed);

  auto mean_casted =
      PreprocessMeanRstdInputGroupNorm(this, graph, mean.pt_t, mean.syn_t);
  auto rstd_casted =
      PreprocessMeanRstdInputGroupNorm(this, graph, rstd.pt_t, rstd.syn_t);

  PARAMS_STUB_VARS(ns_BatchNormKernel::ParamsV2, params, size);
  params->epsilon = 1e-05;
  params->isTraining = true;
  params->momentum = 0;
  auto bn_bwd = BuildOp(
      graph,
      get_guid_with_precision("batch_norm_bwd", metas[0].dtype),
      {bn_bwd_in.get(),
       bn_bwd_grad.get(),
       mean_casted.get(),
       rstd_casted.get(),
       bn_gamma.get()},
      {{bn_input_shape, metas[0].dtype},
       {mean_shape_squeezed, at::ScalarType::Float},
       {mean_shape_squeezed, at::ScalarType::Float}},
      params.get(),
      size);

  auto output =
      ReshapeHelper(graph, bn_bwd[0].get(), metas[0].shape, metas[0].dtype, 0);

  size_t reduce_size = 0;
  auto reduce_params =
      FillReduceSumMultiDimParamsGroupNorm(input.pt_t.dim(), reduce_size);

  // WEIGHT
  auto grad_bn_fwd_mult = BuildOp(
      graph,
      get_guid_with_precision("mult_fwd", metas[0].dtype),
      {grad_out.syn_t, bn_fwd_out_reshaped.get()},
      {{metas[0].shape, metas[0].dtype}});

  auto grad_gamma = BuildOp(
      graph,
      get_guid_with_precision("reduce_sum_multi_dim_fwd", metas[2].dtype),
      {grad_bn_fwd_mult[0].get()},
      {{metas[1].shape, metas[1].dtype, 1}},
      reduce_params.get(),
      reduce_size);

  // BIAS
  auto grad_beta = BuildOp(
      graph,
      get_guid_with_precision("reduce_sum_multi_dim_fwd", metas[2].dtype),
      {grad_out.syn_t},
      {{metas[2].shape, metas[2].dtype, 2}},
      reduce_params.get(),
      reduce_size);

  syn_out(0) = std::move(output);
  syn_out(1) = std::move(grad_gamma[0]);
  syn_out(2) = std::move(grad_beta[0]);
}

void NativeGroupNormFwd::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(stack, "NativeGroupNormFwd::AddNode");
  auto input = getNextInput<TensorsPair>(stackGetter);
  auto weight = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto bias = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto metas = OutputMeta(stack);
  size_t size = 0;
  auto params = FillParams(stack, size);
  auto outputsNumber = metas.size();
  if (input.pt_t.numel() == 0) {
    for (unsigned i = 0; i < outputsNumber; i++) {
      auto output =
          BuildOp(graph, "memset", {}, {{metas[i].shape, metas[i].dtype, i}});
      syn_out(i) = std::move(output[0]);
    }
  } else {
    const auto rank = input.pt_t.dim();
    auto layout = [rank]() {
      if (rank == 3) {
        return synapse_helpers::layouts::SynapseLayoutFormat::WCN;
      } else if (rank == 4) {
        return synapse_helpers::layouts::SynapseLayoutFormat::WHCN;
      } else {
        return synapse_helpers::layouts::SynapseLayoutFormat::WHDCN;
      }
    }();

    SetSynapseLayouts(
        {layout,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE},
        {layout,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});

    auto outputs = BuildOp(
        graph,
        GetGuid(),
        {input.syn_t,
         weight.has_value() ? weight.value().syn_t : nullptr,
         bias.has_value() ? bias.value().syn_t : nullptr},
        {{metas[0].shape, metas[0].dtype, 0},
         {metas[1].shape, metas[1].dtype, 1},
         {metas[2].shape, metas[2].dtype, 2}},
        params.get(),
        size);
    for (unsigned i = 0; i < outputsNumber; i++) {
      syn_out(i) = std::move(outputs[i]);
    }
  }
}

} // namespace habana
