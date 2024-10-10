/******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/_fused_dropout.h"
#include "generated/backend/native_dropout.h"
#include "generated/backend/native_dropout_backward.h"
#include "habana_kernels/random_gen_kernels.h"
#include "hpu_ops/habana_random_ops.h"
#include "hpu_ops/op_backend.h"

namespace sh = synapse_helpers;

namespace habana {
std::vector<synapse_helpers::tensor> DropoutCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::shared_ptr<void> params,
    OutputMetaDataVector metas,
    std::vector<synTensor>& input_tensor,
    size_t size,
    int final_result_index = 0) {
  auto dropout = OpBackend::BuildNode(
      op,
      graph,
      {std::move(get_guid_with_precision("dropout_fwd", metas[0].dtype)),
       input_tensor,
       {NodeAttr::NodeOutputAttr{
            metas[0].shape, metas[0].dtype, final_result_index},
        NodeAttr::NodeOutputAttr{
            metas[1].shape, metas[1].dtype, final_result_index + 1}},
       params.get(),
       size});
  return dropout;
}
std::shared_ptr<void> FillFusedNativeDropoutParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_DropoutKernel::Params);
  auto ratioId = (stack.at(0).isTensor() && stack.at(1).isTensor()) ? 2 : 1 ;
  params->ratio = stack.at(ratioId).toScalar().toDouble();
  return params;
}

OutputMetaDataVector FusedNativeDropoutMeta(const at::Stack& stack) {
  auto selfId = (stack.at(0).isTensor() && stack.at(1).isTensor()) ? 1 : 0;
  at::Tensor self = stack_tensor(stack, selfId);
  auto shape = self.sizes().vec();

  OutputMetaDataVector metas(2);
  metas[0].shape = shape;
  metas[0].dtype = self.scalar_type();
  metas[1].shape = shape;
  metas[1].dtype = at::kChar;

  return metas;
}

OutputMetaDataVector FusedNativeDropoutCheckpointMeta(const at::Stack& stack) {
  auto metas = FusedNativeDropoutMeta(stack);

  return {SeedOutputMeta(), metas[0], metas[1]};
}

void FusedNativeDropout::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto seed = stack.at(2);
  size_t size = 0;
  auto params = FillParams(stack, size);
  auto metas = FusedNativeDropoutMeta(stack);

  std::vector<synTensor> inputTensors = {syn_in(0)};
  if (seed.isTensor())
    inputTensors.push_back(syn_in(1));
  else
    inputTensors.push_back(syn_seed());

  auto dropout = DropoutCommon(this, graph, params, metas, inputTensors, size);
  syn_out(0) = std::move(dropout[0]);
  syn_out(1) = std::move(dropout[1]);
}

SharedMetaDataVector NativeDropoutBackwardSharedMeta(const at::Stack& stack) {
  // It is assumed that constant and cast kernels are handled for all
  // dtypes configuration, so shared layer omits validation.

  auto grad_output = stack_tensor(stack, 0);
  auto grad_dtype = grad_output.scalar_type();
  auto grad_rank = grad_output.dim();

  SharedMetaTensor common_data = {grad_rank, grad_dtype};

  SharedMetaData mul1{};
  mul1.guid = "mult_fwd";
  mul1.inputs_data = {2, common_data};
  mul1.outputs_data = {common_data};

  SharedMetaData mul2{};
  mul2.guid = "mult_fwd";
  mul2.inputs_data = {common_data, {1, grad_dtype}};
  mul2.outputs_data = {common_data};

  return {mul1, mul2};
}

void NativeDropoutBackward::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(stack, "NativeDropoutBackward::AddNode");
  auto grad_output = getNextInput<TensorsPair>(stackGetter);
  auto mask = getNextInput<TensorsPair>(stackGetter);
  auto scale = getNextInput<double>(stackGetter);

  auto grad_dtype = grad_output.pt_t.scalar_type();
  auto mask_dtype = mask.pt_t.scalar_type();

  auto scale_t_storage =
      ConstantHelper(graph, static_cast<float>(scale), grad_dtype, {1});

  auto mask_syn_t = mask.syn_t;
  std::optional<sh::tensor> storage;
  if (mask_dtype != grad_dtype) {
    storage = BuildCast(
        this, graph, mask_syn_t, mask.pt_t.sizes(), mask_dtype, grad_dtype);
    mask_syn_t = storage->get();
  }

  std::string mul_node = get_guid_with_precision("mult_fwd", grad_dtype);
  const auto& grad_sizes = grad_output.pt_t.sizes();

  auto mul1 = BuildOp(
      graph,
      mul_node,
      {grad_output.syn_t, mask_syn_t},
      {{grad_sizes, grad_dtype}});

  auto mul2 = BuildOp(
      graph,
      mul_node,
      {mul1[0].get(), scale_t_storage.get()},
      {{grad_sizes, grad_dtype, 0}});

  syn_out(0) = std::move(mul2[0]);
}

//===----------------------------------------------------------------------===//
// This is the implementation of custom native dropout op in `torch.compile`
//===----------------------------------------------------------------------===//
void HabanaNativeDropoutOp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  auto params = FillFusedNativeDropoutParams(stack, size);
  auto metas = FusedNativeDropoutMeta(stack);

  std::vector<synTensor> inputTensors = {syn_in(1), syn_in(0)};
  auto dropout = DropoutCommon(this, graph, params, metas, inputTensors, size);
  syn_out(0) = std::move(dropout[0]);
  syn_out(1) = std::move(dropout[1]);
}

HabanaNativeDropoutOp::HabanaNativeDropoutOp(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "native_dropout",
          scalar_type,
          {1, 1},
          {},
          {},
          false) {
  SetOutputMetaFn(FusedNativeDropoutMeta);
}

void HabanaNativeDropoutOpCheckpoint::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto seed =
      BuildOp(graph, "identity", {syn_in(0)}, {{{}, at::ScalarType::Int, 0}});
  syn_out(0) = std::move(seed[0]);

  size_t size = 0;
  auto params = FillFusedNativeDropoutParams(stack, size);
  auto metas = FusedNativeDropoutMeta(stack);

  std::vector<synTensor> inputTensors = {syn_in(1), syn_in(0)};
  auto dropout =
      DropoutCommon(this, graph, params, metas, inputTensors, size, 1);
  syn_out(1) = std::move(dropout[0]);
  syn_out(2) = std::move(dropout[1]);
}

HabanaNativeDropoutOpCheckpoint::HabanaNativeDropoutOpCheckpoint(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "native_dropout",
          scalar_type,
          {0, 1, 1},
          {},
          {},
          false) {
  SetOutputMetaFn(FusedNativeDropoutCheckpointMeta);
}
} // namespace habana

static const auto& HabanaRandomKernelRegistry =
    habana::KernelRegistry().REGISTER_RANDOM_OP(
        native_dropout,
        NativeDropoutOp);