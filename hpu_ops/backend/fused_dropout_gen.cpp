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

#include "generated/backend/_fused_dropout.h"
#include "generated/backend/native_dropout.h"
#include "generated/backend/native_dropout_backward.h"
#include "habana_kernels/random_gen_kernels.h"
#include "hpu_ops/op_backend.h"

namespace sh = synapse_helpers;

namespace habana {
std::shared_ptr<void> FillFusedNativeDropoutParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_DropoutKernel::Params);
  params->ratio = stack.at(1).toScalar().toDouble();
  return params;
}

OutputMetaDataVector FusedNativeDropoutMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto shape = self.sizes().vec();

  OutputMetaDataVector metas(2);
  metas[0].shape = shape;
  metas[0].dtype = self.scalar_type();
  metas[1].shape = shape;
  metas[1].dtype = at::kChar;

  return metas;
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

  auto dropout = BuildOp(
      graph,
      get_guid_with_precision("dropout_fwd", metas[0].dtype),
      std::move(inputTensors),
      {NodeAttr::NodeOutputAttr{metas[0].shape, metas[0].dtype, 0},
       NodeAttr::NodeOutputAttr{metas[1].shape, metas[1].dtype, 1}},
      params.get(),
      size);

  syn_out(0) = std::move(dropout[0]);
  syn_out(1) = std::move(dropout[1]);
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

} // namespace habana
