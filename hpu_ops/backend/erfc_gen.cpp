/******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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
#include "generated/backend/_foreach_erfc.h"
#include "generated/backend/erfc.h"

namespace habana {

static auto BuildErfc(
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor input,
    at::ScalarType dtype,
    at::IntArrayRef outshape,
    int out_index) {
  auto erf = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("erf_fwd", dtype),
       {input},
       {{outshape, dtype}}});

  auto constant = OpBackend::BuildConstant(op, graph, 1, dtype, outshape);

  return OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("sub", dtype),
       {constant.get(), erf[0].get()},
       {{outshape, dtype, out_index}}});
}

SharedMetaDataVector UnaryForeachErfcSharedMeta(const at::Stack& stack) {
  auto tensors = stack.at(0).toTensorList();
  auto tensorsSize = tensors.size();
  SharedMetaDataVector metaVec;
  const int numberOfKernelPerIteration = 2;
  metaVec.reserve(tensorsSize * numberOfKernelPerIteration);
  for (size_t i = 0; i < tensorsSize; i++) {
    const at::Tensor& tensor = tensors[i];
    auto rank = tensor.dim();
    auto inputType = tensor.scalar_type();
    inputType = inputType != torch::kBFloat16 ? torch::kFloat32 : inputType;
    auto outputType = inputType;

    SharedMetaTensor outputTensor{rank, outputType};
    SharedMetaData erfMeta{"erf_fwd"};
    erfMeta.inputs_data = {{rank, inputType}};
    erfMeta.outputs_data = {outputTensor};
    metaVec.push_back(erfMeta);

    SharedMetaData subMeta{"sub_fwd"};
    subMeta.inputs_data = {outputTensor, outputTensor};
    subMeta.outputs_data = {outputTensor};
    metaVec.push_back(subMeta);
  }
  return metaVec;
}

void ForeachErfc::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& tensors = stack[0].toTensorList();
  for (auto i = 0u; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    const at::ScalarType scalar_type = tensor.scalar_type() != torch::kBFloat16
        ? torch::kFloat32
        : torch::kBFloat16;
    auto out =
        BuildErfc(this, graph, syn_in(i), scalar_type, tensor.sizes(), i);
    syn_out(i) = std::move(out[0]);
  }
}
} // namespace habana
