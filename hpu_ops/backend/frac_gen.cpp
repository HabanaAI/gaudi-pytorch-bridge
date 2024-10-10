/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/_foreach_frac.h"
#include "generated/backend/frac.h"

namespace habana {
static auto BuildFrac(
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor input,
    at::ScalarType dtype,
    at::IntArrayRef outshape,
    int out_index) {
  // sign on input 0
  auto sign = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("sign_fwd", dtype),
       {input},
       {{outshape, dtype}}});

  // abs on output of sign -> modulus
  auto abs_val = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("abs_fwd", dtype),
       {input},
       {{outshape, dtype}}});

  // floor on output of mod
  auto floor_val = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("floor_fwd", dtype),
       {abs_val[0].get()},
       {{outshape, dtype}}});

  // mul on output of floor & sign
  auto mul = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("mult", dtype),
       {floor_val[0].get(), sign[0].get()},
       {{outshape, dtype}}});

  // sub on input & output of mul
  return OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("sub", dtype),
       {input, mul[0].get()},
       {{outshape, dtype, out_index}}});
}

SharedMetaDataVector FracSharedMeta(const at::Stack& stack) {
  auto tensor = stack.at(0).toTensor();
  SharedMetaDataVector metaVec;
  metaVec.reserve(5);
  SharedMetaData sharedMeta;

  auto rank = tensor.dim();
  auto dtype = tensor.scalar_type();

  SharedMetaTensor inOutMetaTensor{rank, dtype};
  sharedMeta.guid = "sign_fwd";
  sharedMeta.inputs_data = {inOutMetaTensor};
  sharedMeta.outputs_data = {inOutMetaTensor};
  metaVec.push_back(sharedMeta);

  sharedMeta.guid = "abs_fwd";
  metaVec.push_back(sharedMeta);

  if (isIntegralType(dtype, true))
    inOutMetaTensor.second = torch::kFloat32;

  sharedMeta.guid = "floor_fwd";
  metaVec.push_back(sharedMeta);

  sharedMeta.guid = "mult";
  sharedMeta.inputs_data = {inOutMetaTensor, inOutMetaTensor};
  metaVec.push_back(sharedMeta);

  sharedMeta.guid = "sub";
  metaVec.push_back(sharedMeta);
  return metaVec;
}

SharedMetaDataVector UnaryForeachFracSharedMeta(const at::Stack& stack) {
  auto tensors = stack.at(0).toTensorList();
  auto tensorsSize = tensors.size();
  SharedMetaDataVector metaVec;
  const int numberOfKernelPerIteration = 5;
  metaVec.reserve(tensorsSize * numberOfKernelPerIteration);
  SharedMetaData sharedMeta;
  for (size_t i = 0; i < tensorsSize; i++) {
    at::Stack fracStack = {c10::IValue(tensors[i])};
    auto fracSharedMetaVec = FracSharedMeta(fracStack);
    metaVec.insert(
        std::end(metaVec),
        std::begin(fracSharedMetaVec),
        std::end(fracSharedMetaVec));
  }
  return metaVec;
}

void Frac::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();
  auto out = BuildFrac(this, graph, syn_in(0), ScalarType(), outshape, 0);
  syn_out(0) = std::move(out[0]);
}

void ForeachFrac::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& tensors = stack[0].toTensorList();
  for (auto i = 0u; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    auto out = BuildFrac(
        this, graph, syn_in(i), tensor.scalar_type(), tensor.sizes(), i);
    syn_out(i) = std::move(out[0]);
  }
}
} // namespace habana
