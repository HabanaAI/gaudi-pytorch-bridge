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

SharedMetaDataVector UnaryForeachErfcSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  auto tensors = stack.at(0).toTensorList();
  auto tensorsSize = tensors.size();
  SharedMetaDataVector metaVec;
  const int numberOfKernelPerIteration = 3;
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

    if (rank > 1) {
      SharedMetaData constantSharedMeta{"constant"};
      constantSharedMeta.outputs_data.emplace_back(rank, inputType);
      metaVec.push_back(erfMeta);
    }

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
