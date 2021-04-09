/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "repeat.h"
#include <perf_lib_layer_params.h>

namespace habana {
std::vector<int64_t> RepeatOperator::compute_output_shape(
    const at::Tensor& self,
    at::IntArrayRef repeats) {
  int64_t num_new_dimensions = repeats.size() - self.dim();
  std::vector<int64_t> padded_size(num_new_dimensions, 1);
  padded_size.insert(
      padded_size.end(), self.sizes().begin(), self.sizes().end());
  std::vector<int64_t> outshape(repeats.size());
  for (size_t i = 0; i < repeats.size(); ++i) {
    outshape[i] = padded_size[i] * repeats[i];
  }
  return outshape;
}

void RepeatOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for repeat operator");
  TORCH_CHECK(
      inputs[1].isIntList(),
      "Input arg2 expected to be intlist for repeat operator");
  auto input = inputs[0].toTensor();
  auto repeats = inputs[1].toIntVector();
  ns_TileKernel::ParamsV2 params{};

  int64_t size = repeats.size();
  for (int64_t i = 0; i < size; ++i) {
    params.repeat[size - i - 1] = repeats[i];
  }

  auto output = habana_helpers::createPTTensor(
      input,
      RepeatOperator::compute_output_shape(input, repeats),
      input.options(),
      is_output_persistent);
  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

} // namespace habana
static auto& KernelRegistry = habana::KernelRegistry().add(
    "aten::repeat",
    [](const int device_id, c10::ScalarType scalar_type) {
      return std::make_shared<habana::RepeatOperator>(device_id, scalar_type);
    });
