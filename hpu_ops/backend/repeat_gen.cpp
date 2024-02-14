/******************************************************************************
 * Copyright (C) 2023-2024 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/repeat.h"
#include "habana_kernels/repeat.h"

namespace habana {

OutputMetaDataVector RepeatMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto repeats = stack.at(1).isTensor() ? stack.at(1).toTensor().sizes().vec()
                                        : stack.at(1).toIntList().vec();

  OutputMetaData meta{};

  meta.dtype = self.scalar_type();
  meta.shape = RepeatOperator::compute_output_shape(self, repeats);

  return {meta};
}

void RepeatOp::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto repeats = stack.at(1).toIntVector();
  int64_t size = repeats.size();
  auto out_shape = RepeatMeta(stack)[0].shape;
  std::vector<synapse_helpers::tensor> reshape_sh_tensor;
  std::vector<synTensor> reshape_syn_tensor;
  // reshape input to same dims as number of entries in repeat array
  if (size > self.ndimension()) {
    auto reshapeSize = RepeatOperator::compute_reshape_output(self, repeats);
    reshape_sh_tensor.emplace_back(
        ReshapeHelper(graph, syn_in(0), reshapeSize, self.scalar_type()));
    reshape_syn_tensor.emplace_back(
        reshape_sh_tensor[reshape_sh_tensor.size() - 1].get());
  } else {
    reshape_syn_tensor.emplace_back(syn_in(0));
  }
  ns_TileKernel::ParamsV2 params{};
  for (int64_t i = 0; i < size; ++i) {
    params.repeat[size - i - 1] = repeats[i];
  }
  auto result = BuildOp(
      graph,
      guid_,
      std::move(reshape_syn_tensor),
      {{out_shape, ScalarType(), 0}},
      &params,
      sizeof(params));
  syn_out(0) = std::move(result[0]);
}
} // namespace habana
