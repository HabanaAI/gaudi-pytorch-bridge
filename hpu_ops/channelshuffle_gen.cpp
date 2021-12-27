/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/hpu_op.h"
#include "hpu_op_helper.h"

namespace habana {
void ChannelShuffle::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  const at::Tensor self = stack_tensor(stack, 0);

  TORCH_CHECK(self.dim() > 2, "channel shuffle expects input with > 2 dim");

  auto outshape = self.sizes();
  int groups = stack[1].toScalar().to<int64_t>();
  int batch = outshape[0];
  auto nchannels = outshape[1];
  int ochannels = nchannels / groups;
  int remaining_elements = 1;
  TORCH_CHECK(
      groups > 0,
      "channel shuffle expects number of groups should be positive");

  TORCH_CHECK(
      (nchannels % groups == 0),
      "channel shuffle expects number of channels to be divisible by groups");

  for (auto i = 0; i < self.dim(); ++i) {
    remaining_elements *= outshape[i];
  }
  remaining_elements = remaining_elements / (batch * groups * ochannels);

  std::vector<int64_t> reshaped = {
      batch, groups, ochannels, remaining_elements};
  std::vector<int64_t> reshaped1 = {
      batch, ochannels, groups, remaining_elements};

  std::vector<synTensor> vectSynTensor{syn_in(0)};
  synTransposeParams trans_params{};
  trans_params.tensorDim = reshaped.size();
  for (int i = 0; i < int(reshaped.size()); ++i) {
    trans_params.permutation[i] = static_cast<TransposePermutationDim>(i);
  }
  std::swap(trans_params.permutation[1], trans_params.permutation[2]);
  auto inp_reshaped =
      BuildOp(graph, "reshape", {syn_in(0)}, {{{reshaped}, ScalarType()}});

  auto transpose = BuildOp(
      graph,
      "transpose",
      {inp_reshaped[0].get()},
      {{{reshaped1}, ScalarType()}},
      &trans_params,
      sizeof(trans_params));

  auto output_tensor = BuildOp(
      graph,
      "reshape",
      {transpose[0].get()},
      {{outshape, ScalarType(), is_output_persistent_list[0], true}});

  syn_out(0) = std::move(output_tensor[0]);
}
} // namespace habana
