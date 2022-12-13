/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/flip.h"
#define GUID "reverse_"

namespace habana {

void Flip::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  TORCH_CHECK(
      stack.size() == 2, "Incorrect size of input arguments for Flip Operator");
  TORCH_CHECK(
      stack.at(0).isTensor(),
      "Input arg 1 for Flip op needs to be tensor type");
  TORCH_CHECK(
      stack.at(1).isIntList(), "Input arg 2 for Flip op needs to be Int List");

  auto self = stack.at(0).toTensor();
  auto ndim = self.dim();
  std::vector<int64_t> dim_list = stack.at(1).toIntList().vec();
  auto dim_list_size = dim_list.size();

  const auto outshape = stack_tensor(stack, 0).sizes();

  if (dim_list_size == 0) {
    auto out = OpBackend::BuildOp(
        graph,
        "memcpy_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0)},
        {{outshape, ScalarType(), 0}});
    syn_out(0) = std::move(out[0]);
    return;
  }

  std::vector<synTensor> intermediate_output_itr;
  std::vector<synapse_helpers::tensor> intermediate_output;

  // add syn_input tensor
  intermediate_output_itr.emplace_back(syn_in(0));

  // Converting scalar dims to tensor
  for (size_t i = 0; i < dim_list_size - 1; i++) {
    int flip_axis = at::maybe_wrap_dim(dim_list[i], ndim, true);
    flip_axis = get_dim_in_tpc_order(flip_axis, ndim);

    auto const_dim = ConstantHelper(graph, flip_axis);

    intermediate_output = BuildOp(
        graph,
        GUID + habana_helpers::name_suffix_from_type(ScalarType()),
        {intermediate_output_itr[i], const_dim.get()},
        {{outshape, ScalarType()}});

    intermediate_output_itr.emplace_back(intermediate_output[0].get());
  }
  int final_flip_axis =
      at::maybe_wrap_dim(dim_list[dim_list_size - 1], ndim, true);
  final_flip_axis = get_dim_in_tpc_order(final_flip_axis, ndim);

  auto final_const_dim = ConstantHelper(graph, final_flip_axis);

  auto flip_output = BuildOp(
      graph,
      GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {intermediate_output_itr[dim_list_size - 1], final_const_dim.get()},
      {{outshape, ScalarType(), 0}});

  syn_out(0) = std::move(flip_output[0]);
}
} // namespace habana
