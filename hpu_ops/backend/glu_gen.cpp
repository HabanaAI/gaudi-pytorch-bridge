/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/glu.h"
#include "generated/backend/glu_backward.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

sizes_vec GluOutputShape(const at::Stack& stack) {
  auto out_shape = stack.at(0).toTensor().sizes().vec();
  const int64_t axis = stack.at(1).toInt();
  auto dim = (axis >= 0) ? axis : stack.at(0).toTensor().dim() + axis;
  out_shape[dim] = out_shape[dim] / 2;
  return {out_shape};
}

sizes_vec GluBwdOutputShape(const at::Stack& stack) {
  auto out_shape = stack.at(1).toTensor().sizes().vec();
  return {out_shape};
}

void Glu::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto in_shape = self.sizes().vec();
  const int64_t axis = stack.at(1).toInt();
  auto split_idx = (axis >= 0) ? axis : self.dim() + axis;

  TORCH_CHECK(
      self.dim() > 0,
      "Glu: Input tensor should be having dimension higher than 0, but got size ",
      self.dim());
  TORCH_CHECK(
      in_shape[split_idx] % 2 == 0, "Glu: Halving dimension must be even");

  auto outshape = GluOutputShape(stack)[0];

  synAxisParams split_params{};
  split_params.axis = self.dim() - 1 - split_idx;

  auto split = BuildOp(
      graph,
      "split",
      {syn_in(0)},
      {{outshape, ScalarType()}, {outshape, ScalarType()}},
      &split_params,
      sizeof(split_params));

  auto sigmoid = BuildOp(
      graph,
      "sigmoid_fwd_" +
          habana_helpers::name_suffix_from_type(self.scalar_type()),
      {split[1].get()},
      {{outshape, ScalarType()}});

  auto mult = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(self.scalar_type()),
      {split[0].get(), sigmoid[0].get()},
      {{outshape, ScalarType(), 0}});

  syn_out(0) = std::move(mult[0]);
}

void GluBwd::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(1).toTensor();
  auto outshape = self.sizes().vec();
  const int64_t axis = stack.at(2).toInt();
  auto split_idx = (axis >= 0) ? axis : self.dim() + axis;

  TORCH_CHECK(
      self.dim() > 0,
      "Glu: Input tensor should be having dimension higher than 0, but got size ",
      self.dim());
  TORCH_CHECK(
      outshape[split_idx] % 2 == 0, "Glu: Halving dimension must be even");

  auto dim = self.dim() - 1 - split_idx;
  outshape[split_idx] = outshape[split_idx] / 2;

  synAxisParams split_params{};
  split_params.axis = dim;

  auto split = BuildOp(
      graph,
      "split",
      {syn_in(1)},
      {{outshape, ScalarType()}, {outshape, ScalarType()}},
      &split_params,
      sizeof(split_params));

  auto sigmoid = BuildOp(
      graph,
      "sigmoid_fwd_" +
          habana_helpers::name_suffix_from_type(self.scalar_type()),
      {split[1].get()},
      {{outshape, ScalarType()}});

  auto grad_in1 = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(self.scalar_type()),
      {syn_in(0), sigmoid[0].get()},
      {{outshape, ScalarType()}});

  auto t1 = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(self.scalar_type()),
      {split[0].get(), grad_in1[0].get()},
      {{outshape, ScalarType()}});

  auto t2 = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(self.scalar_type()),
      {t1[0].get(), sigmoid[0].get()},
      {{outshape, ScalarType()}});

  auto grad_in2 = BuildOp(
      graph,
      "sub_" + habana_helpers::name_suffix_from_type(self.scalar_type()),
      {t1[0].get(), t2[0].get()},
      {{outshape, ScalarType()}});

  auto out_shape = GluBwdOutputShape(stack);
  synConcatenateParams concat_params{};
  concat_params.axis = dim;

  auto concat = BuildOp(
      graph,
      "concat",
      {grad_in1[0].get(), grad_in2[0].get()},
      {{out_shape[0], ScalarType(), 0}},
      &concat_params,
      sizeof(concat_params));

  syn_out(0) = std::move(concat[0]);
}
} // namespace habana
