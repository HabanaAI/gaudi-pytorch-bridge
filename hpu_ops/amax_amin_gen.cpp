/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/amax.h"
#include "generated/amin.h"
#include "generated/aminmax.h"
#include "reduction_template.h"

namespace habana {

sizes_vec AminmaxOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto dim = stack.at(1);
  auto is_dim_none = dim.isNone();
  const bool keepdim = stack.at(2).toBool();

  auto dim_vec =
      is_dim_none ? std::vector<int64_t>{} : std::vector<int64_t>{dim.toInt()};

  auto shapes = ReductionOutputShape(self, dim_vec, keepdim);

  return {shapes[0], shapes[0]};
}

sizes_vec AminAmaxOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto dim = stack.at(1);
  auto is_dim_none = dim.isNone();
  const bool keepdim = stack.at(2).toBool();

  auto dim_vec = is_dim_none ? std::vector<int64_t>{} : dim.toIntVector();

  auto shapes = ReductionOutputShape(self, dim_vec, keepdim);

  return {shapes[0]};
}

static std::vector<synapse_helpers::tensor> AminmaxCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const std::vector<synTensor>& input_tensor,
    const at::IntArrayRef output_shape,
    const torch::Tensor& self,
    const std::vector<int64_t>& dim,
    const bool keepdim,
    c10::optional<int> final_idx1 = c10::nullopt,
    c10::optional<int> final_idx2 = c10::nullopt) {
  std::vector<synapse_helpers::tensor> amin_max;
  const auto& dtype_suffix =
      habana_helpers::name_suffix_from_type(op->ScalarType());

  auto amin = HandleReductionDimAndKeepdim(
      op,
      graph,
      self,
      {input_tensor},
      dim,
      keepdim,
      "reduce_min_fwd_" + dtype_suffix,
      {{output_shape, self.scalar_type(), final_idx1},
       {output_shape, self.scalar_type()}});

  auto amax = HandleReductionDimAndKeepdim(
      op,
      graph,
      self,
      {input_tensor},
      dim,
      keepdim,
      "reduce_max_fwd_" + dtype_suffix,
      {{output_shape, self.scalar_type(), final_idx2},
       {output_shape, self.scalar_type()}});

  amin_max.emplace_back(std::move(amin[0]));
  amin_max.emplace_back(std::move(amax[0]));

  return amin_max;
}

void Aminmax::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto is_dim_none = stack.at(1).isNone();
  const bool keepdim = stack.at(2).toBool();

  auto dim = stack.at(1);
  auto dim_vec =
      is_dim_none ? std::vector<int64_t>{} : std::vector<int64_t>{dim.toInt()};

  const auto output_shape = AminmaxOutputShape(stack)[0];

  auto input = syn_in(0);

  if ((self.scalar_type() == torch::kBool) ||
      (self.scalar_type() == torch::kInt8)) {
    auto cast = HandleReductionDtype(this, graph, self, input, torch::kInt32);

    auto amin_max = AminmaxCommon(
        this,
        graph,
        {cast.value().get()},
        output_shape,
        self,
        dim_vec,
        keepdim);

    auto cast1 = CastHelper(
        graph, amin_max[0].get(), output_shape, torch::kInt32, torch::kBool, 0);

    auto cast2 = CastHelper(
        graph, amin_max[1].get(), output_shape, torch::kInt32, torch::kBool, 1);

    syn_out(0) = std::move(cast1);
    syn_out(1) = std::move(cast2);
  } else {
    auto amin_max = AminmaxCommon(
        this, graph, {syn_in(0)}, output_shape, self, dim_vec, keepdim, 0, 1);

    syn_out(0) = std::move(amin_max[0]);
    syn_out(1) = std::move(amin_max[1]);
  }
}

void AminAmax::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto is_dim_none = stack.at(1).isNone();
  const bool keepdim = stack.at(2).toBool();

  auto dim = stack.at(1);
  auto dim_vec = is_dim_none ? std::vector<int64_t>{} : dim.toIntVector();

  const auto output_shape = AminAmaxOutputShape(stack)[0];

  auto op = HandleReductionDimAndKeepdim(
      this,
      graph,
      self,
      {syn_in(0)},
      dim_vec,
      keepdim,
      guid_,
      {{output_shape, ScalarType(), 0}});

  syn_out(0) = std::move(op[0]);
}
} // namespace habana
