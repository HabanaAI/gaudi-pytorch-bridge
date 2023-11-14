/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/amax.h"
#include "generated/backend/amin.h"
#include "generated/backend/aminmax.h"
#include "hpu_ops/backend/reduction_template.h"

namespace habana {
OutputMetaDataVector AminmaxMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto dim = stack.at(1);
  auto is_dim_none = dim.isNone();
  const bool keepdim = stack.at(2).toBool();

  auto dim_vec =
      is_dim_none ? std::vector<int64_t>{} : std::vector<int64_t>{dim.toInt()};

  auto shapes = ReductionOutputShape(self, dim_vec, keepdim);

  OutputMetaData meta;
  meta.shape = shapes[0];
  meta.dtype = self.scalar_type();
  return {meta, meta};
}

OutputMetaDataVector AminAmaxMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto dim = stack.at(1);
  auto is_dim_none = dim.isNone();
  const bool keepdim = stack.at(2).toBool();

  auto dim_vec = is_dim_none ? std::vector<int64_t>{} : dim.toIntVector();

  auto shapes = ReductionOutputShape(self, dim_vec, keepdim);

  OutputMetaData meta;
  meta.shape = shapes[0];
  meta.dtype = self.scalar_type();
  return {meta};
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

  std::vector<NodeAttr::NodeOutputAttr> amin_output_attrs = {
      {output_shape, self.scalar_type(), final_idx1},
      {output_shape, self.scalar_type()}};
  std::vector<NodeAttr::NodeOutputAttr> amax_output_attrs = {
      {output_shape, self.scalar_type(), final_idx2},
      {output_shape, self.scalar_type()}};

  auto amin = HandleReductionDimAndKeepdim(
      op,
      graph,
      self,
      {input_tensor},
      dim,
      keepdim,
      get_guid_with_precision("reduce_min_fwd", op->ScalarType()),
      amin_output_attrs);

  auto amax = HandleReductionDimAndKeepdim(
      op,
      graph,
      self,
      {input_tensor},
      dim,
      keepdim,
      get_guid_with_precision("reduce_max_fwd", op->ScalarType()),
      amax_output_attrs);

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

  auto output_shape = GetOutputMetaData(0).shape;

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

  const auto meta = OutputMeta(stack)[0];

  auto op = HandleReductionDimAndKeepdim(
      this,
      graph,
      self,
      {syn_in(0)},
      dim_vec,
      keepdim,
      guid_,
      {{meta.shape, meta.dtype, 0}});

  syn_out(0) = std::move(op[0]);
}
} // namespace habana
