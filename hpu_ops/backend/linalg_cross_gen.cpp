/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/linalg_cross.h"

namespace habana {

static sizes_vec SplitOutputShape(
    bool,
    int64_t dim,
    std::vector<int64_t> outshape) {
  TORCH_CHECK(
      outshape[dim] == 3,
      "LinAlgCross: dimension ",
      dim,
      " does not have size 3");
  auto split_shape = outshape;
  split_shape[dim] = split_shape[dim] / 3;
  return {split_shape};
}

static sizes_vec TransposeOutputShape(
    const at::Stack& stack,
    bool,
    int64_t trans_dim1,
    int64_t trans_dim2) {
  auto self = stack.at(0).toTensor();
  std::vector<int64_t> output_size = self.sizes().vec();

  // syn_dim1, syn_dim2 - corresponding dimensions to swap in synapse order
  auto syn_dim1 = get_dim_in_tpc_order(trans_dim1, self.dim());
  auto syn_dim2 = get_dim_in_tpc_order(trans_dim2, self.dim());
  int tmp;

  tmp = output_size.at(syn_dim1);
  output_size.at(syn_dim1) = output_size[syn_dim2];
  output_size.at(syn_dim2) = tmp;
  return {output_size};
}
static std::vector<synapse_helpers::tensor> Transpose(
    OpBackend* op,
    synapse_helpers::graph& graph,
    int64_t dim,
    int64_t trans_dim1,
    int64_t trans_dim2,
    std::vector<synTensor> input,
    const at::IntArrayRef outshape,
    bool is_persistent) {
  // Transpose Params
  synTransposeParams trans_params{};
  trans_params.tensorDim = dim;
  for (int i = 0; i < dim; ++i) {
    trans_params.permutation[i] = static_cast<TransposePermutationDim>(i);
  }
  std::swap(
      trans_params.permutation[trans_dim1],
      trans_params.permutation[trans_dim2]);

  return OpBackend::BuildNode(
      op,
      graph,
      {"transpose",
       std::move(input),
       {{outshape,
         op->ScalarType(),
         is_persistent ? c10::make_optional<int>(0) : c10::nullopt}},
       &trans_params,
       sizeof(trans_params)});
}
static std::vector<synapse_helpers::tensor> Split(
    OpBackend* op,
    synapse_helpers::graph& graph,
    int64_t split_axis,
    std::vector<synTensor> input,
    const at::IntArrayRef outshape) {
  synAxisParams split_params{};
  split_params.axis = split_axis;

  return OpBackend::BuildNode(
      op,
      graph,
      {"split",
       std::move(input),
       {{outshape, op->ScalarType()},
        {outshape, op->ScalarType()},
        {outshape, op->ScalarType()}},
       &split_params,
       sizeof(split_params)});
}
static std::vector<synapse_helpers::tensor> Mul(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> inputs,
    const at::IntArrayRef outshape) {
  return OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("mult_fwd", op->ScalarType()),
       std::move(inputs),
       {{outshape, op->ScalarType()}}});
}
static std::vector<synapse_helpers::tensor> Concat(
    OpBackend* op,
    synapse_helpers::graph& graph,
    int64_t concat_axis,
    std::vector<synTensor> inputs,
    const at::IntArrayRef outshape) {
  synConcatenateParams concat_params{};
  concat_params.axis = concat_axis;

  return OpBackend::BuildNode(
      op,
      graph,
      {"concat",
       std::move(inputs),
       {{outshape, op->ScalarType()}},
       &concat_params,
       sizeof(concat_params)});
}
void LinAlgCross::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto other = stack.at(1).toTensor();
  auto outshape = self.sizes().vec();
  auto ndim = self.dim();
  int64_t dim_axis = -1;

  TORCH_CHECK(
      self.scalar_type() == other.scalar_type(),
      "LinAlgCross: Tensor must have same dtype, but got ",
      self.scalar_type(),
      "and ",
      other.scalar_type(),
      "dtype tensors");
  TORCH_CHECK(
      self.sizes().vec() == other.sizes().vec(),
      "LinAlgCross: Tensor must have same shape, but got ",
      self.sizes().vec(),
      "and ",
      other.sizes().vec(),
      "shape tensors");

  // If dim not provided, takes the very first dimension of outshape where 3
  // presents
  if (stack.at(2).isNone()) {
    for (int64_t i = 0; i < self.dim(); i++) {
      if (outshape[i] == 3) {
        dim_axis = i;
        break;
      }
    }
    TORCH_CHECK(dim_axis >= 0, "LinAlgCross: no dimension of size 3 in input");
  } else {
    dim_axis = stack.at(2).toInt();
    dim_axis =
        dim_axis >= 0 ? dim_axis : stack.at(0).toTensor().dim() + dim_axis;
  }

  int64_t dim =
      (dim_axis >= 0) ? dim_axis : stack.at(0).toTensor().dim() + dim_axis;

  auto syn_dim =
      get_dim_in_tpc_order(dim, self.dim()); // Converting dim to synapse order
  bool is_scd =
      (syn_dim ==
       self.dim() - 1); // Checking whether syndim is SCD (last dimension)

  std::vector<synapse_helpers::tensor> transpose_input1, transpose_input2;

  // Permutation dimensions for transpose
  auto trans_dim1 = syn_dim;
  auto trans_dim2 = ndim - 1;

  auto transpose_shape =
      TransposeOutputShape(stack, true, trans_dim1, trans_dim2)[0];

  // If not scd, transpose the dimensions to convert to scd
  if (!is_scd) {
    // Transpose the first input
    transpose_input1 = Transpose(
        this,
        graph,
        ndim,
        trans_dim1,
        trans_dim2,
        {syn_in(0)},
        {transpose_shape},
        false);

    // Transpose the second input
    transpose_input2 = Transpose(
        this,
        graph,
        ndim,
        trans_dim1,
        trans_dim2,
        {syn_in(1)},
        {transpose_shape},
        false);
  }

  // Finding index where 3 is present
  auto index = std::find(transpose_shape.begin(), transpose_shape.end(), 3);
  int index_position = index - transpose_shape.begin();

  auto split_shape = SplitOutputShape(
      true,
      is_scd ? dim : index_position,
      is_scd ? outshape : transpose_shape)[0];

  // converting index position to synapse order
  auto axis = get_dim_in_tpc_order(index_position, self.dim());

  // Split Params
  int64_t split_axis;
  std::vector<synTensor> split_input, split_input2;

  if (is_scd) {
    split_axis = syn_dim;
    split_input = {syn_in(0)};
    split_input2 = {syn_in(1)};
  } else {
    split_axis = axis;
    split_input = {transpose_input1[0].get()};
    split_input2 = {transpose_input2[0].get()};
  }

  // Split first tensor to 3 components
  auto split1 = Split(this, graph, split_axis, {split_input}, {split_shape});

  // Split second tensor to 3 components
  auto split2 = Split(this, graph, split_axis, {split_input2}, {split_shape});

  // Multiply corresponding results of split
  auto aybz =
      Mul(this, graph, {split1[1].get(), split2[2].get()}, {split_shape});
  auto azby =
      Mul(this, graph, {split1[2].get(), split2[1].get()}, {split_shape});
  auto azbx =
      Mul(this, graph, {split1[2].get(), split2[0].get()}, {split_shape});
  auto axbz =
      Mul(this, graph, {split1[0].get(), split2[2].get()}, {split_shape});
  auto axby =
      Mul(this, graph, {split1[0].get(), split2[1].get()}, {split_shape});
  auto aybx =
      Mul(this, graph, {split1[1].get(), split2[0].get()}, {split_shape});

  // Concat Params
  auto concat_axis = is_scd ? syn_dim : axis;
  auto concat_shape = is_scd ? outshape : transpose_shape;

  // Concat aybz, azbx, axby
  auto concat1 = Concat(
      this,
      graph,
      concat_axis,
      {aybz[0].get(), azbx[0].get(), axby[0].get()},
      {concat_shape});

  // Concat azby, axbz, aybx
  auto concat2 = Concat(
      this,
      graph,
      concat_axis,
      {azby[0].get(), axbz[0].get(), aybx[0].get()},
      {concat_shape});

  auto sub = BuildOp(
      graph,
      get_guid_with_precision("sub", ScalarType()),
      {concat1[0].get(), concat2[0].get()},
      {{is_scd ? outshape : transpose_shape,
        ScalarType(),
        is_scd ? c10::make_optional<int>(0) : c10::nullopt}});

  // if syn_dim is scd, move the output of sub
  if (is_scd) {
    syn_out(0) = std::move(sub[0]);
    return;
  }
  // else transpose the output of sub
  auto transpose_output = Transpose(
      this,
      graph,
      self.dim(),
      trans_dim1,
      trans_dim2,
      {sub[0].get()},
      {outshape},
      true);
  syn_out(0) = std::move(transpose_output[0]);
}
} // namespace habana
