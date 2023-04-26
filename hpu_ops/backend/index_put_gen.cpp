/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include "generated/backend/gather.h"
#include "hpu_ops/backend/reduction_template.h"
#include "hpu_ops/common/index.h"
#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/index_put.h"
#include "hpu_ops/topk_util.h"

namespace habana {

// brodcast index tensor shape and get the correct shape and size
static std::vector<int64_t> broadcast_size(at::TensorList indices) {
  auto size = indices[0].sizes().vec();
  for (size_t i = 1; i < indices.size(); i++) {
    size = at::infer_size(size, indices[i].sizes());
  }
  return size;
}

static void validate_cat_tensor_dim_sizes(
    const std::vector<std::vector<int64_t>>* tensors,
    int64_t dim) {
  unsigned i = 0;
  auto tensor_count = tensors->size();
  auto tempT_i = 0;
  for (i = 1; i < tensor_count; i++) {
    // check whether sizes along dimensions match except for cat dimension.
    unsigned j = 0;
    auto sz1 = tensors->at(i);
    auto sz2 = tensors->at(tempT_i);
    for (j = 0; j < tensors->at(i).size(); j++) {
      if (j != dim && (sz1[j] - sz2[j]) != 0) {
        TORCH_CHECK(
            ((sz1[j] - sz2[j]) == 0),
            "Sizes of tensors along one of the non-cat dimensions don't match");
      }
    }
    tempT_i = i;
  }
}

static std::vector<int64_t> CalcCatOutSize(
    const std::vector<std::vector<int64_t>>* tensors,
    int64_t* dim_inp) {
  auto tensor_count = tensors->size();

  if (tensor_count == 0) // if tensor is empty or its first element is empty,
                         // then concatenate out size is 0
    return {0};
  int64_t dim =
      at::maybe_wrap_dim(*dim_inp, tensors->at(0).size(), /*wrap_scalar=*/true);
  validate_cat_tensor_dim_sizes(tensors, *dim_inp);

  if (dim != *dim_inp) {
    *dim_inp = dim;
  }

  // out tensor size should match along all dimensions for input tensors except
  // along the dim in which to cat
  auto out_size = tensors->at(0);
  if (out_size.size() != 0) {
    out_size[dim] = 0;
    for (unsigned i = 0; i < tensor_count; i++) {
      out_size[dim] += tensors->at(i)[dim];
    }
  }
  return out_size;
}

static synapse_helpers::tensor HandleIndexPutWithAcc(
    OpBackend* op,
    synapse_helpers::graph& graph,
    at::Tensor self,
    synapse_helpers::tensor& catop,
    synapse_helpers::tensor& reshape_val_op,
    synTensor syn_in_0,
    int rank_idx,
    c10::ScalarType indices_scalar_type) {
  auto self_scalar_type = self.scalar_type();
  auto scatter_indices_shape = catop.pt_shape();
  // Convert indices to values (ravelling indices) for sorting
  std::vector<int64_t> indices_shape;
  for (int i = 0; i < rank_idx; i++)
    indices_shape.push_back(self.sizes().vec()[i]);
  // Compute multiplication factor for each dimension
  std::vector<int> mul_factor_v{1};
  for (size_t i = 0; i < indices_shape.size() - 1; i++)
    mul_factor_v.push_back(mul_factor_v[i] * indices_shape[i]);

  // auto mul_factor = torch::from_blob(
  //     mul_factor_v.data(), {1, int64_t(mul_factor_v.size())}, torch::kInt);
  // auto multiplied_indices = at::mul(concatenated_indices, mul_factor);
  std::vector<synTensor> cat_input_synTensor;
  std::vector<synapse_helpers::tensor> cat_input_tensor;
  std::vector<std::vector<int64_t>> cat_input_index;
  std::vector<int64_t> const_shape = {1};
  for (size_t i = 0; i < mul_factor_v.size(); i++) {
    cat_input_tensor.emplace_back(OpBackend::BuildConstant(
        op, graph, mul_factor_v[i], indices_scalar_type, const_shape));
    cat_input_synTensor.emplace_back(
        cat_input_tensor[cat_input_tensor.size() - 1].get());
    cat_input_index.emplace_back(
        cat_input_tensor[cat_input_tensor.size() - 1].pt_shape());
  }
  int64_t cat_dim = 0;
  std::vector<int64_t> cat_out_size =
      CalcCatOutSize(&cat_input_index, &cat_dim);
  cat_dim = cat_out_size.size() > 0
      ? (cat_out_size.size() - cat_dim) - 1
      : 0; // if tensor is empty then dim of the concatenated tensor will be 0
  synConcatenateParams concat_params{};
  concat_params.axis = cat_dim;
  auto catop2 = OpBackend::BuildNode(
      op,
      graph,
      {"concat",
       cat_input_synTensor,
       {{cat_out_size, indices_scalar_type}},
       &concat_params,
       sizeof(concat_params)});
  auto catop2_res = std::move(catop2.at(0));
  std::vector<int64_t> reshape_size({1, (int64_t)mul_factor_v.size()});
  auto reshape_ind_op = OpBackend::BuildReshape(
      op, graph, catop2_res.get(), reshape_size, indices_scalar_type);
  auto mulOp = OpBackend::BuildNode(
      op,
      graph,
      {"mult_fwd_" + habana_helpers::name_suffix_from_type(indices_scalar_type),
       {catop.get(), reshape_ind_op.get()},
       {{catop.pt_shape(), indices_scalar_type}}});

  auto red_output_shape = mulOp.at(0).pt_shape();
  int red_dim = 1;
  // red_output_shape.erase(red_output_shape.cbegin()+red_dim);
  red_output_shape[red_dim] = 1;
  ns_Reduction::Params red_params{};
  red_params.reductionDimension =
      get_dim_in_tpc_order(red_dim /*dim*/, red_output_shape.size());
  auto sumop = OpBackend::BuildNode(
      op,
      graph,
      {"reduce_sum_fwd_" +
           habana_helpers::name_suffix_from_type(indices_scalar_type),
       {mulOp.at(0).get()},
       {{red_output_shape, indices_scalar_type}},
       &red_params,
       sizeof(red_params)});
  std::vector<int64_t> reshape_sum({sumop.at(0).pt_shape()[0]});
  auto reshape_sum_op = OpBackend::BuildReshape(
      op, graph, sumop.at(0).get(), reshape_sum, indices_scalar_type);

  auto sortOp = TopK_Helper(
      op,
      graph,
      {reshape_sum_op.get()},
      0,
      reshape_sum_op.pt_shape(),
      0 /*descending order*/,
      reshape_sum_op.pt_shape().size(),
      reshape_sum_op.pt_shape()[0],
      0 /*median vairiant*/);
  auto sort_res0 = std::move(sortOp.at(0));
  auto sort_res1 = std::move(sortOp.at(1));

  // Fill params for gather
  ns_GatherKernel::Params gather_params;
  auto outshape = catop.pt_shape();
  int size1 = outshape.size();
  int size2 = sort_res1.pt_shape().size();
  int gather_dim = 0;
  gather_params.axis = size1 - gather_dim - 1;
  if (size1) {
    // for gather op, output size is same as index
    if (size1 == size2) {
      outshape = sort_res1.pt_shape();
    } else {
      // for index_select and other index ops
      outshape.erase(outshape.begin() + gather_dim);
      auto v = sort_res1.pt_shape();
      auto numel = std::accumulate(
          std::begin(v), std::end(v), 1, std::multiplies<size_t>());
      outshape.insert(outshape.begin() + gather_dim, numel);
    }
  } else {
    TORCH_CHECK(
        size1,
        "Index put op (acc=True case) - gather op output shape cannot be 0");
  }
  auto gatherOp = OpBackend::BuildNode(
      op,
      graph,
      {"gather_fwd_" +
           habana_helpers::name_suffix_from_type(indices_scalar_type),
       {catop.get(), sort_res1.get()},
       {{outshape, indices_scalar_type}},
       &gather_params,
       sizeof(gather_params)});
  std::vector<int64_t> reshape_size2({sort_res1.pt_shape()[0], 1});
  auto reshape_sort1_op = OpBackend::BuildReshape(
      op, graph, sort_res1.get(), reshape_size2, indices_scalar_type);
  ns_ScatterNDKernel::Params scatter_params{int(catop.pt_shape().size()), {0}};
  // Dims reversed between PT and synapse
  for (int i = scatter_indices_shape.size() - 1, j = 0; i >= 0; --i, ++j) {
    scatter_params.origIndicesShape[j] = scatter_indices_shape[i];
  }
  auto scatter_op = OpBackend::BuildNode(
      op,
      graph,
      {"scatter_nd_fwd_" +
           habana_helpers::name_suffix_from_type(self_scalar_type),
       {gatherOp[0].get(), reshape_sort1_op.get(), reshape_val_op.get()},
       {NodeAttr::NodeOutputAttr{self.sizes().vec(), self_scalar_type}},
       &scatter_params,
       sizeof(scatter_params)});
  auto addOp = OpBackend::BuildNode(
      op,
      graph,
      {"add_fwd_" + habana_helpers::name_suffix_from_type(self_scalar_type),
       {syn_in_0, scatter_op.at(0).get()},
       {{self.sizes().vec(), self_scalar_type, 0}}});
  return std::move(addOp[0]);
}

IndexPutEager::IndexPutEager(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, {}, scalar_type, {0}, {}, {}, false) {}

void IndexPutEager::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto indices = stack.at(1).toTensorList().vec();
  auto values = stack_tensor(stack, 2);
  auto accumulate = stack.at(3).toBool();
  auto max_size = broadcast_size(indices);
  auto indices_scalar_type = indices[0].scalar_type();
  std::vector<at::Tensor> cat_input;

  std::vector<synTensor> cat_input_synTensor;
  std::vector<synapse_helpers::tensor> cat_input_tensor;
  std::vector<std::vector<int64_t>> cat_input_index;

  for (size_t i = 0; i < indices.size(); i++) {
    auto bcastOp =
        BroadcastHelper(graph, syn_in(i + 1), max_size, indices_scalar_type);
    // Reshape broadcasted indices to [N, 1] for concatenation
    auto flattened_size = std::accumulate(
        std::begin(max_size), std::end(max_size), 1, std::multiplies<size_t>());

    std::vector<int64_t> expanded_size = {flattened_size, 1};
    cat_input_tensor.emplace_back(ReshapeHelper(
        graph, bcastOp.get(), expanded_size, indices_scalar_type));

    cat_input_synTensor.emplace_back(
        cat_input_tensor[cat_input_tensor.size() - 1].get());
    cat_input_index.emplace_back(
        cat_input_tensor[cat_input_tensor.size() - 1].pt_shape());
  }
  int64_t cat_dim = 1;
  std::vector<int64_t> cat_out_size =
      CalcCatOutSize(&cat_input_index, &cat_dim);
  cat_dim = cat_out_size.size() > 0
      ? (cat_out_size.size() - cat_dim) - 1
      : 0; // if tensor is empty then dim of the concatenated tensor will be 0

  synConcatenateParams concat_params{};
  concat_params.axis = cat_dim;
  auto catop1 = BuildOp(
      graph,
      "concat",
      cat_input_synTensor,
      {{cat_out_size, indices_scalar_type}},
      &concat_params,
      sizeof(concat_params));

  auto catop = std::move(catop1.at(0));

  // Calculate the dimensionality of updates for broadcasting
  auto rank_inp = self.ndimension();
  auto rank_idx = catop.pt_shape()[1];
  std::vector<int64_t> value_upd_dim{catop.pt_shape()[0]};

  if (((int)indices.size() == self.dim()) && (values.numel() > 1)) {
    value_upd_dim.clear();
    value_upd_dim = values.sizes().vec();
  }

  for (int i = rank_idx; i < rank_inp; i++)
    value_upd_dim.push_back(self.sizes().vec()[i]);
  auto values_scalar_type = values.scalar_type();

  auto bcastOp = BroadcastHelper(
      graph, syn_in(1 + indices.size()), value_upd_dim, values_scalar_type);
  auto self_scalar_type = self.scalar_type();
  if ((int)indices.size() == self.dim()) {
    std::vector<int64_t> reshape_bcast_size({catop.pt_shape()[0]});
    auto reshape_val_op = ReshapeHelper(
        graph, bcastOp.get(), reshape_bcast_size, values_scalar_type);
    if (!accumulate) {
      auto scatter_op = BuildOp(
          graph,
          "scatter_nd_onnx_fwd_" +
              habana_helpers::name_suffix_from_type(self_scalar_type),
          {syn_in(0), catop.get(), reshape_val_op.get()},
          {NodeAttr::NodeOutputAttr{self.sizes().vec(), self_scalar_type, 0}});
      syn_out(0) = std::move(scatter_op[0]);
    } else {
      syn_out(0) = HandleIndexPutWithAcc(
          this,
          graph,
          self,
          catop,
          reshape_val_op,
          syn_in(0),
          rank_idx,
          indices_scalar_type);
    }
  } else {
    if (!accumulate) {
      auto scatter_op = BuildOp(
          graph,
          "scatter_nd_onnx_fwd_" +
              habana_helpers::name_suffix_from_type(self_scalar_type),
          {syn_in(0), catop.get(), bcastOp.get()},
          {NodeAttr::NodeOutputAttr{self.sizes().vec(), self_scalar_type, 0}});
      syn_out(0) = std::move(scatter_op[0]);
    } else {
      syn_out(0) = HandleIndexPutWithAcc(
          this,
          graph,
          self,
          catop,
          bcastOp,
          syn_in(0),
          rank_idx,
          indices_scalar_type);
    }
  }
}
} // namespace habana

static const auto& IndexPutKernelRegistry = habana::KernelRegistry().add(
    "hpu::_index_put_impl_eager",
    KERNEL_FN_GLOBAL(habana::IndexPutEager));
