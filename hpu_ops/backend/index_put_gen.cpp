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
#include <ATen/core/DimVector.h>
#include "hpu_ops/backend/nonzero.h"
#include "hpu_ops/index_put.h"
#include "hpu_ops/topk_util.h"

namespace habana {

// brodcast index tensor shape and get the correct shape and size
static std::vector<int64_t> broadcast_size(
    at::TensorList indices,
    at::Tensor self) {
  std::vector<int64_t> size;
  if ((indices.size() == 1) && (indices[0].sizes().size() == 0)) {
    return size;
  }
  auto isz = indices[0].sizes().vec();
  auto self_sizes = self.sizes().vec();
  if ((indices[0].scalar_type() == c10::ScalarType::Bool)) {
    std::vector<int64_t> sz{isz[0]}; // if index is 2-D (for bool), number of
                                     // rows indicates broadcast size
    size = sz;
  } else {
    size = isz;
  }
  for (size_t i = 1; i < indices.size(); i++) {
    size = at::infer_size(size, indices[i].sizes());
  }
  if ((indices[0].scalar_type() == c10::ScalarType::Bool) &&
      (int)size.size() < self.dim()) {
    size = self_sizes;
  }
  return size;
}

static void validate_cat_tensor_dim_sizes(
    const std::vector<std::vector<int64_t>>* tensors,
    int64_t dim) {
  size_t i = 0;
  auto tensor_count = tensors->size();
  size_t tempT_i = 0;
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
  int64_t dim = at::maybe_wrap_dim(
      *dim_inp,
      static_cast<int64_t>(tensors->at(0).size()),
      /*wrap_scalar=*/true);
  validate_cat_tensor_dim_sizes(tensors, *dim_inp);

  if (dim != *dim_inp) {
    *dim_inp = dim;
  }

  // out tensor size should match along all dimensions for input tensors except
  // along the dim in which to cat
  auto out_size = tensors->at(0);
  if (out_size.size() != 0) {
    out_size[static_cast<size_t>(dim)] = 0;
    for (unsigned i = 0; i < tensor_count; i++) {
      out_size[static_cast<size_t>(dim)] +=
          tensors->at(i)[static_cast<size_t>(dim)];
    }
  }
  return out_size;
}

static bool CheckAndGetCastGuid(
    std::string guid,
    at::ScalarType dtype,
    std::string& cast_guid,
    at::ScalarType& cast_dtype) {
  cast_guid = "";
  switch (dtype) {
    case at::ScalarType::Short:
      if (guid == "scatter_nd_fwd" || guid == "scatter_nd_onnx_fwd") {
        cast_guid = "cast_i32_to_i16";
        cast_dtype = at::ScalarType::Int;
        return true;
      }
      break;

    case at::ScalarType::Byte:
      if (guid == "scatter_nd_fwd") {
        cast_guid = "cast_i32_to_u8";
        cast_dtype = at::ScalarType::Int;
        return true;
      }
      break;
    default:
      break;
  }
  return false;
}

static synapse_helpers::tensor HandleIndexPutWithAcc(
    OpBackend* op,
    synapse_helpers::graph& graph,
    at::Tensor self,
    synapse_helpers::tensor& catop,
    synapse_helpers::tensor& reshape_val_op,
    synTensor syn_in_0,
    size_t rank_idx,
    c10::ScalarType indices_scalar_type) {
  auto self_scalar_type = self.scalar_type();
  auto scatter_indices_shape = catop.pt_shape();
  // Convert indices to values (ravelling indices) for sorting
  std::vector<int64_t> indices_shape;
  for (size_t i = 0; i < rank_idx; ++i)
    indices_shape.push_back(self.sizes().vec()[i]);
  // Compute multiplication factor for each dimension
  std::vector<int64_t> mul_factor_v{1};
  for (size_t i = 0; i < indices_shape.size() - 1; i++)
    mul_factor_v.push_back(mul_factor_v[i] * indices_shape[i]);

  // auto mul_factor = torch::from_blob(
  //     mul_factor_v.data(), {1, int64_t(mul_factor_v.size())}, torch::kInt);
  // auto multiplied_indices = at::mul(concatenated_indices, mul_factor);
  std::vector<synTensor> cat_input_synTensor;
  std::vector<synapse_helpers::tensor> cat_input_tensor;
  std::vector<std::vector<int64_t>> cat_input_index;
  std::vector<int64_t> const_shape = {1};
  for (size_t i = 0; i < mul_factor_v.size(); ++i) {
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
      ? (static_cast<int64_t>(cat_out_size.size()) - cat_dim) - 1
      : 0; // if tensor is empty then dim of the concatenated tensor will be 0
  synConcatenateParams concat_params{};
  concat_params.axis = static_cast<unsigned int>(cat_dim);
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
  auto reshape_ind_op = OpBackend::BuildExpandDims(
      op, graph, catop2_res.get(), reshape_size, indices_scalar_type, 1);
  auto mulOp = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("mult_fwd", indices_scalar_type),
       {catop.get(), reshape_ind_op.get()},
       {{catop.pt_shape(), indices_scalar_type}}});

  auto red_output_shape = mulOp.at(0).pt_shape();
  int red_dim = 1;
  // red_output_shape.erase(red_output_shape.cbegin()+red_dim);
  red_output_shape[static_cast<size_t>(red_dim)] = 1;
  ns_Reduction::Params red_params{};
  red_params.reductionDimension =
      static_cast<unsigned int>(get_dim_in_tpc_order(
          red_dim /*dim*/, static_cast<int64_t>(red_output_shape.size())));

  // reduce_sum_fwd_i64 causes perf drop compared to reduce_sum_fwd_i32
  auto reduce_sum_type = ((indices_scalar_type == c10::ScalarType::Long) &&
                          (common::IsInt64Supported()))
      ? c10::ScalarType::Int
      : indices_scalar_type;
  auto sumop = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("reduce_sum_fwd", reduce_sum_type),
       {mulOp.at(0).get()},
       {{red_output_shape, reduce_sum_type}},
       &red_params,
       sizeof(red_params)});

  // reduce_sum kernel definition assumes AnyType on output, so cast in some
  // cases is required
  std::vector<synapse_helpers::tensor> cast_node;
  if (indices_scalar_type != reduce_sum_type) {
    cast_node.push_back(std::move(OpBackend::BuildCast(
        op,
        graph,
        sumop.at(0).get(),
        red_output_shape,
        reduce_sum_type,
        indices_scalar_type)));
  } else {
    cast_node.push_back(std::move(sumop.at(0)));
  }
  std::vector<int64_t> reshape_sum({cast_node.at(0).pt_shape()[0]});
  auto reshape_sum_op = OpBackend::BuildSqueeze(
      op, graph, cast_node.at(0).get(), reshape_sum, indices_scalar_type);

  auto sortOp = TopK_Helper(
      op,
      graph,
      {reshape_sum_op.get()},
      0,
      reshape_sum_op.pt_shape(),
      0 /*descending order*/,
      static_cast<int>(reshape_sum_op.pt_shape().size()),
      static_cast<int>(reshape_sum_op.pt_shape()[0]),
      0, /*median vairiant*/
      indices_scalar_type);
  auto sort_res0 = std::move(sortOp.at(0));
  auto sort_res1 = std::move(sortOp.at(1));

  // Fill params for gather
  ns_GatherKernel::Params gather_params;
  auto outshape = catop.pt_shape();
  auto size1 = outshape.size();
  auto size2 = sort_res1.pt_shape().size();
  long gather_dim = 0;
  gather_params.axis =
      static_cast<int>(size1 - static_cast<size_t>(gather_dim) - 1);
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
      {get_guid_with_precision("gather_fwd", indices_scalar_type),
       {catop.get(), sort_res1.get()},
       {{outshape, indices_scalar_type}},
       &gather_params,
       sizeof(gather_params)});
  std::vector<int64_t> reshape_size2({sort_res1.pt_shape()[0], 1});
  auto reshape_sort1_op = OpBackend::BuildExpandDims(
      op, graph, sort_res1.get(), reshape_size2, indices_scalar_type, 0);
  ns_ScatterNDKernel::Params scatter_params{int(catop.pt_shape().size()), {0}};
  // Dims reversed between PT and synapse
  for (int64_t i = static_cast<int64_t>(scatter_indices_shape.size()) - 1,
               j = 0;
       i >= 0;
       --i, ++j) {
    scatter_params.origIndicesShape[j] =
        static_cast<int>(scatter_indices_shape[static_cast<size_t>(i)]);
  }
  // scatter_nd_fwd has no support for int16 and u8 , hence we need to cast
  std::string cast_guid{};
  auto scatter_nd_fwd_dtype = self_scalar_type;
  at::ScalarType cast_dtype = self_scalar_type;
  bool cast_needed = false;
  if ((cast_needed = CheckAndGetCastGuid(
           "scatter_nd_fwd", self_scalar_type, cast_guid, cast_dtype))) {
    scatter_nd_fwd_dtype = cast_dtype;
  }
  std::vector<synapse_helpers::tensor> next_node;
  if (cast_needed) {
    auto scatter_op = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("scatter_nd_fwd", scatter_nd_fwd_dtype),
        {gatherOp[0].get(), reshape_sort1_op.get(), reshape_val_op.get()},
        {NodeAttr::NodeOutputAttr{self.sizes().vec(), scatter_nd_fwd_dtype}},
        &scatter_params,
        sizeof(scatter_params)});
    size_t size = 0;
    PARAMS_STUB(ns_CastKernel::Params);
    size = sizeof(params);
    params->round_mode = CAST_ROUND_ZERO;
    next_node = OpBackend::BuildNode(
        op,
        graph,
        {cast_guid,
         {scatter_op[0].get()},
         {NodeAttr::NodeOutputAttr{self.sizes().vec(), self_scalar_type}},
         params.get(),
         size});

  } else {
      next_node = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("scatter_nd_fwd", scatter_nd_fwd_dtype),
        {gatherOp[0].get(), reshape_sort1_op.get(), reshape_val_op.get()},
        {NodeAttr::NodeOutputAttr{self.sizes().vec(), scatter_nd_fwd_dtype}},
        &scatter_params,
        sizeof(scatter_params)});

  }
  auto addOp = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("add_fwd", self_scalar_type), // original dtype
       {syn_in_0, next_node.at(0).get()}, // cast node
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
  auto indices_scalar_type = indices[0].scalar_type();
  std::vector<at::Tensor> cat_input;

  std::vector<synTensor> cat_input_synTensor;
  std::vector<synapse_helpers::tensor> cat_input_tensor;
  std::vector<std::vector<int64_t>> cat_input_index;

  const auto expanded_size_dim0 =
      indices[0].dim() ? indices[0].sizes().vec()[0] : 1;

  for (size_t i = 0; i < indices.size(); i++) {
    std::vector<int64_t> expanded_size = {expanded_size_dim0, 1};
    cat_input_tensor.emplace_back(ExpandDimsHelper(
        graph, syn_in(i + 1), expanded_size, indices_scalar_type, 0));
    cat_input_synTensor.emplace_back(
        cat_input_tensor[cat_input_tensor.size() - 1].get());
    cat_input_index.emplace_back(
        cat_input_tensor[cat_input_tensor.size() - 1].pt_shape());
  }

  int64_t cat_dim = 1;
  std::vector<int64_t> cat_out_size =
      CalcCatOutSize(&cat_input_index, &cat_dim);
  cat_dim = cat_out_size.size() > 0
      ? (static_cast<int64_t>(cat_out_size.size()) - cat_dim) - 1
      : 0; // if tensor is empty then dim of the concatenated tensor will be 0
  synConcatenateParams concat_params{};
  concat_params.axis = static_cast<unsigned int>(cat_dim);
  auto catop1 = BuildOp(
      graph,
      "concat",
      std::move(cat_input_synTensor),
      {{cat_out_size, indices_scalar_type}},
      &concat_params,
      sizeof(concat_params));

  auto catop = std::move(catop1.at(0));
  // Calculate the dimensionality of updates for broadcasting
  auto rank_inp = static_cast<size_t>(self.ndimension());
  auto rank_idx = static_cast<size_t>(catop.pt_shape()[1]);
  std::vector<int64_t> value_upd_dim{catop.pt_shape()[0]};

  std::vector<synapse_helpers::tensor> values_bcast_or_reshape_sh_tensor;
  auto values_scalar_type = values.scalar_type();

  if (((int)indices.size() == self.dim()) && (values.numel() > 1)) {
    value_upd_dim.clear();
    value_upd_dim = values.sizes().vec();
  }

  for (size_t i = rank_idx; i < rank_inp; ++i)
    value_upd_dim.push_back(self.sizes().vec()[i]);

  // value_upd_dim is the final shape we want for values tensor to match
  // scatter_nd_onnx requirements. Either broadcast of reshape input values
  // tensor to get that shape.
  if (values.dim() <= (int)value_upd_dim.size()) {
    auto broadcastToElements = catop.pt_shape()[0];
    auto broadcastFromElements = std::accumulate(
        std::begin(value_upd_dim),
        std::end(value_upd_dim),
        1,
        std::multiplies<int>());
    if (broadcastFromElements < broadcastToElements)
      value_upd_dim.insert(
          std::begin(value_upd_dim),
          broadcastToElements / broadcastFromElements);

  }

  values_bcast_or_reshape_sh_tensor.emplace_back(std::move(BuildOp(
      graph,
      get_guid_with_precision("index_put_broadcast_value", ScalarType()),
      {syn_in(0), catop.get(), syn_in(1 + indices.size())},
      {{value_upd_dim, ScalarType()}})[0]));
  auto self_scalar_type = self.scalar_type();
  // scatter_nd_fwd has no support for int16 and u8 , hence we need to cast
  std::string cast_guid{};
  auto scatter_nd_onnx_fwd_dtype = self_scalar_type;
  at::ScalarType cast_dtype = self_scalar_type;
  bool cast_needed = false;
  if ((cast_needed = CheckAndGetCastGuid(
           "scatter_nd_onnx_fwd", self_scalar_type, cast_guid, cast_dtype))) {
    scatter_nd_onnx_fwd_dtype = cast_dtype;
  }

  if ((int)indices.size() == self.dim()) {
    std::vector<int64_t> reshape_bcast_size({catop.pt_shape()[0]});
    auto reshape_val_op = FlattenHelper(
        graph,
        values_bcast_or_reshape_sh_tensor[0].get(),
        reshape_bcast_size,
        values_scalar_type);

    std::vector<synapse_helpers::tensor> next_node;

    if (!accumulate) {
      if (cast_needed) {
        auto scatter_op = BuildOp(
            graph,
            get_guid_with_precision(
                "scatter_nd_onnx_fwd",
                scatter_nd_onnx_fwd_dtype), // dytpe will be the casted one
            {syn_in(0), catop.get(), reshape_val_op.get()},
            {NodeAttr::NodeOutputAttr{
                self.sizes().vec(), scatter_nd_onnx_fwd_dtype}});
        next_node = BuildOp(
            graph,
            cast_guid,
            {scatter_op[0].get()},
            {{self.sizes().vec(), self_scalar_type, 0}},
             0);

        } else {
        next_node = BuildOp(
            graph,
            get_guid_with_precision(
                "scatter_nd_onnx_fwd",
                scatter_nd_onnx_fwd_dtype), // dytpe will be the casted one
            {syn_in(0), catop.get(), reshape_val_op.get()},
            {NodeAttr::NodeOutputAttr{
                self.sizes().vec(), self_scalar_type, 0}});

      }
      syn_out(0) = std::move(next_node[0]);

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

      std::vector<synapse_helpers::tensor> next_node;
      if (cast_needed) {
        auto scatter_op = BuildOp(
            graph,
            get_guid_with_precision(
                "scatter_nd_onnx_fwd", scatter_nd_onnx_fwd_dtype),
            {syn_in(0),
             catop.get(),
             values_bcast_or_reshape_sh_tensor[0].get()},
            {NodeAttr::NodeOutputAttr{
                self.sizes().vec(), scatter_nd_onnx_fwd_dtype}});

        next_node = BuildOp(
            graph,
            cast_guid,
            {scatter_op[0].get()},
            {NodeAttr::NodeOutputAttr{
                self.sizes().vec(), self_scalar_type, 0}});

      } else {
        next_node = BuildOp(
            graph,
            get_guid_with_precision(
                "scatter_nd_onnx_fwd", scatter_nd_onnx_fwd_dtype),
            {syn_in(0),
             catop.get(),
             values_bcast_or_reshape_sh_tensor[0].get()},
            {NodeAttr::NodeOutputAttr{
                self.sizes().vec(), scatter_nd_onnx_fwd_dtype, 0}});
      }
      syn_out(0) = std::move(next_node[0]);

    } else {
      syn_out(0) = HandleIndexPutWithAcc(
          this,
          graph,
          self,
          catop,
          values_bcast_or_reshape_sh_tensor[0],
          syn_in(0),
          rank_idx,
          indices_scalar_type);
    }
  }
}

IndexPutBoolEager::IndexPutBoolEager(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, {}, scalar_type, {0}, {}, {}, false) {}

void IndexPutBoolEager::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto indices = stack.at(1).toTensorList().vec();
  auto values = stack_tensor(stack, 2);
  auto accumulate = stack.at(3).toBool();
  auto max_size = broadcast_size(indices, self);
  int64_t max_indices_count = 1;
  for (size_t i = 0; i < indices.size(); i++) {
    max_indices_count = (indices[i].numel() > max_indices_count)
        ? indices[i].numel()
        : max_indices_count;
  }
  auto indices_scalar_type =
      (common::IsInt64Supported() &&
       ((max_indices_count > INT_MAX) || graph.is_dynamic_graph()))
      ? c10::ScalarType::Long
      : c10::ScalarType::Int;
  std::vector<synapse_helpers::tensor> nonzero;
  auto shape_tensor_shape = at::DimVector{5};

  auto self_sizes = self.sizes().vec();
  std::vector<at::Tensor> cat_input;
  std::vector<synTensor> cat_input_synTensor;
  std::vector<synapse_helpers::tensor> cat_input_tensor;
  std::vector<std::vector<int64_t>> cat_input_index;
  std::vector<int64_t> nonzero_out_shape;
  auto unsqueeze = [this, &graph](
                       const at::Tensor& t, synTensor st, const int ndims) {
    const auto missing = static_cast<size_t>(ndims - t.dim());
    auto new_shape = t.sizes().vec();
    new_shape.insert(new_shape.end(), missing, 1);
    ns_ExpandMultiDimsKernel::Params params;
    params.expand_axes_mask = 0;
    for (size_t i = 0; i < missing; i++)
      params.expand_axes_mask |= (1 << i);

    return this->BuildOp(
        graph,
        "expand_multi_dims_fwd",
        {st},
        {{new_shape, t.scalar_type()}},
        &params,
        sizeof(params));
  };
  int64_t slice_numel;

  for (size_t i = 0; i < indices.size(); i++) {
    NonZeroParams_t index_params;
    index_params.dtype = indices[i].scalar_type();
    std::vector<synTensor> inputPutBoolBroadcastIndexInputs{syn_in(0)};
    std::vector<synapse_helpers::tensor> storage;
    if (static_cast<int64_t>(max_size.size()) > indices[i].dim()) {
      storage = unsqueeze(
          indices[i], syn_in(i + 1), static_cast<int>(max_size.size()));
      inputPutBoolBroadcastIndexInputs.emplace_back(storage[0].get());
    } else {
      inputPutBoolBroadcastIndexInputs.emplace_back(syn_in(i + 1));
    }

    auto bcastOpInd = BuildOp(
        graph,
        "index_put_bool_broadcast_index",
        std::move(inputPutBoolBroadcastIndexInputs),
        {{max_size, index_params.dtype}});

    index_params.sizes = max_size;
    index_params.numel = std::accumulate(
        std::begin(max_size), std::end(max_size), 1, std::multiplies<size_t>());
    index_params.force_long = false;
    nonzero = NonZeroCommon(
        this,
        graph,
        index_params,
        bcastOpInd[0].get(),
        c10::nullopt,
        c10::nullopt,
        false);

    nonzero_out_shape = nonzero[0].pt_shape();
    synSliceParamsV2 slice_params{};
    slice_numel = index_params.numel;
    slice_params.axes[0] = 1; // always slice on the row of 2D indices (index
                              // vector for each dim is one col)
    slice_params.starts[0] = 0;
    slice_params.ends[0] = slice_numel;
    slice_params.steps[0] = 1;
    int64_t slice_output_shape_second_dim_size =
        (1 == nonzero_out_shape.size()) ? 1 : nonzero_out_shape[1];
    auto slice = BuildOp(
        graph,
        get_guid_with_precision("slice", indices_scalar_type),
        {nonzero[0].get()},
        {{{slice_numel, slice_output_shape_second_dim_size},
          indices_scalar_type}},
        &slice_params,
        sizeof(slice_params));
    cat_input_tensor.emplace_back(std::move(slice.at(0)));
    cat_input_synTensor.emplace_back(
        cat_input_tensor[cat_input_tensor.size() - 1].get());
    cat_input_index.emplace_back(
        cat_input_tensor[cat_input_tensor.size() - 1].pt_shape());
  }
  int64_t cat_dim = 1;
  std::vector<int64_t> cat_out_size =
      CalcCatOutSize(&cat_input_index, &cat_dim);
  cat_dim = cat_out_size.size() > 0
      ? (static_cast<int64_t>(cat_out_size.size()) - cat_dim) - 1
      : 0; // If tensor is empty then dim of the concatenated tensor will be 0
  synConcatenateParams concat_params{};
  concat_params.axis = static_cast<unsigned int>(cat_dim);
  auto catop1 = BuildOp(
      graph,
      "concat",
      std::move(cat_input_synTensor),
      {{cat_out_size, indices_scalar_type}},
      &concat_params,
      sizeof(concat_params));

  auto catop = std::move(catop1.at(0));
  auto cat_pt_shape = catop.pt_shape();
  // Calculate the dimensionality of updates for broadcasting
  auto rank_inp = static_cast<size_t>(self.ndimension());
  size_t rank_idx = 0;
  for (size_t i = 0; i < indices.size(); i++)
    rank_idx += indices[i].dim();
  auto values_scalar_type = values.scalar_type();
  std::vector<int64_t> value_upd_dim;
  if (values.numel() >
      1) { // if values has more than 1 elem, we have to assume the valid
    // count in indices will match values numel
    auto values_sizes = values.sizes().vec();
    if (indices[0].dim() != self.dim() &&
        values.dim() != (1 + (self.dim() - indices[0].dim()))) {
      size_t i = 0;
      while (i < rank_idx) {
        for (int j = 0; j < indices[i].dim(); j++)
          value_upd_dim.push_back(indices[i].sizes()[j]);
        i += indices[i].dim();
      }
      for (size_t i = rank_idx; i < rank_inp; i++)
        value_upd_dim.push_back(self_sizes[i]);
    } else {
      for (size_t i = 0; i < static_cast<size_t>(values.dim()); i++)
        value_upd_dim.push_back(values_sizes[i]);
    }
  } else { // We are assuming uses passes value shapes correctly for scatter
    size_t i = 0;
    while (i < rank_idx) {
      for (int j = 0; j < indices[i].dim(); j++)
        value_upd_dim.push_back(indices[i].sizes()[j]);
      i += indices[i].dim();
    }
    for (size_t i = rank_idx; i < rank_inp; i++)
      value_upd_dim.push_back(self_sizes[i]);
  }

  auto bcastOp = BuildOp(
      graph,
      get_guid_with_precision(
          "index_put_bool_broadcast_value", values_scalar_type),
      {syn_in(0), syn_in(1), syn_in(1 + indices.size()), nonzero[0].get()},
      {{value_upd_dim, values_scalar_type}});

  auto flattened_size = std::accumulate(
      std::begin(value_upd_dim),
      std::end(value_upd_dim),
      1,
      std::multiplies<size_t>());
  auto reshapebcastOp = BuildOp(
      graph,
      "flatten_fwd",
      {bcastOp[0].get()},
      {{{flattened_size}, values_scalar_type}});

  auto self_scalar_type = self.scalar_type();
  if (!accumulate) {
    auto scatter_op = BuildOp(
        graph,
        get_guid_with_precision("scatter_nd_onnx_fwd", self_scalar_type),
        {syn_in(0), catop.get(), reshapebcastOp[0].get(), nonzero.at(1).get()},
        {NodeAttr::NodeOutputAttr{self_sizes, self_scalar_type, 0}});
    syn_out(0) = std::move(scatter_op[0]);
  } else {
    auto zero_op = ConstantHelper(graph, 0, self_scalar_type, self_sizes);
    auto scatter_op = BuildOp(
        graph,
        get_guid_with_precision("scatter_nd_onnx_fwd", self_scalar_type),
        {zero_op.get(),
         catop.get(),
         reshapebcastOp[0].get(),
         nonzero.at(1).get()},
        {NodeAttr::NodeOutputAttr{self_sizes, self_scalar_type}});
    auto add_op = BuildOp(
        graph,
        get_guid_with_precision("add_fwd", self_scalar_type),
        {syn_in(0), scatter_op.at(0).get()},
        {NodeAttr::NodeOutputAttr{self_sizes, self_scalar_type, 0}});
    syn_out(0) = std::move(add_op[0]);
  }
}

static synapse_helpers::tensor IndexPutLongHelper(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Stack& stack,
    std::vector<at::Tensor> indices,
    synTensor self_synin,
    std::vector<synTensor> indices_synin,
    synTensor value_synin) {
  auto self = stack_tensor(stack, 0);
  auto values = stack_tensor(stack, 2);
  auto accumulate = stack.at(3).toBool();
  auto max_size = broadcast_size(indices, self);
  auto indices_scalar_type = indices[0].scalar_type();
  std::vector<at::Tensor> cat_input;
  std::vector<synTensor> cat_input_synTensor;
  std::vector<synapse_helpers::tensor> cat_input_tensor;
  std::vector<std::vector<int64_t>> cat_input_index;
  for (size_t i = 0; i < indices.size(); i++) {
    auto bcastOp = OpBackend::BuildBroadcast(
        op, graph, indices_synin[i], max_size, indices_scalar_type);
    // Reshape broadcasted indices to [N, 1] for concatenation
    auto flattened_size = std::accumulate(
        std::begin(max_size), std::end(max_size), 1, std::multiplies<size_t>());

    std::vector<int64_t> expanded_size = {flattened_size, 1};
    cat_input_tensor.emplace_back(OpBackend::BuildReshape(
        op, graph, bcastOp.get(), expanded_size, indices_scalar_type));
    cat_input_synTensor.emplace_back(
        cat_input_tensor[cat_input_tensor.size() - 1].get());
    cat_input_index.emplace_back(
        cat_input_tensor[cat_input_tensor.size() - 1].pt_shape());
  }

  int64_t cat_dim = 1;
  std::vector<int64_t> cat_out_size =
      CalcCatOutSize(&cat_input_index, &cat_dim);
  cat_dim = cat_out_size.size() > 0
      ? (static_cast<int64_t>(cat_out_size.size()) - cat_dim) - 1
      : 0; // if tensor is empty then dim of the concatenated tensor will be 0
  synConcatenateParams concat_params{};
  concat_params.axis = static_cast<unsigned int>(cat_dim);
  auto catop1 = OpBackend::BuildNode(
      op,
      graph,
      {"concat",
       std::move(cat_input_synTensor),
       {{cat_out_size, indices_scalar_type}},
       &concat_params,
       sizeof(concat_params)});

  auto catop = std::move(catop1.at(0));
  // Calculate the dimensionality of updates for broadcasting
  auto rank_inp = static_cast<size_t>(self.ndimension());
  auto rank_idx = static_cast<size_t>(catop.pt_shape()[1]);
  std::vector<int64_t> value_upd_dim{catop.pt_shape()[0]};
  if (((int)indices.size() == self.dim()) && (values.numel() > 1)) {
    value_upd_dim.clear();
    value_upd_dim = values.sizes().vec();
  }

  for (size_t i = rank_idx; i < rank_inp; ++i)
    value_upd_dim.push_back(self.sizes().vec()[i]);
  auto values_scalar_type = values.scalar_type();
  std::vector<synapse_helpers::tensor> values_bcast_or_reshape_sh_tensor;
  // value_upd_dim is the final shape we want for values tensor to match
  // scatter_nd_onnx requirements. Either broadcast of reshape input values
  // tensor to get that shape.
  if (values.dim() <= (int)value_upd_dim.size()) {
    values_bcast_or_reshape_sh_tensor.emplace_back(OpBackend::BuildBroadcast(
        op, graph, value_synin, value_upd_dim, values_scalar_type));
  } else {
    values_bcast_or_reshape_sh_tensor.emplace_back(OpBackend::BuildReshape(
        op, graph, value_synin, value_upd_dim, values_scalar_type));
  }
  auto self_scalar_type = self.scalar_type();
  // scatter_nd_fwd has no support for int16 and u8 , hence we need to cast
  std::string cast_guid{};
  auto scatter_nd_onnx_fwd_dtype = self_scalar_type;
  at::ScalarType cast_dtype = self_scalar_type;
  bool cast_needed = false;
  if ((cast_needed = CheckAndGetCastGuid(
           "scatter_nd_onnx_fwd", self_scalar_type, cast_guid, cast_dtype))) {
    scatter_nd_onnx_fwd_dtype = cast_dtype;
  }

  std::vector<synapse_helpers::tensor> next_node;
  if ((int)indices.size() == self.dim()) {
    std::vector<int64_t> reshape_bcast_size({catop.pt_shape()[0]});
    auto reshape_val_op = OpBackend::BuildReshape(
        op,
        graph,
        values_bcast_or_reshape_sh_tensor[0].get(),
        reshape_bcast_size,
        values_scalar_type);

    if (!accumulate) {
      if (cast_needed) {
        auto scatter_op = OpBackend::BuildNode(
            op,
            graph,
            {get_guid_with_precision(
                 "scatter_nd_onnx_fwd",
                 scatter_nd_onnx_fwd_dtype), // dytpe will be the casted one
             {self_synin, catop.get(), reshape_val_op.get()},
             {NodeAttr::NodeOutputAttr{
                 self.sizes().vec(), scatter_nd_onnx_fwd_dtype}}});
        next_node = OpBackend::BuildNode(
            op,
            graph,
            {cast_guid,
             {scatter_op[0].get()},
             {{self.sizes().vec(), self_scalar_type, 0}},
             0});

      } else {
        next_node = OpBackend::BuildNode(
            op,
            graph,
            {get_guid_with_precision(
                 "scatter_nd_onnx_fwd",
                 scatter_nd_onnx_fwd_dtype), // dytpe will be the casted one
             {self_synin, catop.get(), reshape_val_op.get()},
             {NodeAttr::NodeOutputAttr{
                 self.sizes().vec(), self_scalar_type, 0}}});
      }
      return std::move(next_node[0]);
    } else {
      return HandleIndexPutWithAcc(
          op,
          graph,
          self,
          catop,
          reshape_val_op,
          self_synin,
          rank_idx,
          indices_scalar_type);
    }
  } else {
    if (!accumulate) {
      if (cast_needed) {
        auto scatter_op = OpBackend::BuildNode(
            op,
            graph,
            {get_guid_with_precision(
                 "scatter_nd_onnx_fwd", scatter_nd_onnx_fwd_dtype),
             {self_synin,
              catop.get(),
              values_bcast_or_reshape_sh_tensor[0].get()},
             {NodeAttr::NodeOutputAttr{
                 self.sizes().vec(), scatter_nd_onnx_fwd_dtype}}});

        next_node = OpBackend::BuildNode(
            op,
            graph,
            {cast_guid,
             {scatter_op[0].get()},
             {NodeAttr::NodeOutputAttr{
                 self.sizes().vec(), self_scalar_type, 0}}});

      } else {
        next_node = OpBackend::BuildNode(
            op,
            graph,
            {get_guid_with_precision(
                 "scatter_nd_onnx_fwd", scatter_nd_onnx_fwd_dtype),
             {self_synin,
              catop.get(),
              values_bcast_or_reshape_sh_tensor[0].get()},
             {NodeAttr::NodeOutputAttr{
                 self.sizes().vec(), scatter_nd_onnx_fwd_dtype, 0}}});
      }
      return std::move(next_node[0]);
    } else {
      return HandleIndexPutWithAcc(
          op,
          graph,
          self,
          catop,
          values_bcast_or_reshape_sh_tensor[0],
          self_synin,
          rank_idx,
          indices_scalar_type);
    }
  };
}

IndexPutCompile::IndexPutCompile(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, {}, scalar_type, {0}, {}, {}, false) {}

void IndexPutCompile::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  std::vector<at::Tensor> indices;
  std::vector<synTensor> indices_synin;
  bool indices_are_bool = false;
  int i = 0;
  int rank_idx_long = 0;
  if (stack.at(1).isOptionalTensorList()) {
    PT_KERNEL_DEBUG(
        "index_put boolmask torch.compile: received list of optional tensors");
    auto opt_tensorlist_args = stack.at(1).toOptionalTensorList();
    for (c10::optional<at::Tensor> input_ind : opt_tensorlist_args) {
      auto input = input_ind.value_or(at::Tensor());
      if (input.defined()) {
        PT_KERNEL_DEBUG(
            "torch.compile : index_put boolmask: indices tensor: ",
            input.scalar_type(),
            " size = ",
            input.sizes());
        if (input.scalar_type() != c10::ScalarType::Bool) {
          rank_idx_long++;
        } else {
          indices_are_bool = true;
        }
        indices.push_back(input);
        indices_synin.push_back(syn_in(i + 1));
        i++;
      } else {
        PT_KERNEL_DEBUG(
            "torch.compile: index_put boolmask: undefined indices tensor");
        HABANA_ASSERT(
            0 &&
            "torch.compile: index_put boolmask: unsupported case: None is not yet supported on HPU for c10::List<c10::optional<Tensor>>");
      }
    }
    HABANA_ASSERT(
        0,
        "torch.compile: index_put boolmask: OptionalTensorList is not handled in kernel");
  } else {
    indices = stack.at(1).toTensorList().vec();
    for (; i < (int)indices.size(); i++) {
      indices_synin.push_back(syn_in(i + 1));
      if (indices[i].scalar_type() != c10::ScalarType::Bool) {
        rank_idx_long++;
      } else {
        indices_are_bool = true;
      }
    }
  }

  if (!indices_are_bool) {
    syn_out(0) = IndexPutLongHelper(
        this,
        graph,
        stack,
        indices,
        syn_in(0),
        indices_synin,
        syn_in(1 + indices.size()));
    return;
  }
  // auto indices = stack.at(1).toTensorList().vec();
  auto values = stack_tensor(stack, 2);
  auto accumulate = stack.at(3).toBool();
  auto max_size = broadcast_size(indices, self);
  int64_t max_indices_count = 1;
  bool all_bool_indices = true;
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i].scalar_type() == c10::ScalarType::Bool) {
      max_indices_count = (indices[i].numel() > max_indices_count)
          ? indices[i].numel()
          : max_indices_count;
    } else {
      all_bool_indices = false;
    }
  }
  auto indices_scalar_type = (common::IsInt64Supported() &&
                              ((max_indices_count > INT_MAX) ||
                               graph.is_dynamic_graph() || !all_bool_indices))
      ? c10::ScalarType::Long
      : c10::ScalarType::Int;
  std::vector<synapse_helpers::tensor> nonzero;
  auto shape_tensor_shape = at::DimVector{5};
  auto self_sizes = self.sizes().vec();
  std::vector<at::Tensor> cat_input;
  std::vector<synTensor> cat_input_synTensor;
  std::vector<synapse_helpers::tensor> cat_input_tensor;
  std::vector<std::vector<int64_t>> cat_input_index;
  std::vector<int64_t> nonzero_out_shape;
  auto unsqueeze = [this, &graph](
                       const at::Tensor& t, synTensor st, const int ndims) {
    const auto missing = static_cast<size_t>(ndims - t.dim());
    auto new_shape = t.sizes().vec();
    new_shape.insert(new_shape.end(), missing, 1);
    return this->ReshapeHelper(graph, st, new_shape, t.scalar_type());
  };
  int64_t slice_numel;

  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i].scalar_type() == c10::ScalarType::Bool) {
      NonZeroParams_t index_params;
      index_params.dtype = indices[i].scalar_type();
      auto bcastOpInd = BroadcastHelper(
          graph,
          static_cast<int64_t>(max_size.size()) > indices[i].dim()
              ? unsqueeze(
                    indices[i],
                    syn_in(i + 1),
                    static_cast<int>(max_size.size()))
                    .get()
              : syn_in(i + 1),
          max_size,
          index_params.dtype);
      index_params.sizes = max_size;
      index_params.numel = std::accumulate(
          std::begin(max_size),
          std::end(max_size),
          1,
          std::multiplies<size_t>());
      index_params.force_long = !all_bool_indices;
      nonzero = NonZeroCommon(
          this,
          graph,
          index_params,
          bcastOpInd.get(),
          c10::nullopt,
          c10::nullopt,
          false);

      nonzero_out_shape = nonzero[0].pt_shape();
      synSliceParamsV2 slice_params{};
      slice_numel = index_params.numel;
      slice_params.axes[0] = 1; // always slice on the row of 2D indices
                                // (index vector for each dim is one col)
      slice_params.starts[0] = 0;
      slice_params.ends[0] = slice_numel;
      slice_params.steps[0] = 1;
      int64_t slice_output_shape_second_dim_size =
          (1 == nonzero_out_shape.size()) ? 1 : nonzero_out_shape[1];
      auto slice = BuildOp(
          graph,
          get_guid_with_precision("slice", indices_scalar_type),
          {nonzero[0].get()},
          {{{slice_numel, slice_output_shape_second_dim_size},
            indices_scalar_type}},
          &slice_params,
          sizeof(slice_params));
      cat_input_tensor.emplace_back(std::move(slice.at(0)));
      cat_input_synTensor.emplace_back(
          cat_input_tensor[cat_input_tensor.size() - 1].get());
      cat_input_index.emplace_back(
          cat_input_tensor[cat_input_tensor.size() - 1].pt_shape());
    } else {
      auto bcastOp =
          BroadcastHelper(graph, syn_in(i + 1), max_size, indices_scalar_type);
      // Reshape broadcasted indices to [N, 1] for concatenation
      auto flattened_size = std::accumulate(
          std::begin(max_size),
          std::end(max_size),
          1,
          std::multiplies<size_t>());

      std::vector<int64_t> expanded_size = {flattened_size, 1};
      cat_input_tensor.emplace_back(ReshapeHelper(
          graph, bcastOp.get(), expanded_size, indices_scalar_type));
      cat_input_synTensor.emplace_back(
          cat_input_tensor[cat_input_tensor.size() - 1].get());
      cat_input_index.emplace_back(
          cat_input_tensor[cat_input_tensor.size() - 1].pt_shape());
    }
  }
  int64_t cat_dim = 1;
  std::vector<int64_t> cat_out_size =
      CalcCatOutSize(&cat_input_index, &cat_dim);
  cat_dim = cat_out_size.size() > 0
      ? (static_cast<int64_t>(cat_out_size.size()) - cat_dim) - 1
      : 0; // If tensor is empty then dim of the concatenated tensor will be 0
  synConcatenateParams concat_params{};
  concat_params.axis = static_cast<unsigned int>(cat_dim);
  auto catop1 = BuildOp(
      graph,
      "concat",
      std::move(cat_input_synTensor),
      {{cat_out_size, indices_scalar_type}},
      &concat_params,
      sizeof(concat_params));
  auto catop = std::move(catop1.at(0));
  auto cat_pt_shape = catop.pt_shape();
  // Calculate the dimensionality of updates for broadcasting
  auto rank_inp = static_cast<size_t>(self.ndimension());
  size_t rank_idx = 0;
  for (size_t i = 0; i < indices.size(); i++)
    rank_idx += indices[i].dim();
  auto values_scalar_type = values.scalar_type();
  std::vector<int64_t> value_upd_dim;
  if (values.numel() >
      1) { // if values has more than 1 elem, we have to assume the valid
    // count in indices will match values numel
    auto values_sizes = values.sizes().vec();
    if (indices[0].dim() != self.dim() &&
        values.dim() != (1 + (self.dim() - indices[0].dim()))) {
      size_t i = 0;
      while (i < rank_idx) {
        for (int j = 0; j < indices[i].dim(); j++)
          value_upd_dim.push_back(indices[i].sizes()[j]);
        i += indices[i].dim();
      }
      for (size_t i = rank_idx; i < rank_inp; i++)
        value_upd_dim.push_back(self_sizes[i]);
    } else {
      for (size_t i = 0; i < static_cast<size_t>(values.dim()); i++)
        value_upd_dim.push_back(values_sizes[i]);
    }
  } else { // We are assuming uses passes value shapes correctly for scatter
    size_t i = 0;
    while (i < rank_idx) {
      for (int j = 0; j < indices[i].dim(); j++)
        value_upd_dim.push_back(indices[i].sizes()[j]);
      i += indices[i].dim();
    }
    for (size_t i = rank_idx; i < rank_inp; i++)
      value_upd_dim.push_back(self_sizes[i]);
  }

  auto bcastOp = BroadcastHelper(
      graph, syn_in(1 + indices.size()), value_upd_dim, values_scalar_type);
  auto flattened_size = std::accumulate(
      std::begin(value_upd_dim),
      std::end(value_upd_dim),
      1,
      std::multiplies<size_t>());
  std::vector<int64_t> reshape_bcast_size({catop.pt_shape()[0]});
  auto reshapebcastOp =
      ReshapeHelper(graph, bcastOp.get(), flattened_size, values_scalar_type);

  auto self_scalar_type = self.scalar_type();
  if (!accumulate) {
    auto scatter_op = BuildOp(
        graph,
        get_guid_with_precision("scatter_nd_onnx_fwd", self_scalar_type),
        {syn_in(0), catop.get(), reshapebcastOp.get(), nonzero.at(1).get()},
        {NodeAttr::NodeOutputAttr{self_sizes, self_scalar_type, 0}});
    syn_out(0) = std::move(scatter_op[0]);
  } else {
    auto zero_op = ConstantHelper(graph, 0, self_scalar_type, self_sizes);
    auto scatter_op = BuildOp(
        graph,
        get_guid_with_precision("scatter_nd_onnx_fwd", self_scalar_type),
        {zero_op.get(), catop.get(), reshapebcastOp.get(), nonzero.at(1).get()},
        {NodeAttr::NodeOutputAttr{self_sizes, self_scalar_type}});
    auto add_op = BuildOp(
        graph,
        get_guid_with_precision("add_fwd", self_scalar_type),
        {syn_in(0), scatter_op.at(0).get()},
        {NodeAttr::NodeOutputAttr{self_sizes, self_scalar_type, 0}});
    syn_out(0) = std::move(add_op[0]);
  }
}

} // namespace habana

static const auto& IndexPutKernelRegistry = habana::KernelRegistry().add(
    "hpu::_index_put_impl_eager",
    KERNEL_FN_GLOBAL(habana::IndexPutEager));

static const auto& IndexPutboolKernelRegistry = habana::KernelRegistry().add(
    "hpu::_index_put_impl_bool_eager",
    KERNEL_FN_GLOBAL(habana::IndexPutBoolEager));

static const auto& IndexPutAtenKernelRegistry = habana::KernelRegistry().add(
    "aten::index_put.hacked_twin",
    KERNEL_FN_GLOBAL(habana::IndexPutCompile));
