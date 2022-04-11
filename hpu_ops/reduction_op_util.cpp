/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "reduction_op_util.h"
#include "generated/hpu_op.h"
#include "hpu_op_helper.h"

namespace habana {

std::shared_ptr<void> ReductionOpParams(
    const int ndim,
    size_t& size,
    int64_t index) {
  PARAMS_STUB(ns_Reduction::Params);
  auto reduction_dim = ndim - 1 - index;
  params->reductionDimension = reduction_dim;
  return params;
}

std::vector<synapse_helpers::tensor> HandleReductionDimAndKeepdim(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> inputs,
    const at::IntArrayRef dims,
    bool keepdim,
    const std::string& guid,
    const at::IntArrayRef self_shape,
    const at::IntArrayRef outshape,
    std::vector<NodeAttr::NodeOutputAttr> output_attr) {
  struct Parameters {
    std::shared_ptr<void> param_list;
    size_t size_list;
    std::vector<int64_t> shape_list;
  };

  auto dim = dims.vec();
  auto mask = std::bitset<64>();
  std::vector<int64_t> orig_shape{self_shape.vec()};
  const int ndims = orig_shape.size();

  std::vector<synTensor> tensor_list;
  std::vector<synapse_helpers::tensor> reshape_list;
  std::vector<synapse_helpers::tensor> tensor_itr;
  std::vector<Parameters> parameters;
  Parameters p;
  // When dim=[], reduce all dimensions based on keepdim value
  if (0 == dim.size()) {
    for (int i = 0; i < ndims; ++i) {
      dim.push_back(i);
    }
  }

  for (const auto& i : dim) {
    mask.set(c10::maybe_wrap_dim(i, ndims, true));
  }

  for (int64_t dimIndex = orig_shape.size() - 1; dimIndex >= 0; dimIndex--) {
    if (mask[dimIndex]) {
      orig_shape[dimIndex] = 1;
      size_t size = 0;
      auto params = ReductionOpParams(ndims, size, dimIndex);

      p.param_list = params;
      p.size_list = size;
      p.shape_list = orig_shape;
      parameters.push_back({p});
    }
  }

  size_t len = parameters.size();
  output_attr[0].sizes = parameters[0].shape_list;
  if (keepdim) {
    // When keepdim value is set to true
    if (len == 1) {
      output_attr[0].final_result_index = 0; // result index
      auto op_out = OpBackend::BuildNode(
          op,
          graph,
          {guid,
           std::move(inputs),
           output_attr,
           parameters[0].param_list.get(),
           parameters[0].size_list});

      return op_out;
    } else {
      auto op_out = OpBackend::BuildNode(
          op,
          graph,
          {guid,
           std::move(inputs),
           output_attr,
           parameters[0].param_list.get(),
           parameters[0].size_list});

      tensor_list.emplace_back(op_out[0].get());

      // Iterating over the for loop when multiple dim values are passed
      for (size_t i = 1; i <= len - 1; i++) {
        output_attr[0].sizes = parameters[i].shape_list;
        tensor_itr = OpBackend::BuildNode(
            op,
            graph,
            {guid,
             {tensor_list[i - 1]},
             output_attr,
             parameters[i].param_list.get(),
             parameters[i].size_list});

        tensor_list.emplace_back(tensor_itr[0].get());
      }

      auto reshape = OpBackend::BuildReshape(
          op, graph, tensor_itr[0].get(), outshape, op->ScalarType(), 0);

      // output of reshape is the output of this op
      reshape_list.emplace_back(std::move(reshape));
      return reshape_list;
    }
  } else {
    // When keepdim value is set to false
    auto op_out = OpBackend::BuildNode(
        op,
        graph,
        {guid,
         std::move(inputs),
         output_attr,
         parameters[0].param_list.get(),
         parameters[0].size_list});

    tensor_list.emplace_back(op_out[0].get());
    if (len == 1) {
      auto reshape = OpBackend::BuildReshape(
          op, graph, op_out[0].get(), outshape, op->ScalarType(), 0);
      reshape_list.emplace_back(std::move(reshape));

      return reshape_list;
    } else {
      // Iterating over the for loop when multiple dim values are passed
      for (size_t i = 1; i <= len - 1; i++) {
        output_attr[0].sizes = parameters[i].shape_list;
        tensor_itr = OpBackend::BuildNode(
            op,
            graph,
            {guid,
             {tensor_list[i - 1]},
             output_attr,
             parameters[i].param_list.get(),
             parameters[i].size_list});

        tensor_list.emplace_back(tensor_itr[0].get());
      }
      // output of reshape is the output of this op
      auto reshape = OpBackend::BuildReshape(
          op, graph, tensor_itr[0].get(), outshape, op->ScalarType(), 0);
      reshape_list.emplace_back(std::move(reshape));
      return reshape_list;
    }
  }
}
} // namespace habana
