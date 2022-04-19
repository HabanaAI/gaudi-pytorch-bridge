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

c10::optional<synapse_helpers::tensor> HandleReductionDtype(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    synTensor syn_in,
    c10::optional<at::ScalarType> dtype) {
  auto dtype_val = dtype.value_or(self.scalar_type());
  if (dtype_val == self.scalar_type()) {
    return c10::nullopt;
  }

  return OpBackend::BuildCast(
      op, graph, syn_in, self.sizes(), self.scalar_type(), dtype_val);
}

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
    const at::Tensor& self,
    std::vector<synTensor> inputs,
    const at::IntArrayRef dims,
    bool keepdim,
    const std::string& guid,
    std::vector<NodeAttr::NodeOutputAttr> output_attr) {
  struct Parameters {
    std::shared_ptr<void> param_list;
    size_t size_list;
    std::vector<int64_t> shape_list;
  };

  auto dim = dims.vec();
  auto mask = std::bitset<64>();
  std::vector<int64_t> orig_shape{self.sizes().vec()};
  const int ndims = orig_shape.size();
  auto num_outputs = output_attr.size();

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
  HABANA_ASSERT(num_outputs <= 2, "Number of outputs is greater than 2.");
  auto reduc_output_attrs =
      [](sizes_vec outshapes,
         std::vector<at::ScalarType> dtypes,
         int num_out) -> std::vector<NodeAttr::NodeOutputAttr> {
    std::vector<NodeAttr::NodeOutputAttr> reduc_output_attrs;
    for (int itr = 0; itr < num_out; itr++) {
      reduc_output_attrs.push_back({outshapes.at(itr), dtypes.at(itr)});
    }

    return reduc_output_attrs;
  };

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
  auto retain_ten_shape =
      num_outputs > 1 ? parameters[0].shape_list : self.sizes().vec();
  size_t len = parameters.size();
  // NOTE: 1. Need to handle when TPC returns two outputs and pytorch
  // returns one output.
  // 2. Flatten the input when dim is none for certain ops.
  if (keepdim) {
    // When keepdim value is set to true
    if (len == 1) {
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
           reduc_output_attrs(
               {parameters[0].shape_list, retain_ten_shape},
               {output_attr[0].dtype, output_attr[1].dtype},
               num_outputs),
           parameters[0].param_list.get(),
           parameters[0].size_list});

      tensor_list.emplace_back(op_out[0].get());

      // Iterating over the for loop when multiple dim values are passed
      for (size_t i = 1; i <= len - 1; i++) {
        tensor_itr = OpBackend::BuildNode(
            op,
            graph,
            {guid,
             {tensor_list[i - 1]},
             reduc_output_attrs(
                 {parameters[i].shape_list, retain_ten_shape},
                 {output_attr[0].dtype, output_attr[1].dtype},
                 num_outputs),
             parameters[i].param_list.get(),
             parameters[i].size_list});

        tensor_list.emplace_back(tensor_itr[0].get());
      }
      for (unsigned int itr = 0; itr < num_outputs; itr++) {
        auto reshape = OpBackend::BuildReshape(
            op,
            graph,
            tensor_itr[itr].get(),
            output_attr[itr].sizes,
            output_attr[itr].dtype,
            output_attr[itr].final_result_index);

        // output of reshape is the output of this op
        reshape_list.emplace_back(std::move(reshape));
      }
      return reshape_list;
    }
  } else {
    // When keepdim value is set to false
    auto op_out = OpBackend::BuildNode(
        op,
        graph,
        {guid,
         std::move(inputs),
         reduc_output_attrs(
             {parameters[0].shape_list, retain_ten_shape},
             {output_attr[0].dtype, output_attr[1].dtype},
             num_outputs),
         parameters[0].param_list.get(),
         parameters[0].size_list});

    tensor_list.emplace_back(op_out[0].get());
    if (len == 1) {
      for (unsigned int itr = 0; itr < num_outputs; itr++) {
        auto reshape = OpBackend::BuildReshape(
            op,
            graph,
            op_out[itr].get(),
            output_attr[itr].sizes,
            output_attr[itr].dtype,
            output_attr[itr].final_result_index);

        // output of reshape is the output of this op
        reshape_list.emplace_back(std::move(reshape));
      }

      return reshape_list;
    } else {
      // Iterating over the for loop when multiple dim values are passed
      for (size_t i = 1; i <= len - 1; i++) {
        tensor_itr = OpBackend::BuildNode(
            op,
            graph,
            {guid,
             {tensor_list[i - 1]},
             reduc_output_attrs(
                 {parameters[i].shape_list, retain_ten_shape},
                 {output_attr[0].dtype, output_attr[1].dtype},
                 num_outputs),
             parameters[i].param_list.get(),
             parameters[i].size_list});

        tensor_list.emplace_back(tensor_itr[0].get());
      }
      // output of reshape is the output of this op
      for (unsigned int itr = 0; itr < num_outputs; itr++) {
        auto reshape = OpBackend::BuildReshape(
            op,
            graph,
            tensor_itr[itr].get(),
            output_attr[itr].sizes,
            output_attr[itr].dtype,
            output_attr[itr].final_result_index);

        // output of reshape is the output of this op
        reshape_list.emplace_back(std::move(reshape));
      }
      return reshape_list;
    }
  }
}
} // namespace habana
