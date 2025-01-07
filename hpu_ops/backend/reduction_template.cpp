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
#include "hpu_ops/backend/reduction_template.h"
#include "backend/helpers/lowering_util.h"
#include "habana_kernels/kernel_utils.h"
#include "hpu_ops/common/reduction_template.h"

namespace habana {

ns_Reduction::ParamsV2 FillReductionParams(
    int64_t ndims,
    c10::IntArrayRef dims,
    bool keepdim) {
  ns_Reduction::ParamsV2 params;
  params.keepDim = keepdim;

  params.reductionDimensionMask = 0;
  for (auto&& dim : dims) {
    auto d = c10::maybe_wrap_dim(dim, ndims);
    if ((d >= 0) && (d < ndims)) {
      params.reductionDimensionMask |= (1 << (ndims - d - 1));
    }
  }

  return params;
}

inline bool reduction_support_f32(const std::string& guid) {
  return guid.find("reduce_prod_multi_dim_fwd") != std::string::npos or
      guid.find("reduce_mean_multi_dim_fwd") != std::string::npos;
}

inline bool reduction_support_i32(const std::string& guid) {
  return guid.find("reduce_sum_multi_dim") != std::string::npos;
}

// Returns the input after cast to the supplied dtype. If dtype is none or if
// dtype is same as input's dtype, returns nullopt.
c10::optional<synapse_helpers::tensor> HandleReductionDtype(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    synTensor syn_in,
    at::optional<at::ScalarType> dtype) {
  auto dtype_val = dtype.value_or(self.scalar_type());
  std::string guid = op->GetGuid();
  if (at::isIntegralType(self.scalar_type(), true) and
      reduction_support_i32(guid)) {
    if (dtype_val != at::kFloat) {
      dtype_val = at::kInt;
    }
  } else if (
      reduction_support_f32(guid) and
      at::isIntegralType(self.scalar_type(), true)) {
    dtype_val = at::kFloat;
  } else {
    return c10::nullopt;
    // do nothing
  }
  op->SetGuid(update_guid_dtype(guid, dtype_val));
  if (habana_helpers::getInternalDtype(dtype_val) ==
      habana_helpers::getInternalDtype(self.scalar_type())) {
    return c10::nullopt;
  }

  op->SetScalarType(dtype_val);

  return OpBackend::BuildCast(
      op, graph, syn_in, self.sizes(), self.scalar_type(), dtype_val);
}

std::vector<int64_t> CalculateReductionMultiDimAndKeepdimOutputSize(
    const std::vector<int64_t>& inputSize,
    const std::vector<int64_t>& dimsToReduce,
    bool keepDim) {
  if (keepDim) {
    std::vector<int64_t> outputSize = inputSize;
    for (const int64_t dim : dimsToReduce) {
      outputSize[dim] = 1;
    }
    return outputSize;
  } else {
    const size_t numOfDimsLeft = inputSize.size() - dimsToReduce.size();
    if (numOfDimsLeft == 0) {
      return {1};
    }
    std::vector<int64_t> outputSize;
    outputSize.reserve(numOfDimsLeft);

    for (size_t i = 0; i < inputSize.size(); ++i) {
      if (std::find(dimsToReduce.begin(), dimsToReduce.end(), i) ==
          dimsToReduce.end()) {
        outputSize.push_back(inputSize[i]);
      }
    }
    return outputSize;
  }
}

std::vector<synapse_helpers::tensor> HandleReduction(
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor syn_in,
    const std::string& guid,
    c10::IntArrayRef dimsToReduce,
    const int64_t inputRank,
    const bool keepdim,
    std::vector<NodeAttr::NodeOutputAttr> output_attr) {
  auto params = FillReductionParams(inputRank, dimsToReduce, keepdim);

  return OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision(guid, op->ScalarType()),
       {syn_in},
       std::move(output_attr),
       &params,
       sizeof(params)});
}
} // namespace habana
