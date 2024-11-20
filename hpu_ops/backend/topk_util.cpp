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

#include "hpu_ops/topk_util.h"
#include "generated/backend/topk.h"
namespace habana {

std::vector<synapse_helpers::tensor> TopK_Helper(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input,
    int reduction_axis,
    const at::IntArrayRef topk_outshape,
    int descending_order,
    int ndimension,
    int kvalue,
    int variant,
    c10::optional<at::ScalarType> out_dtype) {
  synBeamParams Topk_params{};
  Topk_params.bsw = kvalue;
  Topk_params.axis = reduction_axis;
  Topk_params.bottomK = descending_order;
  auto indices_dtype =
      common::IsInt64Supported() ? c10::ScalarType::Long : c10::ScalarType::Int;
  if (variant == 1)
    Topk_params.axis = get_dim_in_tpc_order(reduction_axis, ndimension);
  at::ScalarType topk_dtype =
      (out_dtype == c10::nullopt) ? op->ScalarType() : out_dtype.value();
  return OpBackend::BuildNode(
      op,
      graph,
      {"topk",
       std::move(input),
       {{topk_outshape, topk_dtype}, {topk_outshape, indices_dtype}},
       &Topk_params,
       sizeof(Topk_params)});
}

} // namespace habana
