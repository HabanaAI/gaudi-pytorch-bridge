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

#include "generated/backend/nan_to_num.h"

namespace habana {
void NantoNum::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const at::Tensor self = stack_tensor(stack, 0);
  const auto outshape = stack_tensor(stack, 0).sizes();

  // If input is of integral dtype then copy it into output
  if (c10::isIntegralType(ScalarType(), true)) {
    auto copy =
        BuildOp(graph, "memcpy", {syn_in(0)}, {{outshape, ScalarType(), 0}});
    syn_out(0) = std::move(copy[0]);
  } else {
    auto nan_constant = stack.at(1).isNone() ? 0.0 : stack.at(1).toDouble();

    auto const_nan =
        ConstantHelper(graph, nan_constant, ScalarType(), outshape);

    c10::Scalar max_value;

    if (self.dtype() == c10::ScalarType::BFloat16) {
      max_value = std::numeric_limits<c10::BFloat16>::max();
    } else {
      max_value = std::numeric_limits<float>::max();
    }

    auto posinf_constant =
        stack.at(2).isNone() ? max_value : stack.at(2).toDouble();

    auto const_posinf =
        ConstantHelper(graph, posinf_constant, ScalarType(), outshape);

    c10::Scalar min_value;

    if (self.dtype() == c10::ScalarType::BFloat16) {
      min_value = std::numeric_limits<c10::BFloat16>::lowest();
    } else {
      min_value = std::numeric_limits<float>::lowest();
    }

    auto neginf_constant =
        stack.at(3).isNone() ? min_value : stack.at(3).toDouble();

    auto const_neginf =
        ConstantHelper(graph, neginf_constant, ScalarType(), outshape);

    const at::ScalarType& result_type = c10::ScalarType::Bool;

    auto nan_mask = BuildOp(
        graph,
        get_guid_with_precision("isnan_fwd", ScalarType()),
        {syn_in(0)},
        {{outshape, result_type}});

    ns_IsInfKernel::Params params_pos{};
    params_pos.detect_positive = 1;

    auto posinf_mask = BuildOp(
        graph,
        get_guid_with_precision("isinf_fwd", ScalarType()),
        {syn_in(0)},
        {{outshape, result_type}},
        &params_pos,
        sizeof(params_pos));

    ns_IsInfKernel::Params params_neg{};
    params_neg.detect_negative = 1;

    auto neginf_mask = BuildOp(
        graph,
        get_guid_with_precision("isinf_fwd", ScalarType()),
        {syn_in(0)},
        {{outshape, result_type}},
        &params_neg,
        sizeof(params_neg));

    auto where_nan = BuildOp(
        graph,
        get_guid_with_precision("where_fwd", ScalarType()),
        {nan_mask[0].get(), const_nan.get(), syn_in(0)},
        {{outshape, ScalarType()}});

    auto where_pos = BuildOp(
        graph,
        get_guid_with_precision("where_fwd", ScalarType()),
        {posinf_mask[0].get(), const_posinf.get(), where_nan[0].get()},
        {{outshape, ScalarType()}});

    auto where_neg = BuildOp(
        graph,
        get_guid_with_precision("where_fwd", ScalarType()),
        {neginf_mask[0].get(), const_neginf.get(), where_pos[0].get()},
        {{outshape, ScalarType(), 0}});
    syn_out(0) = std::move(where_neg[0]);
  }
}
} // namespace habana
