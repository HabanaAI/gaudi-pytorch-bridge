/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <bitset>
#include "generated/hpu_op.h"
#include "habana_kernels/reduction_kernels.h"
#include "hpu_op_helper.h"
#define GUID "reduce_log_sum_exp_fwd_"

namespace habana {

sizes_vec LogSumExpOutputShape(const at::Stack& stack, bool) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  std::vector<int64_t> dim = stack.at(1).toIntList().vec();
  const bool keepdim = stack.at(2).toBool();
  std::vector<int64_t> compute_shape =
      ReduceOperator::compute_output_shape(self, dim, keepdim);
  return {compute_shape};
}

std::shared_ptr<void> LogSumExpParams(
    const at::Stack& stack,
    size_t& size,
    int64_t index) {
  PARAMS_STUB(ns_Reduction::Params);
  auto ndim = static_cast<int>(stack.at(0).toTensor().dim());
  auto reduction_dim = ndim - 1 - index;
  params->reductionDimension = reduction_dim;
  return params;
}

struct Parameters {
  std::shared_ptr<void> param_list;
  size_t size_list;
  std::vector<int64_t> shape_list;
};

void LogSumExp::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  auto self = stack.at(0).toTensor();

  const bool keepdim = stack.at(2).toBool();
  auto ndim = self.dim();
  auto mask = std::bitset<64>();
  auto self_shape = self.sizes().vec();
  auto dim = stack.at(1).toIntList();
  // When dim=[], reduce all dimensions based on keepdim value
  if (0 == dim.size()) {
    for (int i = 0; i < ndim; ++i) {
      dim.push_back(i);
    }
  }

  for (const auto& i : dim) {
    mask.set(c10::maybe_wrap_dim(i, ndim, true));
  }

  auto new_shape = LogSumExpOutputShape(stack)[0];

  std::vector<int64_t> orig_shape{self.sizes().vec()};
  std::vector<synTensor> logsumexp_list;
  std::vector<synapse_helpers::tensor> logsumexp_itr;
  std::vector<Parameters> parameters;
  Parameters p;

  for (int64_t dimIndex = orig_shape.size() - 1; dimIndex >= 0; dimIndex--) {
    if (mask[dimIndex]) {
      orig_shape[dimIndex] = 1;
      size_t size = 0;
      auto params = LogSumExpParams(stack, size, dimIndex);

      p.param_list = params;
      p.size_list = size;
      p.shape_list = orig_shape;
      parameters.push_back({p});
    }
  }
  size_t len = parameters.size();

  if (keepdim) {
    // When keepdim value is set to true
    if (len == 1) {
      auto logsumexp = BuildOp(
          graph,
          GUID + habana_helpers::name_suffix_from_type(ScalarType()),
          {syn_in(0)},
          {{parameters[0].shape_list,
            ScalarType(),
            is_output_persistent_list[0]},
           {self_shape[0], ScalarType(), is_output_persistent_list[1]}},
          parameters[0].param_list.get(),
          parameters[0].size_list);

      // output of logsumexp is the output of this op
      syn_out(0) = std::move(logsumexp[0]);

    } else if (len > 1) {
      auto logsumexp = BuildOp(
          graph,
          GUID + habana_helpers::name_suffix_from_type(ScalarType()),
          {syn_in(0)},
          {{parameters[0].shape_list, ScalarType()},
           {self_shape, ScalarType()}},
          parameters[0].param_list.get(),
          parameters[0].size_list);

      logsumexp_list.emplace_back(logsumexp[0].get());

      // Iterating over the for loop when multiple dim values are passed
      for (size_t i = 1; i <= len - 1; i++) {
        logsumexp_itr = BuildOp(
            graph,
            GUID + habana_helpers::name_suffix_from_type(ScalarType()),
            {logsumexp_list[i - 1]},
            {{parameters[i].shape_list, ScalarType()},
             {self_shape, ScalarType()}},
            parameters[i].param_list.get(),
            parameters[i].size_list);

        // Reshape occurs when multiple dim values are passed
        if (i == len - 1) {
          auto reshape = BuildOp(
              graph,
              "reshape",
              {logsumexp_itr[0].get()},
              {{new_shape, ScalarType(), is_output_persistent_list[0], true}});

          // output of reshape is the output of this op
          syn_out(0) = std::move(reshape[0]);
        }
        logsumexp_list.emplace_back(logsumexp_itr[0].get());
      }
    }

  } else {
    // When keepdim value is set to false
    auto logsumexp = BuildOp(
        graph,
        GUID + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0)},
        {{parameters[0].shape_list, ScalarType()}, {self_shape, ScalarType()}},
        parameters[0].param_list.get(),
        parameters[0].size_list);

    logsumexp_list.emplace_back(logsumexp[0].get());

    if (len > 1) {
      // Iterating over the for loop when multiple dim values are passed
      for (size_t i = 1; i <= len - 1; i++) {
        logsumexp_itr = BuildOp(
            graph,
            GUID + habana_helpers::name_suffix_from_type(ScalarType()),
            {logsumexp_list[i - 1]},
            {{parameters[i].shape_list, ScalarType()},
             {self_shape, ScalarType()}},
            parameters[i].param_list.get(),
            parameters[i].size_list);

        // Reshape occurs when multiple dim values are passed
        if (i == len - 1) {
          auto reshape = BuildOp(
              graph,
              "reshape",
              {logsumexp_itr[0].get()},
              {{new_shape, ScalarType(), is_output_persistent_list[0], true}});

          // output of reshape is the output of this op
          syn_out(0) = std::move(reshape[0]);
        }
        logsumexp_list.emplace_back(logsumexp_itr[0].get());
      }
    }
    // Reshape occurs when single dim value is passed
    if (len == 1) {
      auto reshape = BuildOp(
          graph,
          "reshape",
          {logsumexp[0].get()},
          {{new_shape, ScalarType(), is_output_persistent_list[0], true}});

      // output of reshape is the output of this op
      syn_out(0) = std::move(reshape[0]);
    }
  }
}
} // namespace habana
