/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/hpu_op.h"
#include "habana_kernels/reduction_kernels.h"
#include "hpu_op_helper.h"
#include "reduction_op_util.h"

#define guidReducesum "reduce_sum_fwd_"

namespace habana {
sizes_vec NanSumOutputShape(const at::Stack& stack, bool) {
  static_cast<void>(stack);
  return {{}};
}

sizes_vec NanSumIntListOutputShape(const at::Stack& stack, bool) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  std::vector<int64_t> dim = stack.at(1).toIntList().vec();
  const bool keepdim = stack.at(2).toBool();
  std::vector<int64_t> shape =
      ReduceOperator::compute_output_shape(self, dim, keepdim);
  return {shape};
}

std::shared_ptr<void> NanSumParams(
    const at::Stack& stack,
    size_t& size,
    int64_t index) {
  PARAMS_STUB(ns_Reduction::Params);
  auto ndim = static_cast<int>(stack.at(0).toTensor().dim());
  auto reduction_dim = ndim - 1 - index;
  params->reductionDimension = reduction_dim;
  return params;
}

void NansumList::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();
  auto dtype = c10::ScalarType::Char;

  auto self = stack.at(0).toTensor();
  auto dim = stack.at(1).toIntVector();

  bool keepdim = stack.at(2).toBool();

  auto new_shape = NanSumIntListOutputShape(stack)[0];

  auto guid =
      guidReducesum + habana_helpers::name_suffix_from_type(ScalarType());

  // isNan on input
  auto is_nan = BuildOp(
      graph,
      "isnan_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0)},
      {{outshape, dtype}});

  auto zero_constant = ConstantHelper(graph, 0.0f, ScalarType(), outshape);

  // where on is_nan
  auto where = BuildOp(
      graph,
      "where_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {is_nan[0].get(), zero_constant.get(), syn_in(0)},
      {{outshape, ScalarType()}});

  auto reduce_sum = HandleReductionDimAndKeepdim(
      this,
      graph,
      self,
      {where[0].get()},
      dim,
      keepdim,
      guid,
      {{new_shape, ScalarType(), 0}});
  syn_out(0) = std::move(reduce_sum[0]);
}

void Nansum::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();
  auto dtype = c10::ScalarType::Char;

  struct Parameters {
    std::shared_ptr<void> param_list;
    size_t size_list;
    std::vector<int64_t> shape_list;
  };

  auto self = stack.at(0).toTensor();

  std::vector<int64_t> new_shape{1};
  std::vector<int64_t> orig_shape{self.sizes().vec()};

  std::vector<synTensor> reduce_sum_list;
  std::vector<synapse_helpers::tensor> reduce_sum_itr;
  std::vector<Parameters> parameters;
  Parameters p;
  auto guid =
      guidReducesum + habana_helpers::name_suffix_from_type(ScalarType());

  for (int64_t dimIndex = orig_shape.size() - 1; dimIndex >= 0; dimIndex--) {
    orig_shape[dimIndex] = 1;

    size_t size = 0;
    auto params = NanSumParams(stack, size, dimIndex);
    parameters.push_back({params, size, orig_shape});
  }

  // isNan on input
  auto is_nan = BuildOp(
      graph,
      "isnan_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0)},
      {{outshape, dtype}});

  auto zero_constant = ConstantHelper(graph, 0.0f, ScalarType(), outshape);

  // where on is_nan
  auto where = BuildOp(
      graph,
      "where_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {is_nan[0].get(), zero_constant.get(), syn_in(0)},
      {{outshape, ScalarType()}});

  int64_t len = parameters.size();

  auto reduce_sum = BuildOp(
      graph,
      guid,
      {where[0].get()},
      {{parameters[0].shape_list, ScalarType()}},
      parameters[0].param_list.get(),
      parameters[0].size_list);
  reduce_sum_list.emplace_back(reduce_sum[0].get());

  if (len > 1) {
    // Iterating over the for loop when multiple dim values are passed
    for (int i = 1; i <= len - 1; i++) {
      reduce_sum_itr = BuildOp(
          graph,
          guid,
          {reduce_sum_list[i - 1]},
          {{parameters[i].shape_list, ScalarType()}},
          parameters[i].param_list.get(),
          parameters[i].size_list);

      // Reshape the last node to final output shape
      if (i == len - 1) {
        auto reshape = ReshapeHelper(
            graph, reduce_sum_itr[0].get(), new_shape, ScalarType(), 0);

        syn_out(0) = std::move(reshape);
      }
      reduce_sum_list.push_back(reduce_sum_itr[0].get());
    }
  }

  if (len == 1) {
    auto reshape =
        ReshapeHelper(graph, reduce_sum[0].get(), new_shape, ScalarType(), 0);

    syn_out(0) = std::move(reshape);
  }
}
} // namespace habana
