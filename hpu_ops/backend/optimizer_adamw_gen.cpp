/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
#include "hpu_ops/backend/reduction_template.h"
#include "hpu_ops/fp8_ops.h"
#include "hpu_ops/op_backend.h"

namespace sh = synapse_helpers;

namespace habana {

static std::tuple<sh::tensor, sh::tensor> GetMomentInFp8WithScale(
    OpBackend* op,
    sh::graph& graph,
    const at::Tensor& pt_input,
    const sh::tensor& input,
    synTensor old_scale,
    std::vector<sh::tensor>& constants,
    const at::ScalarType& original_dtype,
    const at::ScalarType& destination_dtype,
    const int out_ids,
    const int out_scale) {
  auto abs_input = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("abs", original_dtype),
       {input.get()},
       {{pt_input.sizes().vec(), original_dtype}}});

  ns_Reduction::ParamsV2 reduce_params;
  reduce_params.reductionDimensionMask = 0;
  reduce_params.keepDim = false;

  auto amax = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("reduce_max_multi_dim_fwd", original_dtype),
       {abs_input[0].get()},
       {{{1}, original_dtype}},
       &reduce_params,
       sizeof(reduce_params)});

  auto amax_div = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("div_fwd", original_dtype),
       {constants[destination_dtype == c10::ScalarType::Float8_e4m3fn ? 2 : 3]
            .get(),
        amax[0].get()},
       {{{1}, original_dtype}}});

  auto amax_log = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("log2_fwd", original_dtype),
       {amax_div[0].get()},
       {{{1}, original_dtype}}});
  auto exp = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("floor_fwd", original_dtype),
       {amax_log[0].get()},
       {{{1}, original_dtype}}});

  auto new_scale = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("pow_fwd", original_dtype),
       {constants[0].get(), exp[0].get()},
       {{{1}, original_dtype}}});

  auto mask = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("greater_fwd", original_dtype),
       {new_scale[0].get(), constants[1].get()},
       {{{1}, torch::kBool}}});

  auto updated_scale = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("where_fwd", original_dtype),
       {mask[0].get(), new_scale[0].get(), old_scale},
       {{{1}, original_dtype, out_scale}}});

  auto cast_params = GetCastParams(true, original_dtype, destination_dtype);
  auto result = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("convert_to_fp8", original_dtype),
       {input.get(), updated_scale[0].get()},
       {{pt_input.sizes().vec(), destination_dtype, out_ids}},
       &cast_params,
       sizeof(cast_params)});

  return std::make_tuple(std::move(result[0]), std::move(updated_scale[0]));
}

class OptimizerFusedAdamWOperator : public OpBackend {
 public:
  OptimizerFusedAdamWOperator(int device_id, c10::ScalarType scalar_type)
      : OpBackend(
            device_id,
            NO_TPC + "optimizer_fused_AdamwOperator_",
            scalar_type,
            {},
            {1, 2, 3, 10, 11}, // inplace ids
            {},
            false) {}

  void AddNode(sh::graph& graph, const at::Stack& stack) override;
  void CustomHandler([[maybe_unused]] sh::graph&, at::Stack&) override;
};

void OptimizerFusedAdamWOperator::CustomHandler(sh::graph&, at::Stack& stack) {
  const bool is_fp8 =
      at::isFloat8Type(stack.at(2).toTensorList().get(0).scalar_type());
  if (is_fp8) {
    auto tensor_list = stack.at(10).toOptional<c10::List<at::Tensor>>();
    if (tensor_list.has_value()) {
      stack.at(10) = tensor_list.value();
    }
    auto tensor_list_2 = stack.at(11).toOptional<c10::List<at::Tensor>>();
    if (tensor_list_2.has_value()) {
      stack.at(11) = tensor_list_2.value();
    }
  }
}

void OptimizerFusedAdamWOperator::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "OptimizerFusedAdamWOperator::AddNode");
  auto gradient_vec = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto weight_vec = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto exp_avg_vec = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto exp_avg_sq_vec = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto neg_step_t = getNextInput<TensorsPair>(stackGetter);
  auto beta1 = getNextInput<double>(stackGetter);
  auto beta2 = getNextInput<double>(stackGetter);
  auto epsilon = getNextInput<double>(stackGetter);
  auto weight_decay = getNextInput<TensorsPair>(stackGetter);
  auto has_weight_decay = getNextInput<bool>(stackGetter);
  auto exp_avg_scales =
      getNextInput<c10::optional<std::vector<TensorsPair>>>(stackGetter);
  auto exp_avg_sq_scales =
      getNextInput<c10::optional<std::vector<TensorsPair>>>(stackGetter);

  if ((gradient_vec.size() != weight_vec.size()) ||
      (gradient_vec.size() != exp_avg_vec.size()) ||
      (gradient_vec.size() != exp_avg_sq_vec.size())) {
    std::stringstream ss;
    ss << "All 4 vector inputs must have the same number of elements but they respectively have: "
       << gradient_vec.size() << ", " << weight_vec.size() << ", "
       << exp_avg_vec.size() << ", " << exp_avg_sq_vec.size();
    AT_ERROR(ss.str());
  }

  const auto& first_moment_dtype = exp_avg_vec.front().pt_t.scalar_type();
  const bool is_fp8 = at::isFloat8Type(first_moment_dtype);
  const auto scalar_dtype = is_fp8
      ? ScalarType()
      : at::promote_types(ScalarType(), first_moment_dtype);

  std::string add_node = get_guid_with_precision("add_fwd", scalar_dtype);
  std::string mul_node = get_guid_with_precision("mult_fwd", scalar_dtype);
  std::string div_node = get_guid_with_precision("div_fwd", scalar_dtype);
  std::string sqrt_node = get_guid_with_precision("sqrt_fwd", scalar_dtype);
  std::string from_fp8_node =
      get_guid_with_precision("convert_from_fp8", scalar_dtype);

  int64_t scalar_shape[] = {1};

  double constant_values[] = {beta1, beta2, 1.0 - beta1, 1.0 - beta2, epsilon};
  std::array<synTensor, std::size(constant_values)> constant_ts{};
  const auto& beta1_t = constant_ts[0];
  const auto& beta2_t = constant_ts[1];
  const auto& one_minus_beta1_t = constant_ts[2];
  const auto& one_minus_beta2_t = constant_ts[3];
  const auto& epsilon_t = constant_ts[4];

  std::vector<sh::tensor> storage;
  storage.reserve(constant_ts.size() + 1);
  for (size_t i = 0; i < constant_ts.size(); ++i) {
    storage.push_back(ConstantHelper(
        graph,
        static_cast<float>(constant_values[i]),
        scalar_dtype,
        scalar_shape));
    constant_ts[i] = storage.back().get();
  }

  std::vector<sh::tensor> fp8_constants;
  if (is_fp8) {
    fp8_constants.push_back(
        ConstantHelper(graph, 2.0, scalar_dtype, scalar_shape));
    fp8_constants.push_back(
        ConstantHelper(graph, 0.0, scalar_dtype, scalar_shape));
    fp8_constants.push_back(ConstantHelper(
        graph, 240.0, scalar_dtype, scalar_shape)); // Float8_e4m3fn max
    fp8_constants.push_back(ConstantHelper(
        graph, 57344.0, scalar_dtype, scalar_shape)); // Float8_e5m2 max
  }

  size_t vec_size = gradient_vec.size();
  for (size_t i = 0; i < vec_size; ++i) {
    const auto& gradient = gradient_vec[i];
    const auto& weight = weight_vec[i];
    const auto& exp_avg = exp_avg_vec[i];
    const auto& exp_avg_sq = exp_avg_sq_vec[i];

    c10::optional<synTensor> exp_avg_scale_syn, exp_avg_sq_scale_syn;
    std::optional<sh::tensor> exp_avg_casted, exp_avg_sq_casted;
    std::optional<sh::tensor> exp_avg_scale_updated, exp_avg_sq_scale_updated;
    if (is_fp8) {
      exp_avg_scale_syn = exp_avg_scales.value()[i].syn_t;
      exp_avg_sq_scale_syn = exp_avg_sq_scales.value()[i].syn_t;
      auto cast_result = BuildOp(
          graph,
          from_fp8_node,
          {exp_avg.syn_t, exp_avg_scale_syn.value()},
          {{exp_avg.pt_t.sizes(), scalar_dtype}});
      exp_avg_casted = std::move(cast_result[0]);

      auto cast_sq_result = BuildOp(
          graph,
          from_fp8_node,
          {exp_avg_sq.syn_t, exp_avg_sq_scale_syn.value()},
          {{exp_avg_sq.pt_t.sizes(), scalar_dtype}});
      exp_avg_sq_casted = std::move(cast_sq_result[0]);
    }

    std::vector<NodeAttr::NodeOutputAttr> gradient_attr = {
        {gradient.pt_t.sizes(), scalar_dtype}};
    std::vector<NodeAttr::NodeOutputAttr> weight_attr = {
        {weight.pt_t.sizes(), scalar_dtype}};
    std::vector<NodeAttr::NodeOutputAttr> exp_avg_attr = {
        {exp_avg.pt_t.sizes(), scalar_dtype}};
    std::vector<NodeAttr::NodeOutputAttr> exp_avg_sq_attr = {
        {exp_avg_sq.pt_t.sizes(), scalar_dtype}};

    auto exp_avg_mul_beta1 = BuildOp(
        graph,
        mul_node,
        {is_fp8 ? exp_avg_casted.value().get() : exp_avg.syn_t, beta1_t},
        exp_avg_attr);

    auto grad_scaled = BuildOp(
        graph, mul_node, {gradient.syn_t, one_minus_beta1_t}, gradient_attr);

    auto exp_avg_1 = BuildOp(
        graph,
        add_node,
        {exp_avg_mul_beta1[0].get(), grad_scaled[0].get()},
        {{gradient.pt_t.sizes(), scalar_dtype}});

    std::optional<sh::tensor> exp_avg_1_out;
    if (is_fp8) {
      auto moment_and_scale = GetMomentInFp8WithScale(
          this,
          graph,
          exp_avg.pt_t,
          exp_avg_1[0],
          exp_avg_scale_syn.value(),
          fp8_constants,
          scalar_dtype,
          exp_avg.pt_t.scalar_type(),
          i + 1 * vec_size,
          i + 3 * vec_size);
      exp_avg_1_out = std::move(std::get<0>(moment_and_scale));
      exp_avg_scale_updated = std::move(std::get<1>(moment_and_scale));
    } else {
      exp_avg_1_out = exp_avg.pt_t.scalar_type() == scalar_dtype
          ? IdentityHelper(
                graph,
                exp_avg_1[0].get(),
                exp_avg.pt_t.sizes(),
                scalar_dtype,
                i + vec_size)
          : BuildCast(
                this,
                graph,
                exp_avg_1[0].get(),
                exp_avg.pt_t.sizes(),
                scalar_dtype,
                exp_avg.pt_t.scalar_type(),
                i + vec_size);
    }

    auto grad_sq = BuildOp(
        graph, mul_node, {gradient.syn_t, gradient.syn_t}, gradient_attr);

    auto grad_sq_scaled = BuildOp(
        graph, mul_node, {grad_sq[0].get(), one_minus_beta2_t}, gradient_attr);

    auto exp_avg_sq_mul_beta2 = BuildOp(
        graph,
        mul_node,
        {is_fp8 ? exp_avg_sq_casted.value().get() : exp_avg_sq.syn_t, beta2_t},
        exp_avg_sq_attr);

    auto exp_avg_sq_1 = BuildOp(
        graph,
        add_node,
        {exp_avg_sq_mul_beta2[0].get(), grad_sq_scaled[0].get()},
        {NodeAttr::NodeOutputAttr{gradient.pt_t.sizes(), scalar_dtype}});

    std::optional<sh::tensor> exp_avg_sq_1_out;
    if (is_fp8) {
      auto moment_and_scale = GetMomentInFp8WithScale(
          this,
          graph,
          exp_avg_sq.pt_t,
          exp_avg_sq_1[0],
          exp_avg_sq_scale_syn.value(),
          fp8_constants,
          scalar_dtype,
          exp_avg_sq.pt_t.scalar_type(),
          i + 2 * vec_size,
          i + 4 * vec_size);
      exp_avg_sq_1_out = std::move(std::get<0>(moment_and_scale));
      exp_avg_sq_scale_updated = std::move(std::get<1>(moment_and_scale));
    } else {
      exp_avg_sq_1_out = exp_avg_sq.pt_t.scalar_type() == scalar_dtype
          ? IdentityHelper(
                graph,
                exp_avg_sq_1[0].get(),
                exp_avg_sq.pt_t.sizes(),
                scalar_dtype,
                i + 2 * vec_size)
          : BuildCast(
                this,
                graph,
                exp_avg_sq_1[0].get(),
                exp_avg_sq.pt_t.sizes(),
                scalar_dtype,
                exp_avg_sq.pt_t.scalar_type(),
                i + 2 * vec_size);
    }

    auto exp_avg_sq_sqrt =
        BuildOp(graph, sqrt_node, {exp_avg_sq_1[0].get()}, exp_avg_sq_attr);

    auto denom = BuildOp(
        graph,
        add_node,
        {exp_avg_sq_sqrt[0].get(), epsilon_t},
        exp_avg_sq_attr);

    auto ratio = BuildOp(
        graph, div_node, {exp_avg_1[0].get(), denom[0].get()}, gradient_attr);

    auto scaled_ratio = BuildOp(
        graph, mul_node, {ratio[0].get(), neg_step_t.syn_t}, gradient_attr);

    auto weight_modified = weight.syn_t;
    if (has_weight_decay) {
      storage.push_back(std::move(BuildOp(
          graph,
          mul_node,
          {weight_modified, weight_decay.syn_t},
          weight_attr)[0]));
      weight_modified = storage.back().get();
    }

    auto result = BuildOp(
        graph,
        add_node,
        {weight_modified, scaled_ratio[0].get()},
        {NodeAttr::NodeOutputAttr{weight.pt_t.sizes(), scalar_dtype, i}});

    syn_out(i) = std::move(result[0]);
    syn_out(i + vec_size) = std::move(exp_avg_1_out.value());
    syn_out(i + 2 * vec_size) = std::move(exp_avg_sq_1_out.value());
    if (is_fp8) {
      syn_out(i + 3 * vec_size) = std::move(exp_avg_scale_updated.value());
      syn_out(i + 4 * vec_size) = std::move(exp_avg_sq_scale_updated.value());
    }
  }
}

} // namespace habana

static auto& OptimizerKernelsKernelRegistry = habana::KernelRegistry().add(
    "hpu::optimizer_adamw",
    KERNEL_FN(OptimizerFusedAdamWOperator));