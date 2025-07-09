/**
 * Copyright (c) 2021-2025 Intel Corporation
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
#include "generated/backend/_fused_adamw.h"
#include "hpu_ops/backend/reduction_template.h"
#include "hpu_ops/fp8_ops.h"
#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/op_backend.h"

namespace sh = synapse_helpers;

namespace habana {

using namespace std::literals;

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
      {get_guid_with_precision("abs"sv, original_dtype),
       {input.get()},
       {{pt_input.sizes().vec(), original_dtype}}});

  ns_Reduction::ParamsV2 reduce_params;
  reduce_params.reductionDimensionMask = 0;
  reduce_params.keepDim = false;

  auto amax = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("reduce_max_multi_dim_fwd"sv, original_dtype),
       {abs_input[0].get()},
       {{{1}, original_dtype}},
       &reduce_params,
       sizeof(reduce_params)});

  auto amax_div = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("div_fwd"sv, original_dtype),
       {constants[destination_dtype == c10::ScalarType::Float8_e4m3fn ? 2 : 3]
            .get(),
        amax[0].get()},
       {{{1}, original_dtype}}});

  auto amax_log = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("log2_fwd"sv, original_dtype),
       {amax_div[0].get()},
       {{{1}, original_dtype}}});
  auto exp = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("floor_fwd"sv, original_dtype),
       {amax_log[0].get()},
       {{{1}, original_dtype}}});

  auto new_scale = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("pow_fwd"sv, original_dtype),
       {constants[0].get(), exp[0].get()},
       {{{1}, original_dtype}}});

  auto mask = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("greater_fwd"sv, original_dtype),
       {new_scale[0].get(), constants[1].get()},
       {{{1}, torch::kBool}}});

  auto updated_scale = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("where_fwd"sv, original_dtype),
       {mask[0].get(), new_scale[0].get(), old_scale},
       {{{1}, original_dtype, out_scale}}});

  auto cast_params = GetCastParams(true, original_dtype, destination_dtype);
  auto result = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("convert_to_fp8"sv, original_dtype),
       {input.get(), updated_scale[0].get()},
       {{pt_input.sizes().vec(), destination_dtype, out_ids}},
       &cast_params,
       sizeof(cast_params)});

  return std::make_tuple(std::move(result[0]), std::move(updated_scale[0]));
}

SharedMetaDataVector GetMomentInFp8WithScaleSharedMeta(
    const at::Tensor& input,
    const at::Tensor& old_scale,
    const at::ScalarType& precision_type,
    const at::ScalarType& dest_dtype) {
  SharedMetaDataVector shared_meta_vec;
  shared_meta_vec.reserve(9);

  const auto input_rank = input.dim();
  SharedMetaData abs_shared_meta{"abs"};
  abs_shared_meta.inputs_data.emplace_back(input_rank, precision_type);
  abs_shared_meta.outputs_data = abs_shared_meta.inputs_data;
  shared_meta_vec.push_back(abs_shared_meta);

  SharedMetaData amax_shared_meta{"reduce_max_multi_dim_fwd"};
  amax_shared_meta.inputs_data = abs_shared_meta.outputs_data;
  amax_shared_meta.outputs_data.emplace_back(1, precision_type);
  shared_meta_vec.push_back(amax_shared_meta);

  SharedMetaTensor constant_tensor{1, precision_type};
  SharedMetaData amax_div_shared_meta{"div_fwd"};
  amax_div_shared_meta.inputs_data = {
      constant_tensor, amax_shared_meta.outputs_data[0]};
  amax_div_shared_meta.outputs_data = amax_shared_meta.outputs_data;
  shared_meta_vec.push_back(amax_div_shared_meta);

  SharedMetaData amax_log_shared_meta{"log2_fwd"};
  amax_log_shared_meta.inputs_data = amax_div_shared_meta.outputs_data;
  amax_log_shared_meta.outputs_data = amax_log_shared_meta.inputs_data;
  shared_meta_vec.push_back(amax_log_shared_meta);

  SharedMetaData exp_shared_meta{"floor_fwd"};
  exp_shared_meta.inputs_data = amax_log_shared_meta.outputs_data;
  exp_shared_meta.outputs_data = exp_shared_meta.inputs_data;
  shared_meta_vec.push_back(exp_shared_meta);

  SharedMetaData new_scale_shared_meta{"pow_fwd"};
  new_scale_shared_meta.inputs_data = {
      constant_tensor, exp_shared_meta.outputs_data[0]};
  new_scale_shared_meta.outputs_data = {new_scale_shared_meta.inputs_data[1]};
  shared_meta_vec.push_back(new_scale_shared_meta);

  SharedMetaData mask_shared_meta{"greater_fwd"};
  mask_shared_meta.inputs_data = {
      new_scale_shared_meta.outputs_data[0], constant_tensor};
  mask_shared_meta.outputs_data.emplace_back(1, at::ScalarType::Bool);
  shared_meta_vec.push_back(mask_shared_meta);

  SharedMetaData updated_scale_shared_meta{"where_fwd"};
  updated_scale_shared_meta.inputs_data = {
      mask_shared_meta.outputs_data[0],
      new_scale_shared_meta.outputs_data[0],
      {old_scale.dim(), precision_type}};
  updated_scale_shared_meta.outputs_data.emplace_back(1, precision_type);
  shared_meta_vec.push_back(updated_scale_shared_meta);

  SharedMetaData convert_to_fp8_shared_meta{"convert_to_fp8"};
  convert_to_fp8_shared_meta.inputs_data = {
      {input_rank, precision_type}, updated_scale_shared_meta.outputs_data[0]};
  convert_to_fp8_shared_meta.outputs_data.emplace_back(input_rank, dest_dtype);
  shared_meta_vec.push_back(convert_to_fp8_shared_meta);

  return shared_meta_vec;
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

SharedMetaDataVector OptimizerAdamWSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& gradient_vec = stack.at(0).toTensorVector();
  const auto& weight_vec = stack.at(1).toTensorVector();
  const auto& exp_avg_vec = stack.at(2).toTensorVector();
  const auto& exp_avg_sq_vec = stack.at(3).toTensorVector();
  const auto& neg_step_t = stack.at(4).toTensor();
  const auto& weight_decay = stack.at(8).toTensor();
  const auto has_weight_decay = stack.at(9).toBool();
  const auto& exp_avg_scales =
      stack.at(10).to<std::optional<std::vector<at::Tensor>>>();
  const auto& exp_avg_sq_scales =
      stack.at(11).to<std::optional<std::vector<at::Tensor>>>();
  auto precision_type = gradient_vec.front().scalar_type();
  const auto first_moment_dtype = exp_avg_vec.front().scalar_type();
  const bool is_fp8 = at::isFloat8Type(first_moment_dtype);
  precision_type = is_fp8
      ? precision_type
      : at::promote_types(precision_type, first_moment_dtype);

  SharedMetaTensor constant_tensor{1, precision_type};
  SharedMetaDataVector shared_meta_vec;
  size_t vec_size = gradient_vec.size();
  for (size_t i = 0; i < vec_size; ++i) {
    const auto& gradient = gradient_vec[i];
    const auto gradient_rank = gradient.dim();
    const auto& weight = weight_vec[i];
    const auto& exp_avg = exp_avg_vec[i];
    const auto exp_avg_rank = exp_avg.dim();
    const auto exp_avg_dtype = exp_avg.scalar_type();
    const auto& exp_avg_sq = exp_avg_sq_vec[i];
    const auto exp_avg_sq_rank = exp_avg_sq.dim();
    const auto exp_avg_sq_dtype = exp_avg_sq.scalar_type();

    if (is_fp8) {
      if (exp_avg_scales.has_value()) {
        const auto& exp_avg_scale = exp_avg_scales.value()[i];
        SharedMetaData exp_avg_convert_from_fp8_shared_meta{"convert_from_fp8"};
        exp_avg_convert_from_fp8_shared_meta.inputs_data.emplace_back(
            exp_avg_rank, exp_avg_dtype);
        exp_avg_convert_from_fp8_shared_meta.inputs_data.emplace_back(
            exp_avg_scale.dim(), exp_avg_scale.scalar_type());
        exp_avg_convert_from_fp8_shared_meta.outputs_data.emplace_back(
            exp_avg_rank, precision_type);
        shared_meta_vec.push_back(exp_avg_convert_from_fp8_shared_meta);
      }
      if (exp_avg_sq_scales.has_value()) {
        const auto& exp_avg_sq_scale = exp_avg_sq_scales.value()[i];
        SharedMetaData exp_avg_sq_convert_from_fp8_shared_meta{
            "convert_from_fp8"};
        exp_avg_sq_convert_from_fp8_shared_meta.inputs_data.emplace_back(
            exp_avg_sq_rank, exp_avg_sq_dtype);
        exp_avg_sq_convert_from_fp8_shared_meta.inputs_data.emplace_back(
            exp_avg_sq_scale.dim(), exp_avg_sq_scale.scalar_type());
        exp_avg_sq_convert_from_fp8_shared_meta.outputs_data.emplace_back(
            exp_avg_sq_rank, precision_type);
        shared_meta_vec.push_back(exp_avg_sq_convert_from_fp8_shared_meta);
      }
    }

    SharedMetaData exp_avg_mul_beta_shared_meta{"mult_fwd"};
    exp_avg_mul_beta_shared_meta.inputs_data.emplace_back(
        exp_avg_rank, precision_type);
    exp_avg_mul_beta_shared_meta.inputs_data.push_back(constant_tensor);
    exp_avg_mul_beta_shared_meta.outputs_data.emplace_back(
        exp_avg_rank, precision_type);
    shared_meta_vec.push_back(exp_avg_mul_beta_shared_meta);

    SharedMetaData grad_scaled_shared_meta{"mult_fwd"};
    grad_scaled_shared_meta.inputs_data.emplace_back(
        gradient_rank, precision_type);
    grad_scaled_shared_meta.inputs_data.push_back(constant_tensor);
    grad_scaled_shared_meta.outputs_data.emplace_back(
        gradient_rank, precision_type);
    shared_meta_vec.push_back(grad_scaled_shared_meta);

    SharedMetaData exp_avg_add_shared_meta{"add_fwd"};
    exp_avg_add_shared_meta.inputs_data.push_back(
        exp_avg_mul_beta_shared_meta.outputs_data[0]);
    exp_avg_add_shared_meta.inputs_data.push_back(
        grad_scaled_shared_meta.outputs_data[0]);
    exp_avg_add_shared_meta.outputs_data = grad_scaled_shared_meta.outputs_data;
    shared_meta_vec.push_back(exp_avg_add_shared_meta);

    if (is_fp8 && exp_avg_scales.has_value()) {
      const auto& exp_avg_scale = exp_avg_scales.value()[i];
      auto fp8_with_scale_meta_vec = GetMomentInFp8WithScaleSharedMeta(
          exp_avg, exp_avg_scale, precision_type, exp_avg_dtype);
      shared_meta_vec.insert(
          std::end(shared_meta_vec),
          std::begin(fp8_with_scale_meta_vec),
          std::end(fp8_with_scale_meta_vec));
    }
    SharedMetaData gradient_sq_shared_meta{"mult_fwd"};
    gradient_sq_shared_meta.inputs_data = {
        {gradient_rank, precision_type}, {gradient_rank, precision_type}};
    gradient_sq_shared_meta.outputs_data.emplace_back(
        gradient_rank, precision_type);
    shared_meta_vec.push_back(gradient_sq_shared_meta);

    SharedMetaData gradient_sq_scaled_shared_meta{"mult_fwd"};
    gradient_sq_scaled_shared_meta.inputs_data = {
        gradient_sq_shared_meta.outputs_data[0], constant_tensor};
    gradient_sq_scaled_shared_meta.outputs_data =
        gradient_sq_shared_meta.outputs_data;
    shared_meta_vec.push_back(gradient_sq_scaled_shared_meta);

    SharedMetaData exp_avg_sq_mul_beta_shared_meta{"mult_fwd"};
    exp_avg_sq_mul_beta_shared_meta.inputs_data = {
        {exp_avg_sq_rank, precision_type}, constant_tensor};
    exp_avg_sq_mul_beta_shared_meta.outputs_data.emplace_back(
        exp_avg_sq_rank, precision_type);
    shared_meta_vec.push_back(exp_avg_sq_mul_beta_shared_meta);

    SharedMetaData exp_avg_sq_add_shared_meta{"add_fwd"};
    exp_avg_sq_add_shared_meta.inputs_data = {
        exp_avg_sq_mul_beta_shared_meta.outputs_data[0],
        gradient_sq_scaled_shared_meta.outputs_data[0]};
    exp_avg_sq_add_shared_meta.outputs_data.emplace_back(
        gradient_rank, precision_type);
    shared_meta_vec.push_back(exp_avg_sq_add_shared_meta);

    if (is_fp8 && exp_avg_sq_scales.has_value()) {
      const auto& exp_avg_sq_scale = exp_avg_sq_scales.value()[i];
      auto fp8_with_scale_meta_vec = GetMomentInFp8WithScaleSharedMeta(
          exp_avg_sq, exp_avg_sq_scale, precision_type, exp_avg_sq_dtype);
      shared_meta_vec.insert(
          std::end(shared_meta_vec),
          std::begin(fp8_with_scale_meta_vec),
          std::end(fp8_with_scale_meta_vec));
    }

    SharedMetaData exp_avg_sq_sqrt_shared_meta{"sqrt_fwd"};
    exp_avg_sq_sqrt_shared_meta.inputs_data.emplace_back(
        exp_avg_sq_rank, precision_type);
    exp_avg_sq_sqrt_shared_meta.outputs_data =
        exp_avg_sq_sqrt_shared_meta.inputs_data;
    shared_meta_vec.push_back(exp_avg_sq_sqrt_shared_meta);

    SharedMetaData denom_shared_meta{"add_fwd"};
    denom_shared_meta.inputs_data = {
        exp_avg_sq_sqrt_shared_meta.outputs_data[0], constant_tensor};
    denom_shared_meta.outputs_data = exp_avg_sq_sqrt_shared_meta.outputs_data;
    shared_meta_vec.push_back(denom_shared_meta);

    SharedMetaData ratio_shared_meta{"div_fwd"};
    ratio_shared_meta.inputs_data = {
        exp_avg_add_shared_meta.outputs_data[0],
        denom_shared_meta.outputs_data[0]};
    ratio_shared_meta.outputs_data = exp_avg_add_shared_meta.outputs_data;
    shared_meta_vec.push_back(ratio_shared_meta);

    SharedMetaData scaled_ratio_shared_meta{"mult_fwd"};
    scaled_ratio_shared_meta.inputs_data = {
        ratio_shared_meta.outputs_data[0], {neg_step_t.dim(), precision_type}};
    scaled_ratio_shared_meta.outputs_data = ratio_shared_meta.outputs_data;
    shared_meta_vec.push_back(scaled_ratio_shared_meta);

    if (has_weight_decay) {
      SharedMetaData weight_mul_shared_meta{"mult_fwd"};
      weight_mul_shared_meta.inputs_data.emplace_back(
          weight.dim(), precision_type);
      weight_mul_shared_meta.inputs_data.emplace_back(
          weight_decay.dim(), precision_type);
      weight_mul_shared_meta.outputs_data.push_back(
          weight_mul_shared_meta.inputs_data[0]);
      shared_meta_vec.push_back(weight_mul_shared_meta);

      SharedMetaData result_shared_meta{"add_fwd"};
      result_shared_meta.inputs_data = {
          weight_mul_shared_meta.outputs_data[0],
          scaled_ratio_shared_meta.outputs_data[0]};
      result_shared_meta.outputs_data = weight_mul_shared_meta.outputs_data;
      shared_meta_vec.push_back(result_shared_meta);
    } else {
      SharedMetaData result_shared_meta{"add_fwd"};
      result_shared_meta.inputs_data = {
          {weight.dim(), precision_type},
          scaled_ratio_shared_meta.outputs_data[0]};
      result_shared_meta.outputs_data.push_back(
          result_shared_meta.inputs_data[0]);
      shared_meta_vec.push_back(result_shared_meta);
    }
  }
  return shared_meta_vec;
}

void OptimizerFusedAdamWOperator::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "OptimizerFusedAdamWOperator::AddNode");
  auto gradient_vec = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto weight_vec = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto exp_avg_vec = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto exp_avg_sq_vec = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto neg_step_t = stackGetter.getNextInput<TensorsPair>();
  auto beta1 = stackGetter.getNextInput<double>();
  auto beta2 = stackGetter.getNextInput<double>();
  auto epsilon = stackGetter.getNextInput<double>();
  auto weight_decay = stackGetter.getNextInput<TensorsPair>();
  auto has_weight_decay = stackGetter.getNextInput<bool>();
  auto exp_avg_scales =
      stackGetter.getNextInput<std::optional<std::vector<TensorsPair>>>();
  auto exp_avg_sq_scales =
      stackGetter.getNextInput<std::optional<std::vector<TensorsPair>>>();

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

  std::string add_node = get_guid_with_precision("add_fwd"sv, scalar_dtype);
  std::string mul_node = get_guid_with_precision("mult_fwd"sv, scalar_dtype);
  std::string div_node = get_guid_with_precision("div_fwd"sv, scalar_dtype);
  std::string sqrt_node = get_guid_with_precision("sqrt_fwd"sv, scalar_dtype);
  std::string from_fp8_node =
      get_guid_with_precision("convert_from_fp8"sv, scalar_dtype);

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

    std::optional<synTensor> exp_avg_scale_syn, exp_avg_sq_scale_syn;
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
      storage.push_back(
          std::move(BuildOp(
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

SharedMetaDataVector AdamWSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& self_vec = stack.at(0).toTensorVector();
  const auto& grads_vec = stack.at(1).toTensorVector();
  const auto& exp_avgs_vec = stack.at(2).toTensorVector();
  const auto& exp_avg_sqs_vec = stack.at(3).toTensorVector();
  const auto& max_exp_avg_sqs_vec = stack.at(4).toTensorVector();
  const auto& state_steps_vec = stack.at(5).toTensorVector();
  [[maybe_unused]] const auto lr = stack.at(6).toDouble();
  [[maybe_unused]] const auto beta1 = stack.at(7).toDouble();
  [[maybe_unused]] const auto beta2 = stack.at(8).toDouble();
  [[maybe_unused]] const auto weight_decay = stack.at(9).toDouble();
  [[maybe_unused]] const auto eps = stack.at(10).toDouble();
  [[maybe_unused]] const auto amsgrad = stack.at(11).toBool();
  [[maybe_unused]] const auto maximize = stack.at(12).toBool();
  [[maybe_unused]] const auto& grad_scale =
      stack.at(13).to<std::optional<at::Tensor>>();
  [[maybe_unused]] const auto& found_inf =
      stack.at(14).to<std::optional<at::Tensor>>();

  auto precision_type = grads_vec.front().scalar_type();
  const auto first_moment_dtype = exp_avgs_vec.front().scalar_type();
  precision_type = at::promote_types(precision_type, first_moment_dtype);

  SharedMetaTensor constant_tensor{1, precision_type};
  SharedMetaDataVector shared_meta_vec;
  size_t vec_size = grads_vec.size();
  for (size_t i = 0; i < vec_size; ++i) {
    const auto& gradient = grads_vec[i];
    const auto gradient_rank = gradient.dim();
    const auto& weight = self_vec[i];
    const auto weight_rank = weight.dim();

    const auto& exp_avg = exp_avgs_vec[i];
    const auto exp_avg_rank = exp_avg.dim();

    const auto& exp_avg_sq = exp_avg_sqs_vec[i];
    const auto exp_avg_sq_rank = exp_avg_sq.dim();

    //------will not calculate maximize part----------//
    //------calculate weight----------//
    // theta_t = theta_t_1 - gamma * weight_decay *theta_t_1
    SharedMetaData weight_lr_shared_meta{"mult_fwd"};
    weight_lr_shared_meta.inputs_data.emplace_back(weight_rank, precision_type);
    weight_lr_shared_meta.inputs_data.push_back(constant_tensor);
    weight_lr_shared_meta.outputs_data.emplace_back(
        weight_rank, precision_type);
    shared_meta_vec.push_back(weight_lr_shared_meta);

    SharedMetaData weight_shared_meta{"sub_fwd"};
    weight_shared_meta.inputs_data.emplace_back(weight_rank, precision_type);
    weight_shared_meta.inputs_data.push_back(
        weight_lr_shared_meta.outputs_data[0]);
    weight_shared_meta.outputs_data.emplace_back(weight_rank, precision_type);
    shared_meta_vec.push_back(weight_shared_meta);

    //------calculate first moment----------//
    // m_t = beta1 * m_t_1 + (1-beta1) * g_t
    // here beta1 and (1-beta1) we will count as a constant tensors
    SharedMetaData exp_avg_mul_beta_shared_meta{"mult_fwd"};
    exp_avg_mul_beta_shared_meta.inputs_data.emplace_back(
        exp_avg_rank, precision_type);
    exp_avg_mul_beta_shared_meta.inputs_data.push_back(constant_tensor);
    exp_avg_mul_beta_shared_meta.outputs_data.emplace_back(
        exp_avg_rank, precision_type);
    shared_meta_vec.push_back(exp_avg_mul_beta_shared_meta);

    SharedMetaData grad_scaled_shared_meta{"mult_fwd"};
    grad_scaled_shared_meta.inputs_data.emplace_back(
        gradient_rank, precision_type);
    grad_scaled_shared_meta.inputs_data.push_back(constant_tensor);
    grad_scaled_shared_meta.outputs_data.emplace_back(
        gradient_rank, precision_type);
    shared_meta_vec.push_back(grad_scaled_shared_meta);

    SharedMetaData exp_avg_add_shared_meta{"add_fwd"};
    exp_avg_add_shared_meta.inputs_data.push_back(
        exp_avg_mul_beta_shared_meta.outputs_data[0]);
    exp_avg_add_shared_meta.inputs_data.push_back(
        grad_scaled_shared_meta.outputs_data[0]);
    exp_avg_add_shared_meta.outputs_data = grad_scaled_shared_meta.outputs_data;
    shared_meta_vec.push_back(exp_avg_add_shared_meta);

    //------calculate second moment----------//
    // v_t = beta2 * v_t_1 + (1-beta2) * g_t * g_t
    // here beta2 and (1-beta2) we will count as a constant tensors
    SharedMetaData gradient_sq_shared_meta{"mult_fwd"};
    gradient_sq_shared_meta.inputs_data = {
        {gradient_rank, precision_type}, {gradient_rank, precision_type}};
    gradient_sq_shared_meta.outputs_data.emplace_back(
        gradient_rank, precision_type);
    shared_meta_vec.push_back(gradient_sq_shared_meta);

    SharedMetaData gradient_sq_scaled_shared_meta{"mult_fwd"};
    gradient_sq_scaled_shared_meta.inputs_data = {
        gradient_sq_shared_meta.outputs_data[0], constant_tensor};
    gradient_sq_scaled_shared_meta.outputs_data =
        gradient_sq_shared_meta.outputs_data;
    shared_meta_vec.push_back(gradient_sq_scaled_shared_meta);

    SharedMetaData exp_avg_sq_mul_beta_shared_meta{"mult_fwd"};
    exp_avg_sq_mul_beta_shared_meta.inputs_data = {
        {exp_avg_sq_rank, precision_type}, constant_tensor};
    exp_avg_sq_mul_beta_shared_meta.outputs_data.emplace_back(
        exp_avg_sq_rank, precision_type);
    shared_meta_vec.push_back(exp_avg_sq_mul_beta_shared_meta);

    SharedMetaData exp_avg_sq_add_shared_meta{"add_fwd"};
    exp_avg_sq_add_shared_meta.inputs_data = {
        exp_avg_sq_mul_beta_shared_meta.outputs_data[0],
        gradient_sq_scaled_shared_meta.outputs_data[0]};
    exp_avg_sq_add_shared_meta.outputs_data.emplace_back(
        gradient_rank, precision_type);
    shared_meta_vec.push_back(exp_avg_sq_add_shared_meta);

    //------calculate m^------------------------//
    // m^ = m_t/(1-beta_1^t)
    SharedMetaData exp_avg_cap_shared_meta{"div_fwd"};
    exp_avg_cap_shared_meta.inputs_data = {
        exp_avg_add_shared_meta.outputs_data[0], constant_tensor};
    exp_avg_cap_shared_meta.outputs_data = exp_avg_add_shared_meta.outputs_data;
    shared_meta_vec.push_back(exp_avg_cap_shared_meta);

    //------calculate only amsgrad else part---------------//
    //------calculate v^-------------------//
    // v_t^ = v_t/(1-beta_2^t)
    SharedMetaData exp_avg_sq_cap_shared_meta{"div_fwd"};
    exp_avg_sq_cap_shared_meta.inputs_data = {
        exp_avg_sq_add_shared_meta.outputs_data[0], constant_tensor};
    exp_avg_sq_cap_shared_meta.outputs_data =
        exp_avg_sq_add_shared_meta.outputs_data;
    shared_meta_vec.push_back(exp_avg_sq_cap_shared_meta);

    //------calculate final weight------------//
    // theta_t = theta_t - gamma * m_t^/(sqrt(v_t^)+epsilon)

    SharedMetaData exp_avg_sq_sqrt_shared_meta{"sqrt_fwd"};
    exp_avg_sq_sqrt_shared_meta.inputs_data = {
        exp_avg_sq_cap_shared_meta.outputs_data[0]};
    exp_avg_sq_sqrt_shared_meta.outputs_data =
        exp_avg_sq_cap_shared_meta.outputs_data;
    shared_meta_vec.push_back(exp_avg_sq_sqrt_shared_meta);

    SharedMetaData denom_shared_meta{"add_fwd"};
    denom_shared_meta.inputs_data = {
        exp_avg_sq_sqrt_shared_meta.outputs_data[0], constant_tensor};
    denom_shared_meta.outputs_data = exp_avg_sq_sqrt_shared_meta.outputs_data;
    shared_meta_vec.push_back(denom_shared_meta);

    SharedMetaData ratio_shared_meta{"div_fwd"};
    ratio_shared_meta.inputs_data = {
        exp_avg_cap_shared_meta.outputs_data[0],
        denom_shared_meta.outputs_data[0]};
    ratio_shared_meta.outputs_data = exp_avg_cap_shared_meta.outputs_data;
    shared_meta_vec.push_back(ratio_shared_meta);

    SharedMetaData scaled_ratio_shared_meta{"mult_fwd"};
    scaled_ratio_shared_meta.inputs_data = {
        ratio_shared_meta.outputs_data[0], constant_tensor};
    scaled_ratio_shared_meta.outputs_data = ratio_shared_meta.outputs_data;
    shared_meta_vec.push_back(scaled_ratio_shared_meta);

    SharedMetaData final_weight_shared_meta{"sub_fwd"};
    final_weight_shared_meta.inputs_data = {
        weight_shared_meta.outputs_data[0],
        scaled_ratio_shared_meta.outputs_data[0]};
    final_weight_shared_meta.outputs_data.emplace_back(
        weight_rank, precision_type);
    shared_meta_vec.push_back(final_weight_shared_meta);
  }
  return shared_meta_vec;
}

// In this function, we are changing the order of the PT output
// tensors to be in the zigzag order to match the order of the synapse
// output tensors.
void ZigzagOutputTensor(OpBackend& op, size_t vec_size) {
  auto& outputInfMeta = op.GetOutputInfMeta();
  auto outputInfMeta1 = op.GetOutputInfMeta();
  auto length = outputInfMeta.GetOutputTensor().size();

  size_t j = 0;
  size_t step = static_cast<int>(length / vec_size);
  for (size_t k = 0; k < step; k++) {
    for (size_t i = k; i < length; i += step) {
      auto t = outputInfMeta1.GetOutputTensor(i);
      outputInfMeta.InsertOutputIdx(j, t);
      j++;
    }
  }
}

void FusedAdamW::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "FusedAdamW::AddNode");
  auto self = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto grads = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto exp_avgs = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto exp_avg_sqs = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto max_exp_avg_sqs = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto state_steps = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto lr = stackGetter.getNextInput<double>();
  auto beta1 = stackGetter.getNextInput<double>();
  auto beta2 = stackGetter.getNextInput<double>();
  auto weight_decay = stackGetter.getNextInput<double>();
  auto eps = stackGetter.getNextInput<double>();
  auto amsgrad = stackGetter.getNextInput<bool>();
  auto maximize = stackGetter.getNextInput<bool>();
  auto grad_scale = stackGetter.getNextInput<std::optional<TensorsPair>>();
  auto found_inf = stackGetter.getNextInput<std::optional<TensorsPair>>();
  if (grad_scale) {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "not supported grad_scale yet"); // will handle later, change
                                         // accordingly in AdamWSharedMeta
  }
  if (found_inf) {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false, "not supported found_inf yet"); // will handle later, change
                                               // accordingly in AdamWSharedMeta
  }
  if ((grads.size() != self.size()) || (grads.size() != exp_avgs.size()) ||
      (grads.size() != exp_avg_sqs.size())) {
    std::stringstream ss;
    ss << "All 4 vector inputs must have the same number of elements but they respectively have: "
       << grads.size() << ", " << self.size() << ", " << exp_avgs.size() << ", "
       << exp_avg_sqs.size();
    AT_ERROR(ss.str());
  }

  const auto& first_moment_dtype = exp_avgs.front().pt_t.scalar_type();
  const auto scalar_dtype = at::promote_types(ScalarType(), first_moment_dtype);

  std::string add_node = get_guid_with_precision("add_fwd"sv, scalar_dtype);
  std::string sub_node = get_guid_with_precision("sub_fwd"sv, scalar_dtype);
  std::string mul_node = get_guid_with_precision("mult_fwd"sv, scalar_dtype);
  std::string div_node = get_guid_with_precision("div_fwd"sv, scalar_dtype);
  std::string sqrt_node = get_guid_with_precision("sqrt_fwd"sv, scalar_dtype);
  std::string pow_node = get_guid_with_precision("pow_fwd"sv, scalar_dtype);

  int64_t scalar_shape[] = {1};

  double constant_values[] = {
      1.0,
      beta1,
      beta2,
      1.0 - beta1,
      1.0 - beta2,
      lr,
      1 - lr * weight_decay,
      eps};

  size_t vec_size = grads.size();
  for (size_t i = 0; i < vec_size; ++i) {
    const auto& gradient = grads[i];
    const auto& weight = self[i];
    const auto& exp_avg = exp_avgs[i];
    const auto& exp_avg_sq = exp_avg_sqs[i];
    const auto& state_step = state_steps[i];
    const auto& one_t = ConstantHelper(
        graph,
        static_cast<float>(constant_values[0]),
        scalar_dtype,
        scalar_shape);
    const auto& beta1_t = ConstantHelper(
        graph,
        static_cast<float>(constant_values[1]),
        scalar_dtype,
        scalar_shape);
    const auto& beta2_t = ConstantHelper(
        graph,
        static_cast<float>(constant_values[2]),
        scalar_dtype,
        scalar_shape);
    const auto& one_minus_beta1_t = ConstantHelper(
        graph,
        static_cast<float>(constant_values[3]),
        scalar_dtype,
        scalar_shape);
    const auto& one_minus_beta2_t = ConstantHelper(
        graph,
        static_cast<float>(constant_values[4]),
        scalar_dtype,
        scalar_shape);
    const auto& lr_t = ConstantHelper(
        graph,
        static_cast<float>(constant_values[5]),
        scalar_dtype,
        scalar_shape);
    const auto& one_minus_lr_wt_decay_t = ConstantHelper(
        graph,
        static_cast<float>(constant_values[6]),
        scalar_dtype,
        scalar_shape);
    const auto& epsilon_t = ConstantHelper(
        graph,
        static_cast<float>(constant_values[7]),
        scalar_dtype,
        scalar_shape);

    std::vector<NodeAttr::NodeOutputAttr> gradient_attr = {
        {gradient.pt_t.sizes(), scalar_dtype}};
    std::vector<NodeAttr::NodeOutputAttr> weight_attr = {
        {weight.pt_t.sizes(), scalar_dtype}};
    std::vector<NodeAttr::NodeOutputAttr> exp_avg_attr = {
        {exp_avg.pt_t.sizes(), scalar_dtype}};
    std::vector<NodeAttr::NodeOutputAttr> exp_avg_sq_attr = {
        {exp_avg_sq.pt_t.sizes(), scalar_dtype}};
    std::vector<NodeAttr::NodeOutputAttr> scalar_1d_attr = {
        {scalar_shape, scalar_dtype}};

    //------calculate maximize part----------//
    auto grad = IdentityHelper(
        graph,
        gradient.syn_t,
        gradient.pt_t.sizes().vec(),
        scalar_dtype,
        i + vec_size);
    if (maximize) {
      // will handle later
      TORCH_CHECK_NOT_IMPLEMENTED(false, "not supported maximize yet");
    }

    //------calculate weight----------//
    // theta_t = theta_t_1 - gamma * weight_decay *theta_t_1
    auto weight_1 = BuildOp(
        graph,
        mul_node,
        {weight.syn_t, one_minus_lr_wt_decay_t.get()},
        weight_attr);

    //------calculate first moment----------//
    // m_t = beta1 * m_t_1 + (1-beta1) * g_t
    auto exp_avg_mul_beta1 =
        BuildOp(graph, mul_node, {exp_avg.syn_t, beta1_t.get()}, exp_avg_attr);

    auto grad_scaled = BuildOp(
        graph, mul_node, {grad.get(), one_minus_beta1_t.get()}, gradient_attr);

    auto exp_avg_1 = BuildOp(
        graph,
        add_node,
        {exp_avg_mul_beta1[0].get(), grad_scaled[0].get()},
        {{exp_avg.pt_t.sizes(), scalar_dtype, i + 2 * vec_size}});

    //------calculate second moment----------//
    // v_t = beta2 * v_t_1 + (1-beta2) * g_t * g_t
    // here beta2 and (1-beta2) we will count as a constant tensors
    auto grad_sq =
        BuildOp(graph, mul_node, {grad.get(), grad.get()}, gradient_attr);

    auto grad_sq_scaled = BuildOp(
        graph,
        mul_node,
        {grad_sq[0].get(), one_minus_beta2_t.get()},
        gradient_attr);

    auto exp_avg_sq_mul_beta2 = BuildOp(
        graph, mul_node, {exp_avg_sq.syn_t, beta2_t.get()}, exp_avg_sq_attr);

    auto exp_avg_sq_1 = BuildOp(
        graph,
        add_node,
        {exp_avg_sq_mul_beta2[0].get(), grad_sq_scaled[0].get()},
        {{gradient.pt_t.sizes(), scalar_dtype, i + 3 * vec_size}});

    //------calculate m^------------------------//
    // m^ = m_t/(1-beta_1^t) // need to handle power of t properly
    auto beta1_n = BuildOp(
        graph, pow_node, {beta1_t.get(), state_step.syn_t}, scalar_1d_attr);

    auto one_minus_beta1_n = BuildOp(
        graph, sub_node, {one_t.get(), beta1_n[0].get()}, scalar_1d_attr);

    auto exp_avg_cap = BuildOp(
        graph,
        div_node,
        {exp_avg_1[0].get(), one_minus_beta1_n[0].get()},
        gradient_attr);

    //------calculate amsgrad part---------------//
    auto beta2_n = BuildOp(
        graph, pow_node, {beta2_t.get(), state_step.syn_t}, scalar_1d_attr);

    auto one_minus_beta2_n = BuildOp(
        graph, sub_node, {one_t.get(), beta2_n[0].get()}, scalar_1d_attr);

    if (amsgrad) {
      TORCH_CHECK_NOT_IMPLEMENTED(false, "not supported amsgrad yet");
      // will handle later
    } else {
      // will handle later
    }
    //------calculate v^-------------------//
    // v_t^ = v_t/(1-beta_2^t) // need to handle power of t
    auto exp_avg_sq_cap = BuildOp(
        graph,
        div_node,
        {exp_avg_sq_1[0].get(), one_minus_beta2_n[0].get()},
        gradient_attr);

    //------calculate final weight------------//
    // theta_t = theta_t - gamma * m_t^/(sqrt(v_t^)+epsilon)
    auto exp_avg_sq_cap_sqrt =
        BuildOp(graph, sqrt_node, {exp_avg_sq_cap[0].get()}, exp_avg_sq_attr);

    auto denom = BuildOp(
        graph,
        add_node,
        {exp_avg_sq_cap_sqrt[0].get(), epsilon_t.get()},
        exp_avg_sq_attr);

    auto lr_exp_avg_cap = BuildOp(
        graph, mul_node, {exp_avg_cap[0].get(), lr_t.get()}, exp_avg_sq_attr);

    auto ratio = BuildOp(
        graph,
        div_node,
        {lr_exp_avg_cap[0].get(), denom[0].get()},
        gradient_attr);

    auto result = BuildOp(
        graph,
        sub_node,
        {weight_1[0].get(), ratio[0].get()},
        {NodeAttr::NodeOutputAttr{weight.pt_t.sizes(), scalar_dtype, i}});

    syn_out(i) = std::move(result[0]);
    syn_out(i + vec_size) = std::move(grad);
    syn_out(i + 2 * vec_size) = std::move(exp_avg_1[0]);
    syn_out(i + 3 * vec_size) = std::move(exp_avg_sq_1[0]);
    if (!max_exp_avg_sqs.empty()) {
      TORCH_CHECK_NOT_IMPLEMENTED(false, "not supported max_exp_avg_sqs yet");
    }
  }
  if (isOutputInfMode()) {
    ZigzagOutputTensor(*this, vec_size);
  }
}

} // namespace habana

static auto& OptimizerKernelsKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu::optimizer_adamw",
        habana::OptimizerFusedAdamWOperator);
