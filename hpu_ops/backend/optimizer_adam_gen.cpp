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
#include "generated/backend/_fused_adam.h"
#include "hpu_ops/backend/reduction_template.h"
#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/op_backend.h"

namespace sh = synapse_helpers;

namespace habana {

using namespace std::literals;

SharedMetaDataVector AdamSharedMetaCommon(
    const std::vector<at::Tensor>& self_vec,
    const std::vector<at::Tensor>& grads_vec,
    const std::vector<at::Tensor>& exp_avgs_vec,
    const std::vector<at::Tensor>& exp_avg_sqs_vec,
    [[maybe_unused]] const std::vector<at::Tensor>& max_exp_avg_sqs_vec,
    [[maybe_unused]] const std::vector<at::Tensor>& state_steps_vec,
    [[maybe_unused]] std::optional<at::Tensor>& lr_tensor,
    double weight_decay) {
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
    //------calculate grad when weight decay not equal to zero----------//
    SharedMetaData grad_shared_meta{"copy_fwd"};
    if (weight_decay != 0) {
      // g_t = g_t + weight_decay * theta_t_1
      SharedMetaData weight_lr_shared_meta{"mult_fwd"};
      weight_lr_shared_meta.inputs_data.emplace_back(
          weight_rank, precision_type);
      weight_lr_shared_meta.inputs_data.push_back(constant_tensor);
      weight_lr_shared_meta.outputs_data.emplace_back(
          weight_rank, precision_type);
      shared_meta_vec.push_back(weight_lr_shared_meta);

      SharedMetaData weight_shared_meta{"add_fwd"};
      weight_shared_meta.inputs_data.emplace_back(
          gradient_rank, precision_type);
      weight_shared_meta.inputs_data.push_back(
          weight_lr_shared_meta.outputs_data[0]);
      weight_shared_meta.outputs_data.emplace_back(
          gradient_rank, precision_type);
      shared_meta_vec.push_back(weight_shared_meta);

      grad_shared_meta.inputs_data.emplace_back(
          weight_shared_meta.outputs_data[0]);
      grad_shared_meta.outputs_data.emplace_back(gradient_rank, precision_type);
      shared_meta_vec.push_back(grad_shared_meta);
    } else {
      // g_t = g_t
      grad_shared_meta.inputs_data.emplace_back(gradient_rank, precision_type);
      grad_shared_meta.outputs_data.emplace_back(gradient_rank, precision_type);
      shared_meta_vec.push_back(grad_shared_meta);
    }

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
        grad_shared_meta.outputs_data[0]);
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
        {grad_shared_meta.outputs_data[0]}, {grad_shared_meta.outputs_data[0]}};
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

    //------calculate only amsgrad else part for simplicity---------------//
    //------calculate v^-------------------//
    // v_t^ = v_t/(1-beta_2^t)
    SharedMetaData exp_avg_sq_cap_shared_meta{"div_fwd"};
    exp_avg_sq_cap_shared_meta.inputs_data = {
        exp_avg_sq_add_shared_meta.outputs_data[0], constant_tensor};
    exp_avg_sq_cap_shared_meta.outputs_data =
        exp_avg_sq_add_shared_meta.outputs_data;
    shared_meta_vec.push_back(exp_avg_sq_cap_shared_meta);

    //------calculate final weight------------//
    // theta_t = theta_t_1 - gamma * m_t^/(sqrt(v_t^)+epsilon)

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
    if (lr_tensor.has_value() && lr_tensor.value().defined()) {
      scaled_ratio_shared_meta.inputs_data.emplace_back(
          ratio_shared_meta.outputs_data[0]);
      scaled_ratio_shared_meta.inputs_data.emplace_back(
          lr_tensor.value().dim(), precision_type);
    } else {
      scaled_ratio_shared_meta.inputs_data = {
          ratio_shared_meta.outputs_data[0], constant_tensor};
    }
    scaled_ratio_shared_meta.outputs_data = ratio_shared_meta.outputs_data;
    shared_meta_vec.push_back(scaled_ratio_shared_meta);

    SharedMetaData final_weight_shared_meta{"sub_fwd"};
    final_weight_shared_meta.inputs_data.emplace_back(
        weight_rank, precision_type);
    final_weight_shared_meta.inputs_data.push_back(
        scaled_ratio_shared_meta.outputs_data[0]);
    final_weight_shared_meta.outputs_data.emplace_back(
        weight_rank, precision_type);
    shared_meta_vec.push_back(final_weight_shared_meta);
  }
  return shared_meta_vec;
}

SharedMetaDataVector AdamSharedMeta(
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
  std::optional<at::Tensor> lr_tensor = std::nullopt;
  SharedMetaDataVector shared_meta_vec = AdamSharedMetaCommon(
      self_vec,
      grads_vec,
      exp_avgs_vec,
      exp_avg_sqs_vec,
      max_exp_avg_sqs_vec,
      state_steps_vec,
      lr_tensor,
      weight_decay);
  return shared_meta_vec;
}
// followed implementation : aten/src/ATen/native/cpu/FusedAdamKernel.cpp
std::vector<synapse_helpers::tensor> FusedAdamCommon(
    OpBackend* op,
    sh::graph& graph,
    const TensorsPair& weight,
    const TensorsPair& gradient,
    const TensorsPair& exp_avg,
    const TensorsPair& exp_avg_sq,
    const TensorsPair& max_exp_avg_sq,
    const TensorsPair& state_step,
    synTensor lr_t,
    synTensor beta1_t,
    synTensor beta2_t,
    synTensor wt_decay_t,
    double weight_decay_s,
    synTensor epsilon_t,
    synTensor one_t,
    synTensor one_minus_beta1_t,
    synTensor one_minus_beta2_t,
    bool amsgrad,
    bool maximize,
    [[maybe_unused]] std::optional<TensorsPair> grad_scale,
    [[maybe_unused]] std::optional<TensorsPair> found_inf,
    c10::ScalarType scalar_dtype,
    int64_t vec_size,
    int64_t i,
    bool empty_max_exp_avg_sqs) {
  std::vector<synapse_helpers::tensor> output;
  int64_t scalar_shape[] = {1};
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
  std::optional<int> finalIndex =
      !(maximize) ? c10::make_optional<int>(i + vec_size) : std::nullopt;
  std::vector<synapse_helpers::tensor> grad_old = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("identity"sv, scalar_dtype),
       {gradient.syn_t},
       {{gradient.pt_t.sizes(), scalar_dtype, finalIndex}}});
  if (maximize) {
    grad_old = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("neg_fwd"sv, scalar_dtype),
         {grad_old[0].get()},
         {{gradient.pt_t.sizes(), scalar_dtype}}});
  }

  //------calculate grad when weight decay not equal to zero----------//
  // g_t = g_t + weight_decay * theta_t_1
  std::vector<synapse_helpers::tensor> grad;
  if (weight_decay_s != 0) {
    // g_t = g_t + weight_decay * theta_t_1
    auto weight_decay = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("mult_fwd"sv, scalar_dtype),
         {weight.syn_t, wt_decay_t},
         weight_attr});

    grad = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("add_fwd"sv, scalar_dtype),
         {grad_old[0].get(), weight_decay[0].get()},
         {{gradient.pt_t.sizes(), scalar_dtype}}});
  } else {
    // g_t = g_t
    grad = OpBackend::BuildNode(
        op,
        graph,
        {"identity",
         {grad_old[0].get()},
         {{gradient.pt_t.sizes(), scalar_dtype}}});
  }

  //------calculate first moment----------//
  // m_t = beta1 * m_t_1 + (1-beta1) * g_t
  auto exp_avg_mul_beta1 = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("mult_fwd"sv, scalar_dtype),
       {exp_avg.syn_t, beta1_t},
       exp_avg_attr});

  auto grad_scaled = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("mult_fwd"sv, scalar_dtype),
       {grad[0].get(), one_minus_beta1_t},
       gradient_attr});

  auto exp_avg_1 = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("add_fwd"sv, scalar_dtype),
       {exp_avg_mul_beta1[0].get(), grad_scaled[0].get()},
       {{exp_avg.pt_t.sizes(), scalar_dtype, i + 2 * vec_size}}});

  //------calculate second moment----------//
  // v_t = beta2 * v_t_1 + (1-beta2) * g_t * g_t
  auto grad_sq = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("mult_fwd"sv, scalar_dtype),
       {grad[0].get(), grad[0].get()},
       gradient_attr});

  auto grad_sq_scaled = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("mult_fwd"sv, scalar_dtype),
       {grad_sq[0].get(), one_minus_beta2_t},
       gradient_attr});

  auto exp_avg_sq_mul_beta2 = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("mult_fwd"sv, scalar_dtype),
       {exp_avg_sq.syn_t, beta2_t},
       exp_avg_sq_attr});

  auto exp_avg_sq_1 = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("add_fwd"sv, scalar_dtype),
       {exp_avg_sq_mul_beta2[0].get(), grad_sq_scaled[0].get()},
       {{gradient.pt_t.sizes(), scalar_dtype, i + 3 * vec_size}}});

  //------calculate m^------------------------//
  // m^ = m_t/(1-beta_1^t) // need to handle power of t properly
  auto beta1_n = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("pow_fwd"sv, scalar_dtype),
       {beta1_t, state_step.syn_t},
       scalar_1d_attr});

  auto one_minus_beta1_n = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("sub_fwd"sv, scalar_dtype),
       {one_t, beta1_n[0].get()},
       scalar_1d_attr});

  auto exp_avg_cap = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("div_fwd"sv, scalar_dtype),
       {exp_avg_1[0].get(), one_minus_beta1_n[0].get()},
       gradient_attr});

  //------calculate amsgrad part---------------//
  auto beta2_n = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("pow_fwd"sv, scalar_dtype),
       {beta2_t, state_step.syn_t},
       scalar_1d_attr});

  auto one_minus_beta2_n = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("sub_fwd"sv, scalar_dtype),
       {one_t, beta2_n[0].get()},
       scalar_1d_attr});

  std::vector<synapse_helpers::tensor> exp_avg_sq_cap, exp_avg_sq_max;
  if (amsgrad) {
    //------calculate v^max-------------------//
    // v_t^max = max(v_t^, v_t)
    exp_avg_sq_max = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("max_fwd"sv, scalar_dtype),
         {exp_avg_sq_1[0].get(), max_exp_avg_sq.syn_t},
         {{exp_avg_sq.pt_t.sizes(), scalar_dtype, (i + 4 * vec_size)}}});
    //------calculate v^-------------------//
    // v_t^ = v_t^max/(1-beta_2^t) //
    exp_avg_sq_cap = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("div_fwd"sv, scalar_dtype),
         {exp_avg_sq_max[0].get(), one_minus_beta2_n[0].get()},
         exp_avg_sq_attr});
  } else {
    //------calculate v^-------------------//
    // v_t^ = v_t/(1-beta_2^t)
    exp_avg_sq_cap = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("div_fwd"sv, scalar_dtype),
         {exp_avg_sq_1[0].get(), one_minus_beta2_n[0].get()},
         {{gradient.pt_t.sizes(), scalar_dtype}}});
  }
  //------calculate final weight------------//
  // theta_t = theta_t_1 - gamma * m_t^/(sqrt(v_t^)+epsilon)
  auto exp_avg_sq_cap_sqrt = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("sqrt_fwd"sv, scalar_dtype),
       {exp_avg_sq_cap[0].get()},
       exp_avg_sq_attr});

  auto denom = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("add_fwd"sv, scalar_dtype),
       {exp_avg_sq_cap_sqrt[0].get(), epsilon_t},
       exp_avg_sq_attr});

  auto lr_exp_avg_cap = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("mult_fwd"sv, scalar_dtype),
       {exp_avg_cap[0].get(), lr_t},
       exp_avg_sq_attr});

  auto ratio = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("div_fwd"sv, scalar_dtype),
       {lr_exp_avg_cap[0].get(), denom[0].get()},
       gradient_attr});

  auto result = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("sub_fwd"sv, scalar_dtype),
       {weight.syn_t, ratio[0].get()},
       {NodeAttr::NodeOutputAttr{weight.pt_t.sizes(), scalar_dtype, i}}});
  // store the outputs
  output.push_back(std::move(result[0]));
  if (maximize) {
    auto grad_updated = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("neg_fwd"sv, scalar_dtype),
         {grad_old[0].get()},
         {{gradient.pt_t.sizes(), scalar_dtype, i + vec_size}}});
    // when maximize is true it expects original grad
    output.push_back(std::move(grad_updated[0]));
  } else {
    output.push_back(std::move(grad_old[0]));
  }
  output.push_back(std::move(exp_avg_1[0]));
  output.push_back(std::move(exp_avg_sq_1[0]));
  if (!empty_max_exp_avg_sqs) {
    output.push_back(std::move(exp_avg_sq_max[0]));
  }
  return output;
}

void FusedAdam::AddNode(sh::graph& graph, const at::Stack& stack) {
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
  bool empty_max_exp_avg_sqs = max_exp_avg_sqs.empty();
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
  const auto scalar_dtype = self.front().pt_t.scalar_type();

  int64_t scalar_shape[] = {1};
  double constant_values[] = {
      1.0, beta1, beta2, 1.0 - beta1, 1.0 - beta2, lr, weight_decay, eps};
  const auto& one_t = OpBackend::ConstantHelper(
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
  const auto& wt_decay_t = ConstantHelper(
      graph,
      static_cast<float>(constant_values[6]),
      scalar_dtype,
      scalar_shape);
  const auto& epsilon_t = ConstantHelper(
      graph,
      static_cast<float>(constant_values[7]),
      scalar_dtype,
      scalar_shape);

  size_t vec_size = grads.size();
  for (size_t i = 0; i < vec_size; ++i) {
    const auto& gradient = grads[i];
    const auto& weight = self[i];
    const auto& exp_avg = exp_avgs[i];
    const auto& exp_avg_sq = exp_avg_sqs[i];
    const auto& max_exp_avg_sq = max_exp_avg_sqs.empty()
        ? exp_avg_sq
        : max_exp_avg_sqs[i]; // adding a dummy variable chack with carefully
    const auto& state_step = state_steps[i];
    auto result = FusedAdamCommon(
        this,
        graph,
        weight,
        gradient,
        exp_avg,
        exp_avg_sq,
        max_exp_avg_sq,
        state_step,
        lr_t.get(),
        beta1_t.get(),
        beta2_t.get(),
        wt_decay_t.get(),
        weight_decay,
        epsilon_t.get(),
        one_t.get(),
        one_minus_beta1_t.get(),
        one_minus_beta2_t.get(),
        amsgrad,
        maximize,
        grad_scale,
        found_inf,
        scalar_dtype,
        vec_size,
        i,
        empty_max_exp_avg_sqs);
    syn_out(i) = std::move(result[0]);
    syn_out(i + vec_size) = std::move(result[1]);
    syn_out(i + 2 * vec_size) = std::move(result[2]);
    syn_out(i + 3 * vec_size) = std::move(result[3]);
    if (!empty_max_exp_avg_sqs) {
      syn_out(i + 4 * vec_size) = std::move(result[4]);
    }
  }
}

SharedMetaDataVector AdamLrSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& self_vec = stack.at(0).toTensorVector();
  const auto& grads_vec = stack.at(1).toTensorVector();
  const auto& exp_avgs_vec = stack.at(2).toTensorVector();
  const auto& exp_avg_sqs_vec = stack.at(3).toTensorVector();
  const auto& max_exp_avg_sqs_vec = stack.at(4).toTensorVector();
  const auto& state_steps_vec = stack.at(5).toTensorVector();
  [[maybe_unused]] const auto lr = stack.at(6).toTensor();
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

  std::optional<at::Tensor> lr_tensor = std::optional<at::Tensor>(lr);
  SharedMetaDataVector shared_meta_vec = AdamSharedMetaCommon(
      self_vec,
      grads_vec,
      exp_avgs_vec,
      exp_avg_sqs_vec,
      max_exp_avg_sqs_vec,
      state_steps_vec,
      lr_tensor,
      weight_decay);
  return shared_meta_vec;
}

void FusedAdamLr::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "FusedAdamW::AddNode");
  auto self = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto grads = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto exp_avgs = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto exp_avg_sqs = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto max_exp_avg_sqs = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto state_steps = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto lr = stackGetter.getNextInput<TensorsPair>();
  auto beta1 = stackGetter.getNextInput<double>();
  auto beta2 = stackGetter.getNextInput<double>();
  auto weight_decay = stackGetter.getNextInput<double>();
  auto eps = stackGetter.getNextInput<double>();
  auto amsgrad = stackGetter.getNextInput<bool>();
  auto maximize = stackGetter.getNextInput<bool>();
  auto grad_scale = stackGetter.getNextInput<std::optional<TensorsPair>>();
  auto found_inf = stackGetter.getNextInput<std::optional<TensorsPair>>();
  bool empty_max_exp_avg_sqs = max_exp_avg_sqs.empty();
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

  int64_t scalar_shape[] = {1};

  double constant_values[] = {
      1.0, beta1, beta2, 1.0 - beta1, 1.0 - beta2, weight_decay, eps};
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
  const auto& wt_decay_t = ConstantHelper(
      graph,
      static_cast<float>(constant_values[5]),
      scalar_dtype,
      scalar_shape);
  const auto& epsilon_t = ConstantHelper(
      graph,
      static_cast<float>(constant_values[6]),
      scalar_dtype,
      scalar_shape);

  size_t vec_size = grads.size();
  for (size_t i = 0; i < vec_size; ++i) {
    const auto& gradient = grads[i];
    const auto& weight = self[i];
    const auto& exp_avg = exp_avgs[i];
    const auto& exp_avg_sq = exp_avg_sqs[i];
    const auto& max_exp_avg_sq = max_exp_avg_sqs.empty()
        ? exp_avg_sq
        : max_exp_avg_sqs[i]; // adding a dummy variable chack with carefully
    const auto& state_step = state_steps[i];

    auto result = FusedAdamCommon(
        this,
        graph,
        weight,
        gradient,
        exp_avg,
        exp_avg_sq,
        max_exp_avg_sq,
        state_step,
        lr.syn_t,
        beta1_t.get(),
        beta2_t.get(),
        wt_decay_t.get(),
        weight_decay,
        epsilon_t.get(),
        one_t.get(),
        one_minus_beta1_t.get(),
        one_minus_beta2_t.get(),
        amsgrad,
        maximize,
        grad_scale,
        found_inf,
        scalar_dtype,
        vec_size,
        i,
        empty_max_exp_avg_sqs);

    syn_out(i) = std::move(result[0]);
    syn_out(i + vec_size) = std::move(result[1]);
    syn_out(i + 2 * vec_size) = std::move(result[2]);
    syn_out(i + 3 * vec_size) = std::move(result[3]);
    if (!empty_max_exp_avg_sqs) {
      syn_out(i + 4 * vec_size) = std::move(result[4]);
    }
  }
}

} // namespace habana
