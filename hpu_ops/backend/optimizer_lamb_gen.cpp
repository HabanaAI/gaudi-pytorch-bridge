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

#include "hpu_ops/optimizer_lamb_gen.h"
#include "backend/create_pt_tensor.h"

namespace habana {

// OptimizerLambNorm

OutputMetaDataVector ComputeLambOutputMetadata(const at::Stack& stack) {
  OutputMetaData meta;
  auto tensors = stack[0].toTensorList();
  const at::Tensor& tensor = tensors[0];
  meta.shape = {1};
  meta.dtype = tensor.scalar_type();
  return {meta};
}

OptimizerLambNorm::OptimizerLambNorm(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "optimizer_lamb_norm_fwd_",
          scalar_type,
          {0},
          {},
          {},
          false) {
  SetOutputMetaFn(ComputeLambOutputMetadata);
}

void OptimizerLambNorm::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(
      stack.size() == 2, "OptimizerLambNorm must have 2 input arguments");

  StackGetter stackGetter(stack, "OptimizerLambNorm::AddNode");
  auto gradients = getNextInput<std::vector<TensorsPair>>(stackGetter);
  float max_grad_norm = static_cast<float>(getNextInput<double>(stackGetter));

  TORCH_CHECK(
      gradients.size() > 0,
      "Gradiens list in OptimizerLambNorm cannot be empty");

  auto dtype = gradients[0].pt_t.scalar_type();

  auto syn_max_grad_norm = ConstantHelper(graph, max_grad_norm, dtype, {1});

  auto num_params = gradients.size();
  std::vector<synapse_helpers::tensor> intermediate_reduce;
  std::vector<synTensor> concat_inputs;
  for (size_t i = 0; i < num_params; ++i) {
    ns_Reduction::ParamsV2 params{};
    params.reductionDimensionMask = 0;
    params.keepDim = false;
    intermediate_reduce.emplace_back(std::move(OpBackend::BuildNode(
        this,
        graph,
        {get_guid_with_precision("reduce_sum_square_multi_dim_fwd", dtype),
         {gradients[i].syn_t},
         {{{1}, dtype}},
         &params,
         sizeof(params)})[0]));

    concat_inputs.emplace_back(intermediate_reduce.back().get());
  }

  synConcatenateParams concat_params{};
  concat_params.axis = 0;

  auto concated = OpBackend::BuildOp(
      graph,
      "concat",
      std::move(concat_inputs),
      {{{static_cast<long>(num_params)}, dtype}},
      &concat_params,
      sizeof(concat_params));

  ns_Reduction::Params reduce_params{};
  reduce_params.reductionDimension = 0;

  auto sum_final = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("reduce_sum_fwd", dtype),
       {concated[0].get()},
       {{{1}, dtype}},
       &reduce_params,
       sizeof(reduce_params)});

  auto sqrt = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("sqrt_fwd", dtype),
       {sum_final[0].get()},
       {{{1}, dtype}}});

  auto div = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("div_fwd", dtype),
       {sqrt[0].get(), syn_max_grad_norm.get()},
       {{{1}, dtype}}});

  auto less = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("less_equal_fwd", dtype),
       {sqrt[0].get(), syn_max_grad_norm.get()},
       {{{1}, at::kBool}}});

  auto less_cast =
      OpBackend::BuildCast(this, graph, less[0].get(), {1}, at::kBool, dtype);

  auto syn_clip_norm = ConstantHelper(graph, 1.0, dtype, {1});
  auto mul1 = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("mult_fwd", dtype),
       {less_cast.get(), syn_clip_norm.get()},
       {{{1}, dtype}}});

  auto eq_final = OpBackend::BuildNode(
      this, graph, {"not_fwd_i8", {less[0].get()}, {{{1}, at::kBool}}});

  auto eq_casted = OpBackend::BuildCast(
      this, graph, eq_final[0].get(), {1}, at::kBool, dtype);

  auto mul2 = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("mult_fwd", dtype),
       {eq_casted.get(), div[0].get()},
       {{{1}, dtype}}});

  auto add = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("add_fwd", dtype),
       {mul1[0].get(), mul2[0].get()},
       {{{1}, dtype, 0}}});

  syn_out(0) = std::move(add[0]);
}

// OptimizerLambPhase1

OptimizerLambPhase1::OptimizerLambPhase1(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "optimizer_lamb_phase1",
          scalar_type,
          {},
          {2, 3, 4, 5, 6},
          {},
          false) {}

static std::vector<synapse_helpers::tensor> ComputeNorm(
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor input_syn_tensor,
    const at::IntArrayRef input_shape,
    c10::ScalarType dtype,
    c10::optional<int> final_idx = c10::nullopt) {
  if (input_shape.size() <= 1 || input_shape[0] == 1) {
    auto norm_mul = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("mult_fwd", dtype),
         {input_syn_tensor, input_syn_tensor},
         {{input_shape, dtype}}});

    ns_Reduction::ParamsV2 reduce_params{};
    reduce_params.reductionDimensionMask = 0;
    reduce_params.keepDim = false;
    auto sum = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("reduce_sum_multi_dim_fwd", dtype),
         {norm_mul[0].get()},
         {{{1}, dtype}},
         &reduce_params,
         sizeof(reduce_params)});

    return OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("sqrt_fwd", dtype),
         {sum[0].get()},
         {{{1}, dtype, final_idx}}});
  } else {
    return OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("frobenius_norm_fwd", dtype),
         {input_syn_tensor},
         {{{1}, dtype, final_idx}}});
  }
}

void OptimizerLambPhase1::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(
      stack.size() == 15, "OptimizerLambPhase1 must have 15 input arguments");

  StackGetter stackGetter(stack, "OptimizerLambPhase1::AddNode");
  auto gradients = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto weights = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto exp_avg = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto exp_avg_sq = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto out_wt_norm = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto out_adam_norm = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto out_adam_step = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto clip_global_grad_norm = getNextInput<TensorsPair>(stackGetter);
  auto grad_averaging = getNextInput<int>(stackGetter);
  auto beta1 = getNextInput<double>(stackGetter);
  auto beta2 = getNextInput<double>(stackGetter);
  auto epsilon = getNextInput<double>(stackGetter);
  auto bias_correction1 = getNextInput<TensorsPair>(stackGetter);
  auto bias_correction2 = getNextInput<TensorsPair>(stackGetter);
  auto weight_decay = getNextInput<double>(stackGetter);

  float beta3 = 1.0;
  if (grad_averaging) {
    beta3 = 1 - beta1;
  }

  auto beta1_t = ConstantHelper(graph, beta1, at::kFloat, {1});
  auto beta2_t = ConstantHelper(graph, beta2, at::kFloat, {1});
  auto beta3_t = ConstantHelper(graph, beta3, at::kFloat, {1});
  auto one_minus_beta2_t = ConstantHelper(graph, 1.0 - beta2, at::kFloat, {1});
  auto epsilon_t = ConstantHelper(graph, epsilon, at::kFloat, {1});
  auto weight_decay_t = ConstantHelper(graph, weight_decay, at::kFloat, {1});

  auto dtype = gradients[0].pt_t.scalar_type();
  auto num_params = static_cast<int>(weights.size());

  for (auto i = 0; i < num_params; i++) {
    auto div_grad_shape = gradients[i].pt_t.sizes().vec();
    auto div_grad = OpBackend::BuildNode(
        this,
        graph,
        {get_guid_with_precision("div_fwd", dtype),
         {gradients[i].syn_t, clip_global_grad_norm.syn_t},
         {{div_grad_shape, dtype}}});

    auto mul_exp_avg_shape = exp_avg[i].pt_t.sizes().vec();
    auto mul_exp_avg = OpBackend::BuildNode(
        this,
        graph,
        {get_guid_with_precision("mult_fwd", dtype),
         {exp_avg[i].syn_t, beta1_t.get()},
         {{mul_exp_avg_shape, dtype}}});

    auto add_exp_beta_avg = OpBackend::BuildNode(
        this,
        graph,
        {get_guid_with_precision("mult_fwd", dtype),
         {div_grad[0].get(), beta3_t.get()},
         {{div_grad_shape, dtype}}});

    auto add_exp_avg = OpBackend::BuildNode(
        this,
        graph,
        {get_guid_with_precision("add_fwd", dtype),
         {mul_exp_avg[0].get(), add_exp_beta_avg[0].get()},
         {{mul_exp_avg_shape, dtype}}});

    auto add_exp_avg_out = OpBackend::IdentityHelper(
        graph, add_exp_avg[0].get(), mul_exp_avg_shape, dtype, i);

    syn_out(i) = std::move(add_exp_avg_out);

    auto mul_exp_avg_sq_shape = exp_avg_sq[i].pt_t.sizes().vec();
    auto mul_exp_avg_sq = OpBackend::BuildNode(
        this,
        graph,
        {get_guid_with_precision("mult_fwd", dtype),
         {exp_avg_sq[i].syn_t, beta2_t.get()},
         {{mul_exp_avg_sq_shape, dtype}}});

    auto addcmul_a = OpBackend::BuildNode(
        this,
        graph,
        {get_guid_with_precision("mult_fwd", dtype),
         {div_grad[0].get(), div_grad[0].get()},
         {{mul_exp_avg_sq_shape, dtype}}});

    auto addcmul_b = OpBackend::BuildNode(
        this,
        graph,
        {get_guid_with_precision("mult_fwd", dtype),
         {addcmul_a[0].get(), one_minus_beta2_t.get()},
         {{mul_exp_avg_sq_shape, dtype}}});

    auto addcmul_exp_avg_sq = OpBackend::BuildNode(
        this,
        graph,
        {get_guid_with_precision("add_fwd", dtype),
         {mul_exp_avg_sq[0].get(), addcmul_b[0].get()},
         {{mul_exp_avg_sq_shape, dtype}}});

    auto addcmul_exp_avg_sq_out = OpBackend::IdentityHelper(
        graph,
        addcmul_exp_avg_sq[0].get(),
        mul_exp_avg_sq_shape,
        dtype,
        i + num_params);

    syn_out(i + num_params) = std::move(addcmul_exp_avg_sq_out);

    auto div_exp_avg = OpBackend::BuildNode(
        this,
        graph,
        {get_guid_with_precision("div_fwd", dtype),
         {add_exp_avg[0].get(), bias_correction1.syn_t},
         {{mul_exp_avg_shape, dtype}}});

    auto div_exp_avg_sq = OpBackend::BuildNode(
        this,
        graph,
        {get_guid_with_precision("div_fwd", dtype),
         {addcmul_exp_avg_sq[0].get(), bias_correction2.syn_t},
         {{mul_exp_avg_sq_shape, dtype}}});

    auto sqrt_exp_avg_sq = OpBackend::BuildNode(
        this,
        graph,
        {get_guid_with_precision("sqrt_fwd", dtype),
         {div_exp_avg_sq[0].get()},
         {{mul_exp_avg_sq_shape, dtype}}});

    auto add_exp_avg_sq = OpBackend::BuildNode(
        this,
        graph,
        {get_guid_with_precision("add_fwd", dtype),
         {sqrt_exp_avg_sq[0].get(), epsilon_t.get()},
         {{mul_exp_avg_sq_shape, dtype}}});

    const bool is_weight_decay = weight_decay != 0.0;

    c10::optional<int> div_wt_result_index = is_weight_decay
        ? c10::nullopt
        : c10::make_optional<int>(i + 4 * num_params);

    std::vector<synapse_helpers::tensor> norm_input;
    auto div_wt = OpBackend::BuildNode(
        this,
        graph,
        {get_guid_with_precision("div_fwd", dtype),
         {div_exp_avg[0].get(), add_exp_avg_sq[0].get()},
         {{mul_exp_avg_sq_shape, dtype, div_wt_result_index}}});
    norm_input.emplace_back(std::move(div_wt[0]));

    if (is_weight_decay) {
      auto add_wt_beta = OpBackend::BuildNode(
          this,
          graph,
          {get_guid_with_precision("mult_fwd", dtype),
           {weights[i].syn_t, weight_decay_t.get()},
           {{div_grad_shape, dtype}}});

      auto add_wt = OpBackend::BuildNode(
          this,
          graph,
          {get_guid_with_precision("add_fwd", dtype),
           {norm_input.back().get(), add_wt_beta[0].get()},
           {{mul_exp_avg_shape, dtype, i + 4 * num_params}}});
      norm_input.emplace_back(std::move(add_wt[0]));
    }

    auto norm_wt = ComputeNorm(
        this,
        graph,
        weights[i].syn_t,
        mul_exp_avg_shape,
        dtype,
        i + 2 * num_params);
    syn_out(i + 2 * num_params) = std::move(norm_wt[0]);

    auto norm_adam_step = ComputeNorm(
        this,
        graph,
        norm_input.back().get(),
        mul_exp_avg_shape,
        dtype,
        i + 3 * num_params);
    syn_out(i + 3 * num_params) = std::move(norm_adam_step[0]);
    syn_out(i + 4 * num_params) = std::move(norm_input.back());
  }
}

// OptimizerLambPhase2

OptimizerLambPhase2::OptimizerLambPhase2(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "optimizer_lamb_phase2",
          scalar_type,
          {},
          {0},
          {},
          false) {}

void OptimizerLambPhase2::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(
      stack.size() == 7, "OptimizerLambPhase2 must have 7 input arguments");

  StackGetter stackGetter(stack, "OptimizerLambPhase2::AddNode");
  auto weights = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto adam_norms = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto weight_norms = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto adam_steps = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto neg_step = getNextInput<TensorsPair>(stackGetter);
  auto weight_decay = static_cast<float>(getNextInput<double>(stackGetter));
  bool use_lamb = getNextInput<bool>(stackGetter);

  auto dtype = weights[0].pt_t.scalar_type();

  std::optional<synapse_helpers::tensor> zero;
  std::optional<synapse_helpers::tensor> one;
  bool calc_trust_ratio = weight_decay != 0 || use_lamb;
  if (calc_trust_ratio) {
    zero = ConstantHelper(graph, 0, dtype, {1});
    one = ConstantHelper(graph, 1, dtype, {1});
  }

  for (size_t i = 0; i < weights.size(); ++i) {
    std::optional<synapse_helpers::tensor> trust_ratio;
    std::optional<synapse_helpers::tensor> update;

    if (calc_trust_ratio) {
      auto div = OpBackend::BuildNode(
          this,
          graph,
          {get_guid_with_precision("div_fwd", dtype),
           {weight_norms[i].syn_t, adam_norms[i].syn_t},
           {{{1}, dtype}}});

      auto adam_mask = OpBackend::BuildNode(
          this,
          graph,
          {get_guid_with_precision("greater_fwd", dtype),
           {adam_norms[i].syn_t, zero->get()},
           {{{1}, at::kBool}}});

      auto where = OpBackend::BuildNode(
          this,
          graph,
          {get_guid_with_precision("where_fwd", dtype),
           {adam_mask[0].get(), div[0].get(), one->get()},
           {{{1}, dtype}}});

      trust_ratio = std::move(where[0]);
    }

    auto mul = OpBackend::BuildNode(
        this,
        graph,
        {get_guid_with_precision("mult_fwd", dtype),
         {adam_steps[i].syn_t, neg_step.syn_t},
         {{adam_steps[i].pt_t.sizes(), dtype}}});

    if (trust_ratio.has_value()) {
      update = std::move(OpBackend::BuildNode(
          this,
          graph,
          {get_guid_with_precision("mult_fwd", dtype),
           {mul[0].get(), trust_ratio->get()},
           {{adam_steps[i].pt_t.sizes(), dtype}}})[0]);

    } else {
      update = std::move(mul[0]);
    }

    auto updated_weight = OpBackend::BuildNode(
        this,
        graph,
        {get_guid_with_precision("add_fwd", dtype),
         {weights[i].syn_t, update->get()},
         {{weights[i].pt_t.sizes(), dtype, i}}});

    syn_out(i) = std::move(updated_weight[0]);
  }
}
} // namespace habana

static const auto& LambKernelRegistry =
    habana::KernelRegistry()
        .add(
            "hpu::optimizer_lamb_fused_norm",
            KERNEL_FN_GLOBAL(habana::OptimizerLambNorm))
        .add(
            "hpu::optimizer_lamb_phase1",
            KERNEL_FN_GLOBAL(habana::OptimizerLambPhase1))
        .add(
            "hpu::optimizer_lamb_phase2",
            KERNEL_FN_GLOBAL(habana::OptimizerLambPhase2));
