/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

OutputMetaDataVector ComputeLambOutputMetadata(const at::Stack& stack) {
  OutputMetaData meta;
  auto tensors = stack[0].toTensorList();
  const at::Tensor& tensor = tensors[0];
  meta.shape = {1};
  meta.dtype = tensor.scalar_type();
  return {meta};
}

OptimizerFusedLambNorm::OptimizerFusedLambNorm(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "optimizer_lamb_fused_norm_fwd_",
          scalar_type,
          {},
          {},
          {},
          false) {
  SetOutputMetaFn(ComputeLambOutputMetadata);
}

void OptimizerFusedLambNorm::CustomHandler(
    synapse_helpers::graph& graph,
    at::Stack& stack) {
  auto metadata = GetOutputMetaData(0);
  const auto& t = at::detail::make_tensor<c10::TensorImpl>(
      c10::DispatchKeySet{at::DispatchKey::HPU, at::DispatchKey::AutogradHPU},
      c10::scalarTypeToTypeMeta(metadata.dtype),
      c10::Device(c10::kHPU, 0));
  t.unsafeGetTensorImpl()->set_sizes_contiguous(metadata.shape);

  const auto& output = habana::createPTTensor(
      t,
      metadata.shape,
      t.options().dtype(metadata.dtype),
      metadata.persistent);
  AllocateSynapseOutput(graph, output, metadata);
}

void OptimizerFusedLambNorm::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(
      stack.size() == 2, "OptimizerFusedLambNorm must have 2 input arguments");

  StackGetter stackGetter(stack, "OptimizerFusedLambNorm::AddNode");
  auto gradients = getNextInput<std::vector<TensorsPair>>(stackGetter);
  float max_grad_norm = static_cast<float>(getNextInput<double>(stackGetter));

  TORCH_CHECK(
      gradients.size() > 0,
      "Gradiens list in OptimizerFusedLambNorm cannot be empty");

  auto dtype = gradients[0].pt_t.scalar_type();

  auto syn_max_grad_norm = ConstantHelper(graph, max_grad_norm, dtype, {1});

  auto num_params = static_cast<int>(gradients.size());
  std::vector<synapse_helpers::tensor> first_reshape;
  std::vector<synapse_helpers::tensor> intermediate_reduce;
  std::vector<synTensor> concat_inputs;
  for (size_t i = 0; i < num_params; ++i) {
    auto rank = gradients[i].pt_t.dim();
    auto shape = gradients[i].pt_t.sizes().vec();

    int reshape_shape = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      reshape_shape *= shape[i];
    }
    first_reshape.push_back((OpBackend::BuildReshape(
        this, graph, gradients[i].syn_t, {reshape_shape}, dtype)));

    ns_Reduction::Params params{};
    params.reductionDimension = 0;
    intermediate_reduce.emplace_back(std::move(OpBackend::BuildNode(
        this,
        graph,
        {get_guid_with_precision("reduce_sum_square_fwd", dtype),
         {first_reshape.back().get()},
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
      {{{num_params}, dtype}},
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

OptimizerLambFusedPhase2::OptimizerLambFusedPhase2(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "optimizer_lamb_fused_phase2",
          scalar_type,
          {},
          {0},
          {},
          false) {}

void OptimizerLambFusedPhase2::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(
      stack.size() == 7,
      "OptimizerLambFusedPhase2 must have 7 input arguments");

  StackGetter stackGetter(stack, "OptimizerLambFusedPhase2::AddNode");
  auto weights = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto adam_norms = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto weight_norms = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto adam_steps = getNextInput<std::vector<TensorsPair>>(stackGetter);
  float step = static_cast<float>(getNextInput<double>(stackGetter));
  auto weight_decay = static_cast<float>(getNextInput<double>(stackGetter));
  bool use_lamb = getNextInput<bool>(stackGetter);

  auto syn_negative_step = ConstantHelper(graph, -step, torch::kFloat, {1});
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

      auto weight_mask = OpBackend::BuildNode(
          this,
          graph,
          {get_guid_with_precision("greater_fwd", dtype),
           {weight_norms[i].syn_t, zero->get()},
           {{{1}, at::kBool}}});

      auto adam_mask = OpBackend::BuildNode(
          this,
          graph,
          {get_guid_with_precision("greater_fwd", dtype),
           {adam_norms[i].syn_t, zero->get()},
           {{{1}, at::kBool}}});

      auto mask = OpBackend::BuildNode(
          this,
          graph,
          {get_guid_with_precision("and_fwd", at::kBool),
           {weight_mask[0].get(), adam_mask[0].get()},
           {{{1}, at::kBool}}});

      auto where = OpBackend::BuildNode(
          this,
          graph,
          {get_guid_with_precision("where_fwd", dtype),
           {mask[0].get(), div[0].get(), one->get()},
           {{{1}, dtype}}});

      trust_ratio = std::move(where[0]);
    }

    auto mul = OpBackend::BuildNode(
        this,
        graph,
        {get_guid_with_precision("mult_fwd", dtype),
         {adam_steps[i].syn_t, syn_negative_step.get()},
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
            KERNEL_FN_GLOBAL(habana::OptimizerFusedLambNorm))
        .add(
            "hpu::optimizer_lamb_fused_phase2",
            KERNEL_FN_GLOBAL(habana::OptimizerLambFusedPhase2));
