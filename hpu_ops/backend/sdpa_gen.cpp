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

#include "hpu_ops/sdpa_gen.h"

namespace habana {

SDPAFwd::SDPAFwd(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "sdpa_fwd", scalar_type, {0, 0, 0}, {}, {}, false) {}

SDPABwd::SDPABwd(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "sdpa_bwd", scalar_type, {0, 0, 0}, {}, {}, false) {}

sizes_vec SDPAFwdOutputShape(const at::Stack& stack) {
  auto q = stack_tensor(stack, 0);
  auto k = stack_tensor(stack, 1);
  auto v = stack_tensor(stack, 2);

  int64_t rank = q.dim();
  std::vector<int64_t> q_shape = q.sizes().vec();
  std::vector<int64_t> k_shape = k.sizes().vec();
  std::vector<int64_t> v_shape = v.sizes().vec();
  std::vector<int64_t> qkt_shape{q_shape.begin(), q_shape.begin() + rank - 2};
  std::vector<int64_t> out_shape{q_shape.begin(), q_shape.begin() + rank - 1};
  int L_dim = rank - 2; // Target seq len dim
  int S_dim = rank - 2; // Source seq len dim
  int Ev_dim = rank - 1; // head_dim_v dim
  out_shape.push_back(v_shape[Ev_dim]);
  qkt_shape.push_back(q_shape[L_dim]);
  qkt_shape.push_back(k_shape[S_dim]);

  return {out_shape, qkt_shape, qkt_shape};
}

sizes_vec SDPABwdOutputShape(const at::Stack& stack) {
  auto q = stack_tensor(stack, 1);
  auto k = stack_tensor(stack, 2);
  auto v = stack_tensor(stack, 3);

  std::vector<int64_t> q_shape = q.sizes().vec();
  std::vector<int64_t> k_shape = k.sizes().vec();
  std::vector<int64_t> v_shape = v.sizes().vec();

  return {q_shape, k_shape, v_shape};
}

void SDPAFwd::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(stack, "SDPAFwd::AddNode");
  auto q = getNextInput<TensorsPair>(stackGetter);
  auto k = getNextInput<TensorsPair>(stackGetter);
  auto v = getNextInput<TensorsPair>(stackGetter);
  auto attention_mask = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto seed = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto p = getNextInput<double>(stackGetter);
  auto scale = getNextInput<double>(stackGetter);
  auto is_causal = getNextInput<bool>(stackGetter);

  ns_Sdpa::Params params{};
  params.scale = scale;
  params.dropout.ratio = p;
  if (is_causal) {
    params.is_causal = true;
  }
  std::string guid = get_guid_with_precision("sdpa_fwd", q.pt_t.scalar_type());
  auto out_shapes = SDPAFwdOutputShape(stack);

  std::vector<synTensor> syn_inputs = {q.syn_t, k.syn_t, v.syn_t};
  if (attention_mask) {
    syn_inputs.push_back(attention_mask.value().syn_t);
  } else {
    syn_inputs.push_back(nullptr);
  }
  if (p > 0.0) {
    syn_inputs.push_back(seed.value().syn_t);
  } else {
    syn_inputs.push_back(nullptr);
  }

  std::vector<NodeAttr::NodeOutputAttr> output_attrs = {
      {out_shapes[0], q.pt_t.scalar_type(), 0},
      {out_shapes[1], q.pt_t.scalar_type(), 1}};
  if (p > 0.0) {
    output_attrs.push_back({out_shapes[2], at::ScalarType::Char, 2});
  }

  auto output = OpBackend::BuildNode(
      this, graph, {guid, syn_inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(output[0]);
  syn_out(1) = std::move(output[1]);
  if (p > 0.0) {
    syn_out(2) = std::move(output[2]);
  }
}

void SDPABwd::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(stack, "SDPABwd::AddNode");
  auto grad = getNextInput<TensorsPair>(stackGetter);
  auto q = getNextInput<TensorsPair>(stackGetter);
  auto k = getNextInput<TensorsPair>(stackGetter);
  auto v = getNextInput<TensorsPair>(stackGetter);
  auto P = getNextInput<TensorsPair>(stackGetter);
  auto dm = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto p = getNextInput<double>(stackGetter);
  auto scale = getNextInput<double>(stackGetter);

  ns_Sdpa::Params params{};
  params.scale = scale;
  params.dropout.ratio = p;

  std::string guid = get_guid_with_precision("sdpa_bwd", q.pt_t.scalar_type());
  auto out_shapes = SDPABwdOutputShape(stack);

  std::vector<synTensor> syn_inputs = {
      grad.syn_t, q.syn_t, k.syn_t, v.syn_t, P.syn_t};
  if (p > 0.0) {
    syn_inputs.push_back(dm.value().syn_t);
  } else {
    syn_inputs.push_back(nullptr);
  }

  std::vector<NodeAttr::NodeOutputAttr> output_attrs = {
      {out_shapes[0], q.pt_t.scalar_type(), 0},
      {out_shapes[1], q.pt_t.scalar_type(), 1},
      {out_shapes[2], q.pt_t.scalar_type(), 2}};

  auto output = OpBackend::BuildNode(
      this, graph, {guid, syn_inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(output[0]);
  syn_out(1) = std::move(output[1]);
  syn_out(2) = std::move(output[2]);
}

} // namespace habana

static const auto& SDPAKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::sdpa_fwd", KERNEL_FN_GLOBAL(habana::SDPAFwd))
        .add("hpu::sdpa_bwd", KERNEL_FN_GLOBAL(habana::SDPABwd));
