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

#include "hpu_ops/sdpa_gen.h"
#include "hpu_ops/custom_op_outshape.h"

#define SDPA_SET_FLAGS(condition, flags, flag_name) \
  if (condition) {                                  \
    flags |= SdpaFlags_t::SDPA_FLAGS_##flag_name;   \
  }
#define SDPA_ADD_INPUTS(t)                 \
  if (t) {                                 \
    syn_inputs.push_back(t.value().syn_t); \
  } else {                                 \
    syn_inputs.push_back(nullptr);         \
  }

namespace habana {

SDPAFwd::SDPAFwd(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "sdpa_fwd", scalar_type, {0, 0, 0}, {}, {}, false) {}

SDPABwd::SDPABwd(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "sdpa_bwd", scalar_type, {0, 0, 0}, {}, {}, false) {}
Fp8SDPABwd::Fp8SDPABwd(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "fp8_sdpa_bwd",
          scalar_type,
          {0, 0, 0, 0},
          {},
          {},
          false) {}

SDPARecompFwd::SDPARecompFwd(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "sdpa_recomp_fwd",
          scalar_type,
          {0, 0, 0, 0},
          {},
          {},
          false) {}

Fp8SDPARecompFwd::Fp8SDPARecompFwd(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "fp8_sdpa_recomp_fwd",
          scalar_type,
          {0, 0, 0, 0, 0, 0},
          {},
          {},
          false) {}

Fp8SDPAFwd::Fp8SDPAFwd(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "fp8_sdpa_fwd",
          scalar_type,
          {0, 0, 0, 0},
          {},
          {},
          false) {}

SDPARecompBwd::SDPARecompBwd(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "sdpa_recomp_bwd",
          scalar_type,
          {0, 0, 0},
          {},
          {},
          false) {}

static std::vector<int64_t> infer_size_int_or_symint(
    c10::IntArrayRef a,
    c10::IntArrayRef b) {
  return at::infer_size(a, b);
}

static std::vector<c10::SymInt> infer_size_int_or_symint(
    c10::SymIntArrayRef a,
    c10::SymIntArrayRef b) {
  return at::infer_size_symint(a, b);
}

template <class DimT>
sizes_vec_template<DimT> SDPAFwdOutputShapeCommon(
    c10::ArrayRef<DimT> q_shape,
    c10::ArrayRef<DimT> k_shape,
    c10::ArrayRef<DimT> v_shape,
    double dropout_p) {
  int64_t rank = q_shape.size();

  // q, k, v are involved in matmuls (BatchGemm) in attention calc.
  // matmul allows broadcast for the batch dims. i.e, for dims 0, 1.. Rank - 2.
  // So output sizes depend on broadcasting. So get sizes at batch dims for
  // q,k v and compute sizes on these dims after broadcast is done.
  // As per torch.nn.functional.scaled_dot_product_attention() documentation,
  // dim0 can also be excluded from broadcast considerations since the doc
  // mentions that dim0 will be batch size(N) for all tensors.
  // However, in the following shape calculations, dim0 is considered for
  // broad cast shape calc for ease of implementation
  std::vector<DimT> q_bdim_sizes{q_shape.begin(), q_shape.begin() + rank - 2};
  std::vector<DimT> k_bdim_sizes{k_shape.begin(), k_shape.begin() + rank - 2};
  std::vector<DimT> v_bdim_sizes{v_shape.begin(), v_shape.begin() + rank - 2};

  // The following infer sizes also serves to do the shape compatibility
  // checks needed for dynamic shapes causing an exception if the shapes
  // are not compatible for broadcast.
  // Assumption: Only the batch dims are checked for compatibility.
  // The matrix dims are assumed to conform to matrix mul rules.

  // Batch dim sizes of Q@K.transpose after broadcast
  auto qkt_shape = infer_size_int_or_symint(q_bdim_sizes, k_bdim_sizes);
  // Batch dim sizes of output  i.e Q@K.transpose)@v after broadcast
  auto out_shape = infer_size_int_or_symint(qkt_shape, v_bdim_sizes);

  // Append the matrix dims (last 2 dims) to batch dims to get final shape.
  int L_dim = rank - 2; // Target seq len dim
  int S_dim = rank - 2; // Source seq len dim
  int Ev_dim = rank - 1; // head_dim_v dim

  out_shape.push_back(q_shape[L_dim]);
  out_shape.push_back(v_shape[Ev_dim]);

  qkt_shape.push_back(q_shape[L_dim]);
  qkt_shape.push_back(k_shape[S_dim]);

  return {
      out_shape,
      qkt_shape,
      (dropout_p == 0) ? std::vector<DimT>{1} : qkt_shape};
}

sizes_vec SDPAFwdOutputShape(const at::Stack& stack) {
  auto q_or_seed = stack[0].toTensor();
  int q_index = (q_or_seed.sizes().vec().size() < 3) ? 1 : 0;

  auto q = stack_tensor(stack, q_index);
  auto k = stack_tensor(stack, q_index + 1);
  auto v = stack_tensor(stack, q_index + 2);
  int drp_prob_idx = q_index + 4;
  auto drp_prob = stack.at(drp_prob_idx).toDouble();
  return SDPAFwdOutputShapeCommon(q.sizes(), k.sizes(), v.sizes(), drp_prob);
}

/*This is called only in compile*/
sym_sizes_vec fp8_sdpa_fwd_out_shape(
    const std::vector<at::Tensor>& inputs,
    const std::vector<float>& params) {
  TORCH_CHECK(inputs.size() == 3);
  TORCH_CHECK(params.size() == 1);
  sym_sizes_vec out_sizes = SDPAFwdOutputShapeCommon(
      inputs[0].sym_sizes(),
      inputs[1].sym_sizes(),
      inputs[2].sym_sizes(),
      params[0] /*dropout_p*/);
  // insert amax_ds shape
  out_sizes.push_back({1});
  return out_sizes;
}
REGISTER_CUSTOM_OP_OUTSHAPE_FUN(fp8_sdpa_fwd, fp8_sdpa_fwd_out_shape);

/*This is called only in compile*/
sym_sizes_vec sdpa_fwd_out_shape(
    const std::vector<at::Tensor>& inputs,
    const std::vector<float>& params) {
  TORCH_CHECK(inputs.size() == 3);
  TORCH_CHECK(params.size() == 1);

  return SDPAFwdOutputShapeCommon(
      inputs[0].sym_sizes(),
      inputs[1].sym_sizes(),
      inputs[2].sym_sizes(),
      params[0] /*dropout_p*/);
}

REGISTER_CUSTOM_OP_OUTSHAPE_FUN(sdpa_fwd, sdpa_fwd_out_shape);

sizes_vec SDPABwdOutputShape(const at::Stack& stack) {
  // g is grad tensor input into BWD, corr. tensor dO in CGUID
  auto g = stack_tensor(stack, 0);
  auto q = stack_tensor(stack, 1);
  auto k = stack_tensor(stack, 2);
  auto v = stack_tensor(stack, 3);
  auto p = stack_tensor(stack, 4);

  int64_t rank = q.dim();
  std::vector<int64_t> q_shape = q.sizes().vec();
  std::vector<int64_t> k_shape = k.sizes().vec();
  std::vector<int64_t> v_shape = v.sizes().vec();
  std::vector<int64_t> g_shape = g.sizes().vec();
  std::vector<int64_t> p_shape = p.sizes().vec();

  std::vector<int64_t> q_bdim_sizes{
      q_shape.begin(), q_shape.begin() + rank - 2};
  std::vector<int64_t> k_bdim_sizes{
      k_shape.begin(), k_shape.begin() + rank - 2};
  std::vector<int64_t> v_bdim_sizes{
      v_shape.begin(), v_shape.begin() + rank - 2};
  std::vector<int64_t> g_bdim_sizes{
      g_shape.begin(), g_shape.begin() + rank - 2};
  std::vector<int64_t> p_bdim_sizes{
      p_shape.begin(), p_shape.begin() + rank - 2};

  // The following infer sizes are to do the shape compatibility
  // checks needed for dynamic shapes. This will cause an exception
  // if the shapes are not compatible for broadcast.
  // Assumption: Only the batch dims are checked for compatibility.
  // The matrix dims are assumed to conform to matrix mul rules.
  // dp, dv, dq, dk are the derivatives calculated in the SDPA
  // BWD CGUID. So do shape compatibility checks using the tensor
  // shapes involved in the matrtix multiplications needed for
  // these derivatives.
  auto dp_shape = at::infer_size(g_bdim_sizes, v_bdim_sizes);
  auto dv_shape = at::infer_size(p_bdim_sizes, g_bdim_sizes);
  auto dq_shape = at::infer_size(p_bdim_sizes, k_bdim_sizes);
  auto dk_shape = at::infer_size(p_bdim_sizes, q_bdim_sizes);

  return {q_shape, k_shape, v_shape};
}

sizes_vec Fp8SDPABwdOutputShape(const at::Stack& stack) {
  sizes_vec out_shape = SDPABwdOutputShape(stack);
  // insert amax_ds shape
  out_shape.push_back({1});

  return out_shape;
}

static void fillSdpaParams(
    ns_Sdpa::ParamsV3& params,
    double p,
    double scale,
    bool is_causal,
    bool is_inference,
    c10::string_view softmax_mode = "",
    unsigned int flags = 0) {
  SdpaSoftmaxMode_t sfmx_mode = SdpaSoftmaxMode_t::SDPA_DEFAULT_SOFTMAX;
  if (softmax_mode == "fast") {
    sfmx_mode = SdpaSoftmaxMode_t::SDPA_SOFTMAX_HF8_1C;
  }
  params.scale = scale;
  params.dropout.ratio = p;
  params.is_causal = is_causal;
  params.is_inference = is_inference;
  params.softmax_mode = sfmx_mode;
  params.flags = flags;
}

sizes_vec Fp8SDPAFwdOutputShape(const at::Stack& stack) {
  sizes_vec out_shapes = SDPAFwdOutputShape(stack);
  // insert amax_s shape
  out_shapes.push_back({1});
  return out_shapes;
}

void SDPAFwd::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(stack, "SDPAFwd::AddNode");
  auto q_or_seed = stack[0].toTensor();
  bool seed_present = (q_or_seed.sizes().vec().size() < 3);

  synTensor seed_tensor = nullptr;
  if (seed_present) {
    auto seed = getNextInput<TensorsPair>(stackGetter);
    seed_tensor = seed.syn_t;
  }

  auto q = getNextInput<TensorsPair>(stackGetter);
  auto k = getNextInput<TensorsPair>(stackGetter);
  auto v = getNextInput<TensorsPair>(stackGetter);
  auto attention_mask = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto p = getNextInput<double>(stackGetter);
  auto scale = getNextInput<double>(stackGetter);
  auto is_causal = getNextInput<bool>(stackGetter);
  auto softmax_mode = getNextInput<c10::string_view>(stackGetter);
  auto valid_seq_len = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto seq_padding_type = getNextInput<c10::string_view>(stackGetter);
  unsigned int flags = 0;

  SDPA_SET_FLAGS(valid_seq_len, flags, VALID_SEQ_LEN_PRESENT)
  SDPA_SET_FLAGS(seq_padding_type == "left", flags, SEQ_PADDING_LEFT)
  SDPA_SET_FLAGS(seq_padding_type == "right", flags, SEQ_PADDING_RIGHT)

  ns_Sdpa::ParamsV3 params{};
  fillSdpaParams(params, p, scale, is_causal, false, softmax_mode, flags);

  std::string guid = get_guid_with_precision("sdpa_fwd", q.pt_t.scalar_type());
  auto out_shapes = SDPAFwdOutputShape(stack);

  std::vector<synTensor> syn_inputs = {q.syn_t, k.syn_t, v.syn_t};
  if (attention_mask) {
    syn_inputs.push_back(attention_mask.value().syn_t);
  } else {
    syn_inputs.push_back(nullptr);
  }

  syn_inputs.push_back(seed_tensor);

  if (valid_seq_len) {
    syn_inputs.insert(syn_inputs.end(), 6, nullptr);
    syn_inputs.push_back(valid_seq_len.value().syn_t);
  } else {
    syn_inputs.insert(syn_inputs.end(), 7, nullptr);
  }

  std::vector<NodeAttr::NodeOutputAttr> output_attrs = {
      {out_shapes[0], q.pt_t.scalar_type(), 0},
      {out_shapes[1], q.pt_t.scalar_type(), 1}}; // TODO: make optional
  if (p > 0.0) {
    output_attrs.push_back({out_shapes[2], at::ScalarType::Char, 2});
  }

  auto output = OpBackend::BuildNode(
      this, graph, {guid, syn_inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(output[0]);
  syn_out(1) = std::move(output[1]); // TODO: make optional
  if (p > 0.0) {
    syn_out(2) = std::move(output[2]);
  }
}

void Fp8SDPAFwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "Fp8SDPAFwd::AddNode");
  auto q_or_seed = stack[0].toTensor();
  bool seed_present = (q_or_seed.sizes().vec().size() < 3);
  synTensor seed_tensor = nullptr;
  if (seed_present) {
    auto seed = getNextInput<TensorsPair>(stackGetter);
    seed_tensor = seed.syn_t;
  }
  auto q = getNextInput<TensorsPair>(stackGetter);
  auto k = getNextInput<TensorsPair>(stackGetter);
  auto v = getNextInput<TensorsPair>(stackGetter);
  auto attention_mask = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto p = getNextInput<double>(stackGetter);
  auto scale = getNextInput<double>(stackGetter);
  auto is_causal = getNextInput<bool>(stackGetter);
  auto softmax_mode = getNextInput<c10::string_view>(stackGetter);
  auto d_scale_q = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto d_scale_k = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto d_scale_v = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto q_scale_s = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto q_scale_o = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto d_scale_s = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto is_amax_s = getNextInput<bool>(stackGetter);
  auto valid_seq_len = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto seq_padding_type = getNextInput<c10::string_view>(stackGetter);

  ns_Sdpa::ParamsV3 params{};
  unsigned int flags = 0;

  SDPA_SET_FLAGS(is_amax_s, flags, AMAX_S)
  SDPA_SET_FLAGS(d_scale_q, flags, D_SCALE_Q)
  SDPA_SET_FLAGS(d_scale_k, flags, D_SCALE_K)
  SDPA_SET_FLAGS(d_scale_v, flags, D_SCALE_V)
  SDPA_SET_FLAGS(q_scale_s, flags, Q_SCALE_S)
  SDPA_SET_FLAGS(q_scale_o, flags, Q_SCALE_O)
  SDPA_SET_FLAGS(valid_seq_len, flags, VALID_SEQ_LEN_PRESENT)
  SDPA_SET_FLAGS(seq_padding_type == "left", flags, SEQ_PADDING_LEFT)
  SDPA_SET_FLAGS(seq_padding_type == "right", flags, SEQ_PADDING_RIGHT)
  if (d_scale_s) {
    // TODO: add the flag definition to perf_lib_layer_paras.h
    flags |= (1 << 13);
  }

  fillSdpaParams(
      params, p, scale, is_causal, false /*is_inference*/, softmax_mode, flags);

  std::string guid = get_guid_with_precision("sdpa_fwd", q.pt_t.scalar_type());

  std::vector<synTensor> syn_inputs = {q.syn_t, k.syn_t, v.syn_t};
  if (attention_mask) {
    syn_inputs.push_back(attention_mask.value().syn_t);
  } else {
    syn_inputs.push_back(nullptr);
  }

  syn_inputs.push_back(seed_tensor);

  SDPA_ADD_INPUTS(d_scale_q)
  SDPA_ADD_INPUTS(d_scale_k)
  SDPA_ADD_INPUTS(d_scale_v)
  SDPA_ADD_INPUTS(q_scale_s)
  SDPA_ADD_INPUTS(q_scale_o)
  SDPA_ADD_INPUTS(d_scale_s)
  if (valid_seq_len) {
    syn_inputs.push_back(valid_seq_len.value().syn_t);
  }

  auto out_shapes = Fp8SDPAFwdOutputShape(stack);

  auto fwdOutType = q.pt_t.scalar_type();

  // Normally SDPA FWD Fp8 dtype is e4m3. Supporting
  // e5m2 for experiments
  if (q.pt_t.scalar_type() == at::ScalarType::Float8_e4m3fn ||
      q.pt_t.scalar_type() == at::ScalarType::Float8_e5m2) {
    if (q_scale_o.has_value()) {
      fwdOutType = q.pt_t.scalar_type();
    } else {
      fwdOutType = at::ScalarType::BFloat16;
    }
  }

  auto sfmxType = q.pt_t.scalar_type();
  // if (is_amax_s) {
  //  sfmxType = c10::ScalarType::BFloat16;
  // }

  std::vector<NodeAttr::NodeOutputAttr> output_attrs = {
      {out_shapes[0], fwdOutType, 0},
      {out_shapes[1], sfmxType, 1}}; // TODO: make optional
  if (p > 0.0 || is_amax_s) {
    output_attrs.push_back({out_shapes[2], at::ScalarType::Char, 2});
  }
  if (is_amax_s) {
    output_attrs.push_back({out_shapes[3], c10::ScalarType::Float, 3});
  }

  auto output = OpBackend::BuildNode(
      this, graph, {guid, syn_inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(output[0]);
  syn_out(1) = std::move(output[1]);
  if (p > 0.0 || is_amax_s) {
    syn_out(2) = std::move(output[2]);
  }
  if (is_amax_s) {
    syn_out(3) = std::move(output[3]);
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
  auto is_causal = getNextInput<bool>(stackGetter);
  auto p = getNextInput<double>(stackGetter);
  auto scale = getNextInput<double>(stackGetter);
  auto fwd_out = getNextInput<TensorsPair>(stackGetter);

  bool use_fwd_out = GET_ENV_FLAG_NEW(PT_HPU_SDPA_SFMX_BWD_V2);

  ns_Sdpa::ParamsV3 params{};
  fillSdpaParams(params, p, scale, is_causal, false /*is_inference*/);

  std::string guid = get_guid_with_precision("sdpa_bwd", q.pt_t.scalar_type());
  auto out_shapes = SDPABwdOutputShape(stack);

  std::vector<synTensor> syn_inputs = {
      grad.syn_t, q.syn_t, k.syn_t, v.syn_t, P.syn_t};
  if (p > 0.0) {
    syn_inputs.push_back(dm.value().syn_t);
  } else {
    syn_inputs.push_back(nullptr);
  }
  // Same CGUID is used for fp8 and non-fp8. So fill null ptr
  // for all the fp8 scales
  syn_inputs.insert(syn_inputs.end(), 8, nullptr);
  if (use_fwd_out) {
    syn_inputs.push_back(fwd_out.syn_t);
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

void Fp8SDPABwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "Fp8SDPABwd::AddNode");
  auto grad = getNextInput<TensorsPair>(stackGetter);
  auto q = getNextInput<TensorsPair>(stackGetter);
  auto k = getNextInput<TensorsPair>(stackGetter);
  auto v = getNextInput<TensorsPair>(stackGetter);
  auto P = getNextInput<TensorsPair>(stackGetter);
  auto dm = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto is_causal = getNextInput<bool>(stackGetter);
  auto p = getNextInput<double>(stackGetter);
  auto scale = getNextInput<double>(stackGetter);
  auto d_scale_q = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto d_scale_k = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto d_scale_v = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto d_scale_s = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto d_scale_do = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto d_scale_ds = getNextInput<c10::optional<TensorsPair>>(stackGetter);

  auto q_scale_s = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto q_scale_ds = getNextInput<c10::optional<TensorsPair>>(stackGetter);

  auto is_amax_ds = getNextInput<bool>(stackGetter);
  auto fwd_out = getNextInput<TensorsPair>(stackGetter);

  bool use_fwd_out = GET_ENV_FLAG_NEW(PT_HPU_SDPA_SFMX_BWD_V2);

  ns_Sdpa::ParamsV3 params{};
  unsigned int flags = 0;
  SDPA_SET_FLAGS(is_amax_ds, flags, AMAX_dS)
  SDPA_SET_FLAGS(d_scale_q, flags, D_SCALE_Q)
  SDPA_SET_FLAGS(d_scale_k, flags, D_SCALE_K)
  SDPA_SET_FLAGS(d_scale_v, flags, D_SCALE_V)

  SDPA_SET_FLAGS(d_scale_s, flags, D_SCALE_S)
  SDPA_SET_FLAGS(d_scale_do, flags, D_SCALE_dO)
  SDPA_SET_FLAGS(d_scale_ds, flags, D_SCALE_dS)

  SDPA_SET_FLAGS(q_scale_s, flags, Q_SCALE_S)
  SDPA_SET_FLAGS(q_scale_ds, flags, Q_SCALE_dS)

  fillSdpaParams(
      params,
      p,
      scale,
      is_causal,
      false /*is_inference*/,
      "None" /*softmax_mode*/,
      flags);

  // TODO: check if  143 or 152 may matter
  std::string guid = get_guid_with_precision("sdpa_bwd", q.pt_t.scalar_type());

  std::vector<synTensor> syn_inputs = {
      grad.syn_t, q.syn_t, k.syn_t, v.syn_t, P.syn_t};
  if (p > 0.0) {
    syn_inputs.push_back(dm.value().syn_t);
  } else {
    syn_inputs.push_back(nullptr);
  }
  SDPA_ADD_INPUTS(d_scale_q)
  SDPA_ADD_INPUTS(d_scale_k)
  SDPA_ADD_INPUTS(d_scale_v)

  SDPA_ADD_INPUTS(d_scale_s)
  SDPA_ADD_INPUTS(d_scale_do)
  SDPA_ADD_INPUTS(d_scale_ds)
  SDPA_ADD_INPUTS(q_scale_s)
  SDPA_ADD_INPUTS(q_scale_ds)

  if (use_fwd_out) {
    syn_inputs.push_back(fwd_out.syn_t);
  }

  auto out_shapes = Fp8SDPABwdOutputShape(stack);
  // set gradType to BF16 for now.
  auto gradType = at::ScalarType::BFloat16;
  std::vector<NodeAttr::NodeOutputAttr> output_attrs = {
      {out_shapes[0], gradType, 0},
      {out_shapes[1], gradType, 1},
      {out_shapes[2], gradType, 2}};
  if (is_amax_ds) {
    output_attrs.push_back({out_shapes[3], c10::ScalarType::Float, 3});
  }

  auto output = OpBackend::BuildNode(
      this, graph, {guid, syn_inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(output[0]);
  syn_out(1) = std::move(output[1]);
  syn_out(2) = std::move(output[2]);
  if (is_amax_ds) {
    syn_out(3) = std::move(output[3]);
  }

  //===========================================
#if 0
ns_Sdpa::ParamsV3 params{};
  unsigned int flags = 0;

  SDPA_SET_FLAGS(is_amax_s, flags, AMAX_S)
  SDPA_SET_FLAGS(is_amax_o, flags, AMAX_O)
  SDPA_SET_FLAGS(d_scale_q, flags, D_SCALE_Q)
  SDPA_SET_FLAGS(d_scale_k, flags, D_SCALE_K)
  SDPA_SET_FLAGS(d_scale_v, flags, D_SCALE_V)
  SDPA_SET_FLAGS(q_scale_s, flags, Q_SCALE_S)
  SDPA_SET_FLAGS(q_scale_o, flags, Q_SCALE_O)
  SDPA_SET_FLAGS(q_scale_s, flags, D_SCALE_S)
  // if (d_scale_s) {
  // TODO: add the flag definition to perf_lib_layer_paras.h
  // flags |= (1 << 13);
  //}

  fillSdpaParams(
      params,
      p,
      scale,
      is_causal,
      !requires_backward /*is_inference*/,
      softmax_mode,
      flags);

  std::string guid =
      get_guid_with_precision("sdpa_recomp_fwd", q.pt_t.scalar_type());

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

  SDPA_ADD_INPUTS(d_scale_q)
  SDPA_ADD_INPUTS(d_scale_k)
  SDPA_ADD_INPUTS(d_scale_v)
  SDPA_ADD_INPUTS(q_scale_s)
  SDPA_ADD_INPUTS(q_scale_o)
  SDPA_ADD_INPUTS(d_scale_s)

  std::vector<NodeAttr::NodeOutputAttr> output_attrs;

  auto out_shapes = Fp8SDPARecompFwdOutputShape(stack);

  auto fwdOutType = q.pt_t.scalar_type();

  if (q.pt_t.scalar_type() == at::ScalarType::Float8_e4m3fn) {
    if (q_scale_o) {
      fwdOutType = at::ScalarType::Float8_e4m3fn;
    } else {
      fwdOutType = at::ScalarType::BFloat16;
    }
  }

  auto linvType = c10::ScalarType::Float;
  auto mType = q.pt_t.scalar_type();
  if (requires_backward || is_amax) {
    if ((softmax_mode == "fast") &&
        (q.pt_t.scalar_type() == c10::ScalarType::BFloat16)) {
      linvType = c10::ScalarType::BFloat16;
    }
    if (q.pt_t.scalar_type() == at::ScalarType::Float8_e4m3fn) {
      linvType = c10::ScalarType::BFloat16;
    }

    if (q.pt_t.scalar_type() == at::ScalarType::Float8_e4m3fn) {
      mType = c10::ScalarType::BFloat16;
    }
  }
  output_attrs.push_back({out_shapes[0], fwdOutType, 0});

  // when amax_s/o is needed, we need to have all outputs preceeding amax_s/o.
  // If any preceding output is not valid, like softmax stats, seed etc  in
  // inference, we still need to have outputs for these. These will then be
  // dummy outputs with shape {1}
  if (requires_backward || is_amax) {
    output_attrs.push_back({out_shapes[1], mType, 1});
    output_attrs.push_back({out_shapes[2], linvType, 2});
    if (p > 0.0 || is_amax) {
      output_attrs.push_back({out_shapes[3], at::ScalarType::Int, 3});
    }
  }
  if (is_amax) {
    output_attrs.push_back({out_shapes[4], c10::ScalarType::Float, 4});
  }
  if (is_amax_o) {
    output_attrs.push_back({out_shapes[5], c10::ScalarType::Float, 5});
  }

  auto output = OpBackend::BuildNode(
      this, graph, {guid, syn_inputs, output_attrs, &params, sizeof(params)});
  syn_out(0) = std::move(output[0]);
  if (requires_backward || is_amax_s) {
    syn_out(1) = std::move(output[1]);
    syn_out(2) = std::move(output[2]);
    if (p > 0.0 || is_amax) {
      syn_out(3) = std::move(output[3]);
    }
  }
  if (is_amax) {
    syn_out(4) = std::move(output[4]);
  }
  if (is_amax_o) {
    syn_out(5) = std::move(output[5]);
  }
#endif
  //===========================================
}

//============= ComputeShape and AddNode for SDPA recompute variant=========
template <class DimT>
sizes_vec_template<DimT> SDPARecompFwdOutputShapeCommon(
    c10::ArrayRef<DimT> q_shape,
    c10::ArrayRef<DimT> k_shape,
    c10::ArrayRef<DimT> v_shape,
    bool requires_backward) {
  int64_t rank = q_shape.size();

  // q, k, v are involved in matmuls (BatchGemm) in attention calc.
  // matmul allows broadcast for the batch dims. i.e, for dims 0, 1.. Rank - 2.
  // So output sizes depend on broadcasting. So get sizes at batch dims for
  // q,k v and compute sizes on these dims after broadcast is done.
  // As per torch.nn.functional.scaled_dot_product_attention() documentation,
  // dim0 can also be excluded from broadcast considerations since the doc
  // mentions that dim0 will be batch size(N) for all tensors.
  // However, in the following shape calculations, dim0 is considered for
  // broad cast shape calc for ease of implementation
  std::vector<DimT> q_bdim_sizes{q_shape.begin(), q_shape.begin() + rank - 2};
  std::vector<DimT> k_bdim_sizes{k_shape.begin(), k_shape.begin() + rank - 2};
  std::vector<DimT> v_bdim_sizes{v_shape.begin(), v_shape.begin() + rank - 2};

  // The following infer sizes also serves to do the shape compatibility
  // checks needed for dynamic shapes causing an exception if the shapes
  // are not compatible for broadcast.
  // Assumption: Only the batch dims are checked for compatibility.
  // The matrix dims are assumed to conform to matrix mul rules.

  // Batch dim sizes of Q@K.transpose after broadcast
  // Softmax stats shape is same as Q@K.transpose shape except for the
  // last dim which is 1
  auto softmax_stats_shape =
      infer_size_int_or_symint(q_bdim_sizes, k_bdim_sizes);
  // Batch dim sizes of output  i.e Q@K.transpose)@v after broadcast
  auto out_shape = infer_size_int_or_symint(softmax_stats_shape, v_bdim_sizes);

  // Append the matrix dims (last 2 dims) to batch dims to get final shape.
  int L_dim = rank - 2; // Target seq len dim
  int Ev_dim = rank - 1; // head_dim_v dim

  out_shape.push_back(q_shape[L_dim]);
  out_shape.push_back(v_shape[Ev_dim]);

  softmax_stats_shape.push_back(q_shape[L_dim]);
  // last dim which of Softmax stats shape is 1
  softmax_stats_shape.push_back(1);
  if (!requires_backward) {
    return {out_shape, {1}, {1}, {1}};
  }
  return {out_shape, softmax_stats_shape, softmax_stats_shape, {1}};
}

sizes_vec SDPARecompFwdOutputShape(const at::Stack& stack) {
  auto q_or_seed = stack_tensor(stack, 0);
  int q_index = (q_or_seed.sizes().vec().size() < 3) ? 1 : 0;

  auto q = stack_tensor(stack, q_index);
  auto k = stack_tensor(stack, q_index + 1);
  auto v = stack_tensor(stack, q_index + 2);
  auto requires_backward = stack.at(q_index + 7).toBool();

  return SDPARecompFwdOutputShapeCommon(
      q.sizes(), k.sizes(), v.sizes(), requires_backward);
}

sym_sizes_vec sdpa_recomp_fwd_out_shape(
    const std::vector<at::Tensor>& inputs,
    const std::vector<int64_t>& params) {
  TORCH_CHECK(inputs.size() == 3);
  TORCH_CHECK(params.size() == 1);
  return SDPARecompFwdOutputShapeCommon(
      inputs[0].sym_sizes(),
      inputs[1].sym_sizes(),
      inputs[2].sym_sizes(),
      static_cast<bool>(params[0]));
}

sym_sizes_vec fp8_sdpa_recomp_fwd_out_shape(
    const std::vector<at::Tensor>& inputs,
    const std::vector<int64_t>& params) {
  TORCH_CHECK(inputs.size() == 3);
  TORCH_CHECK(params.size() == 1);
  sym_sizes_vec out_sizes = SDPARecompFwdOutputShapeCommon(
      inputs[0].sym_sizes(),
      inputs[1].sym_sizes(),
      inputs[2].sym_sizes(),
      static_cast<bool>(params[0]));
  out_sizes.push_back({1});
  out_sizes.push_back({1});
  return out_sizes;
}

REGISTER_CUSTOM_OP_OUTSHAPE_FUN(sdpa_recomp_fwd, sdpa_recomp_fwd_out_shape);
REGISTER_CUSTOM_OP_OUTSHAPE_FUN(
    fp8_sdpa_recomp_fwd,
    fp8_sdpa_recomp_fwd_out_shape);

sizes_vec Fp8SDPARecompFwdOutputShape(const at::Stack& stack) {
  sizes_vec out_shape = SDPARecompFwdOutputShape(stack);
  // insert amax_s shape
  out_shape.push_back({1});
  // insert amax_o shape
  out_shape.push_back({1});
  return out_shape;
}

sizes_vec SDPARecompBwdOutputShape(const at::Stack& stack) {
  auto q = stack_tensor(stack, 1);
  auto k = stack_tensor(stack, 2);
  auto v = stack_tensor(stack, 3);

  std::vector<int64_t> q_shape = q.sizes().vec();
  std::vector<int64_t> k_shape = k.sizes().vec();
  std::vector<int64_t> v_shape = v.sizes().vec();
  // TODO: Add shape checks needed for DS:
  // It must be enough to do shape checks similar to FWD pass
  // in the case of BWD with recomp.
  return {q_shape, k_shape, v_shape};
}

void SDPARecompFwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "SDPARecompFwd::AddNode");
  auto q_or_seed = stack[0].toTensor();
  bool seed_present = (q_or_seed.sizes().vec().size() < 3);

  synTensor seed_tensor = nullptr;
  if (seed_present) {
    auto seed = getNextInput<TensorsPair>(stackGetter);
    seed_tensor = seed.syn_t;
  }
  auto q = getNextInput<TensorsPair>(stackGetter);
  auto k = getNextInput<TensorsPair>(stackGetter);
  auto v = getNextInput<TensorsPair>(stackGetter);
  auto attention_mask = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto p = getNextInput<double>(stackGetter);
  auto scale = getNextInput<double>(stackGetter);
  auto is_causal = getNextInput<bool>(stackGetter);
  auto requires_backward = getNextInput<bool>(stackGetter);
  auto softmax_mode = getNextInput<c10::string_view>(stackGetter);
  auto valid_seq_len = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto seq_padding_type = getNextInput<c10::string_view>(stackGetter);
  unsigned int flags = 0;

  SDPA_SET_FLAGS(valid_seq_len, flags, VALID_SEQ_LEN_PRESENT)
  SDPA_SET_FLAGS(seq_padding_type == "left", flags, SEQ_PADDING_LEFT)
  SDPA_SET_FLAGS(seq_padding_type == "right", flags, SEQ_PADDING_RIGHT)

  ns_Sdpa::ParamsV3 params{};
  fillSdpaParams(
      params,
      p,
      scale,
      is_causal,
      !requires_backward /*is_inference*/,
      softmax_mode,
      flags);

  std::string guid =
      get_guid_with_precision("sdpa_recomp_fwd", q.pt_t.scalar_type());
  auto out_shapes = SDPARecompFwdOutputShape(stack);

  std::vector<synTensor> syn_inputs = {q.syn_t, k.syn_t, v.syn_t};
  if (attention_mask) {
    syn_inputs.push_back(attention_mask.value().syn_t);
  } else {
    syn_inputs.push_back(nullptr);
  }
  syn_inputs.push_back(seed_tensor);
  if (valid_seq_len) {
    syn_inputs.insert(syn_inputs.end(), 6, nullptr);
    syn_inputs.push_back(valid_seq_len.value().syn_t);
  } else {
    syn_inputs.insert(syn_inputs.end(), 7, nullptr);
  }

  std::vector<NodeAttr::NodeOutputAttr> output_attrs;
  output_attrs.push_back({out_shapes[0], q.pt_t.scalar_type(), 0});
  if (requires_backward) {
    auto linvType = c10::ScalarType::Float;

    if ((softmax_mode == "fast") &&
        (q.pt_t.scalar_type() == c10::ScalarType::BFloat16)) {
      linvType = c10::ScalarType::BFloat16;
    }
    output_attrs.push_back({out_shapes[1], q.pt_t.scalar_type(), 1});
    output_attrs.push_back({out_shapes[2], linvType, 2});
    if (p > 0.0) {
      output_attrs.push_back({out_shapes[3], at::ScalarType::Int, 3});
    }
  }

  auto output = OpBackend::BuildNode(
      this, graph, {guid, syn_inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(output[0]);
  if (requires_backward) {
    syn_out(1) = std::move(output[1]);
    syn_out(2) = std::move(output[2]);
    if (p > 0.0) {
      syn_out(3) = std::move(output[3]);
    }
  }
}

void Fp8SDPARecompFwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "Fp8SDPARecompFwd::AddNode");
  auto q_or_seed = stack[0].toTensor();
  bool seed_present = (q_or_seed.sizes().vec().size() < 3);
  synTensor seed_tensor = nullptr;
  if (seed_present) {
    auto seed = getNextInput<TensorsPair>(stackGetter);
    seed_tensor = seed.syn_t;
  }
  auto q = getNextInput<TensorsPair>(stackGetter);
  auto k = getNextInput<TensorsPair>(stackGetter);
  auto v = getNextInput<TensorsPair>(stackGetter);
  auto attention_mask = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto p = getNextInput<double>(stackGetter);
  auto scale = getNextInput<double>(stackGetter);
  auto is_causal = getNextInput<bool>(stackGetter);
  auto requires_backward = getNextInput<bool>(stackGetter);
  auto softmax_mode = getNextInput<c10::string_view>(stackGetter);
  auto d_scale_q = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto d_scale_k = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto d_scale_v = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto q_scale_s = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto q_scale_o = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto d_scale_s = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto is_amax_s = getNextInput<bool>(stackGetter);
  auto is_amax_o = getNextInput<bool>(stackGetter);
  // amax_s and/or amax_o needed
  bool is_amax = is_amax_s or is_amax_o;

  auto valid_seq_len = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto seq_padding_type = getNextInput<c10::string_view>(stackGetter);

  ns_Sdpa::ParamsV3 params{};
  unsigned int flags = 0;

  SDPA_SET_FLAGS(is_amax_s, flags, AMAX_S)
  SDPA_SET_FLAGS(is_amax_o, flags, AMAX_O)
  SDPA_SET_FLAGS(d_scale_q, flags, D_SCALE_Q)
  SDPA_SET_FLAGS(d_scale_k, flags, D_SCALE_K)
  SDPA_SET_FLAGS(d_scale_v, flags, D_SCALE_V)
  SDPA_SET_FLAGS(q_scale_s, flags, Q_SCALE_S)
  SDPA_SET_FLAGS(q_scale_o, flags, Q_SCALE_O)
  SDPA_SET_FLAGS(d_scale_s, flags, D_SCALE_S)
  SDPA_SET_FLAGS(valid_seq_len, flags, VALID_SEQ_LEN_PRESENT)
  SDPA_SET_FLAGS(seq_padding_type == "left", flags, SEQ_PADDING_LEFT)
  SDPA_SET_FLAGS(seq_padding_type == "right", flags, SEQ_PADDING_RIGHT)
  // if (d_scale_s) {
  // TODO: add the flag definition to perf_lib_layer_paras.h
  // flags |= (1 << 13);
  //}

  fillSdpaParams(
      params,
      p,
      scale,
      is_causal,
      !requires_backward /*is_inference*/,
      softmax_mode,
      flags);

  std::string guid =
      get_guid_with_precision("sdpa_recomp_fwd", q.pt_t.scalar_type());

  std::vector<synTensor> syn_inputs = {q.syn_t, k.syn_t, v.syn_t};
  if (attention_mask) {
    syn_inputs.push_back(attention_mask.value().syn_t);
  } else {
    syn_inputs.push_back(nullptr);
  }
  syn_inputs.push_back(seed_tensor);

  SDPA_ADD_INPUTS(d_scale_q)
  SDPA_ADD_INPUTS(d_scale_k)
  SDPA_ADD_INPUTS(d_scale_v)
  SDPA_ADD_INPUTS(q_scale_s)
  SDPA_ADD_INPUTS(q_scale_o)
  SDPA_ADD_INPUTS(d_scale_s)

  if (valid_seq_len) {
    syn_inputs.push_back(valid_seq_len.value().syn_t);
  }

  std::vector<NodeAttr::NodeOutputAttr> output_attrs;

  auto out_shapes = Fp8SDPARecompFwdOutputShape(stack);

  auto fwdOutType = q.pt_t.scalar_type();

  if (q.pt_t.scalar_type() == at::ScalarType::Float8_e4m3fn) {
    if (q_scale_o) {
      fwdOutType = at::ScalarType::Float8_e4m3fn;
    } else {
      fwdOutType = at::ScalarType::BFloat16;
    }
  }

  auto linvType = c10::ScalarType::Float;
  auto mType = q.pt_t.scalar_type();
  if (requires_backward || is_amax) {
    if ((softmax_mode == "fast") &&
        (q.pt_t.scalar_type() == c10::ScalarType::BFloat16)) {
      linvType = c10::ScalarType::BFloat16;
    }
    if (q.pt_t.scalar_type() == at::ScalarType::Float8_e4m3fn) {
      linvType = c10::ScalarType::BFloat16;
    }

    if (q.pt_t.scalar_type() == at::ScalarType::Float8_e4m3fn) {
      mType = c10::ScalarType::BFloat16;
    }
  }
  output_attrs.push_back({out_shapes[0], fwdOutType, 0});

  // when amax_s/o is needed, we need to have all outputs preceeding amax_s/o.
  // If any preceding output is not valid, like softmax stats, seed etc  in
  // inference, we still need to have outputs for these. These will then be
  // dummy outputs with shape {1}
  if (requires_backward || is_amax) {
    output_attrs.push_back({out_shapes[1], mType, 1});
    output_attrs.push_back({out_shapes[2], linvType, 2});
    if (p > 0.0 || is_amax) {
      output_attrs.push_back({out_shapes[3], at::ScalarType::Int, 3});
    }
  }
  if (is_amax) {
    output_attrs.push_back({out_shapes[4], c10::ScalarType::Float, 4});
  }
  if (is_amax_o) {
    output_attrs.push_back({out_shapes[5], c10::ScalarType::Float, 5});
  }

  auto output = OpBackend::BuildNode(
      this, graph, {guid, syn_inputs, output_attrs, &params, sizeof(params)});
  syn_out(0) = std::move(output[0]);
  if (requires_backward || is_amax_s) {
    syn_out(1) = std::move(output[1]);
    syn_out(2) = std::move(output[2]);
    if (p > 0.0 || is_amax) {
      syn_out(3) = std::move(output[3]);
    }
  }
  if (is_amax) {
    syn_out(4) = std::move(output[4]);
  }
  if (is_amax_o) {
    syn_out(5) = std::move(output[5]);
  }
}

void SDPARecompBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "SDPARecompBwd::AddNode");
  auto grad = getNextInput<TensorsPair>(stackGetter);
  auto q = getNextInput<TensorsPair>(stackGetter);
  auto k = getNextInput<TensorsPair>(stackGetter);
  auto v = getNextInput<TensorsPair>(stackGetter);
  auto attention_mask = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto m = getNextInput<TensorsPair>(stackGetter);
  auto linv = getNextInput<TensorsPair>(stackGetter);
  auto seed = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto is_causal = getNextInput<bool>(stackGetter);
  auto p = getNextInput<double>(stackGetter);
  auto scale = getNextInput<double>(stackGetter);
  auto softmax_mode = getNextInput<c10::string_view>(stackGetter);
  auto fwd_out = getNextInput<TensorsPair>(stackGetter);

  bool use_fwd_out = GET_ENV_FLAG_NEW(PT_HPU_SDPA_SFMX_BWD_V2);

  ns_Sdpa::ParamsV3 params{};
  fillSdpaParams(
      params, p, scale, is_causal, false /*is_inference*/, softmax_mode);

  std::string guid =
      get_guid_with_precision("sdpa_recomp_bwd", q.pt_t.scalar_type());
  auto out_shapes = SDPARecompBwdOutputShape(stack);

  std::vector<synTensor> syn_inputs = {grad.syn_t, q.syn_t, k.syn_t, v.syn_t};

  if (attention_mask) {
    syn_inputs.push_back(attention_mask.value().syn_t);
  } else {
    syn_inputs.push_back(nullptr);
  }
  syn_inputs.push_back(m.syn_t);
  syn_inputs.push_back(linv.syn_t);
  if (p > 0.0) {
    syn_inputs.push_back(seed.value().syn_t);
  } else {
    syn_inputs.push_back(nullptr);
  }

  if (use_fwd_out) {
    syn_inputs.push_back(fwd_out.syn_t);
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
        .add("hpu::sdpa_fwd_dropout_seed", KERNEL_FN_GLOBAL(habana::SDPAFwd))
        .add("hpu::sdpa_fwd_non_dropout", KERNEL_FN_GLOBAL(habana::SDPAFwd))
        .add("hpu::sdpa_fwd", KERNEL_FN_GLOBAL(habana::SDPAFwd))
        .add("hpu::sdpa_bwd", KERNEL_FN_GLOBAL(habana::SDPABwd))
        .add("hpu::fp8_sdpa_bwd", KERNEL_FN_GLOBAL(habana::Fp8SDPABwd))
        .add("hpu::sdpa_recomp_fwd", KERNEL_FN_GLOBAL(habana::SDPARecompFwd))
        .add(
            "hpu::sdpa_recomp_fwd_non_dropout",
            KERNEL_FN_GLOBAL(habana::SDPARecompFwd))
        .add(
            "hpu::sdpa_recomp_fwd_dropout_seed",
            KERNEL_FN_GLOBAL(habana::SDPARecompFwd))
        .add("hpu::fp8_sdpa_fwd", KERNEL_FN_GLOBAL(habana::Fp8SDPAFwd))
        .add(
            "hpu::fp8_sdpa_fwd_dropout_seed",
            KERNEL_FN_GLOBAL(habana::Fp8SDPAFwd))
        .add(
            "hpu::fp8_sdpa_fwd_non_dropout",
            KERNEL_FN_GLOBAL(habana::Fp8SDPAFwd))
        .add(
            "hpu::fp8_sdpa_recomp_fwd",
            KERNEL_FN_GLOBAL(habana::Fp8SDPARecompFwd))
        .add(
            "hpu::fp8_sdpa_recomp_fwd_dropout_seed",
            KERNEL_FN_GLOBAL(habana::Fp8SDPARecompFwd))
        .add(
            "hpu::fp8_sdpa_recomp_fwd_non_dropout",
            KERNEL_FN_GLOBAL(habana::Fp8SDPARecompFwd))

        .add("hpu::sdpa_recomp_bwd", KERNEL_FN_GLOBAL(habana::SDPARecompBwd));
