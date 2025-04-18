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

#include <ATen/ATen.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorShape.h>
#include <torch/library.h>
#include "backend/random.h"
#include "common/dump_args.h"
#include "common/random_utils.h"
#include "generated/backend/one_hot.h"
#include "habana_eager/graph_weight_permute.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_eager/ops/mixture_of_experts.h"
#include "habana_helpers/logging.h"
#include "hpu_ops/cpu_fallback.h"
#include "hpu_ops/fp8_ops.h"
#include "hpu_ops/op_logger.h"
#include "hpu_ops/op_validator.h"
#include "hpu_ops/optimizer_lamb_gen.h"
#include "hpu_ops/sdpa_gen.h"
#include "hpu_ops/shared_meta_common.h"
#include "ops/batch_as_strided.h"

using namespace habana;

namespace {
using habana::dispatch_fallback; // For VAL_CUSTOM_FALLBACK_IF_UNSUPPORTED_DTYPE
using habana::to_string; // For DUMP_*ARGS

/***********************************************************************************
 * Custom ops
 **********************************************************************************/

at::Tensor optimizer_lamb_norm(
    const std::vector<at::Tensor>& grad,
    double max_grad_norm) {
  PT_EAGER_TRACE;

  habana::EagerOptimizerLambNorm<at::Tensor> hpu_op{
      "hpu::optimizer_lamb_fused_norm", {grad, max_grad_norm}};
  return hpu_op.call();
}

void optimizer_resource_apply_momentum(
    at::TensorList params_momentum_buf_list,
    const at::TensorList dp_list,
    const double momentum) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "optimizer_resource_apply_momentum :",
      DUMP_3ARGS(params_momentum_buf_list, dp_list, momentum));

  habana::eager::EagerOp<void> hpu_op{
      "hpu::optimizer_resource_apply_momentum",
      {params_momentum_buf_list, dp_list, momentum}};

  hpu_op.call(params_momentum_buf_list);
}

void optimizer_lars(
    const at::TensorList params,
    at::TensorList grads,
    c10::ArrayRef<int64_t> skip_masks,
    const double eeta,
    const double weight_decay,
    const double eps,
    const at::Tensor& lr) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      " optimizer_lars :",
      DUMP_7ARGS(params, grads, skip_masks, eeta, weight_decay, eps, lr));

  habana::eager::EagerOp<void> hpu_op{
      "hpu::optimizer_lars",
      {params, grads, skip_masks, eeta, weight_decay, eps, lr}};

  hpu_op.call(grads);
}

void optimizer_lamb_phase1(
    const at::TensorList gradients,
    const at::TensorList weights,
    at::TensorList exp_avg,
    at::TensorList exp_avg_sq,
    at::TensorList out_weight_norms,
    at::TensorList out_adam_norms,
    at::TensorList out_adam_steps,
    const at::Tensor clip_global_grad_norm,
    const int64_t grad_averaging,
    const double beta1,
    const double beta2,
    const double epsilon,
    const at::Tensor bias_correction1,
    const at::Tensor bias_correction2,
    const double weight_decay) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "optimizer_lamb_phase1:",
      DUMP_12ARGS(
          gradients,
          weights,
          exp_avg,
          exp_avg_sq,
          clip_global_grad_norm,
          grad_averaging,
          beta1,
          beta2,
          epsilon,
          bias_correction1,
          bias_correction2,
          weight_decay));

  habana::eager::EagerOp<void> hpu_op{
      "hpu::optimizer_lamb_phase1",
      {gradients,
       weights,
       exp_avg,
       exp_avg_sq,
       out_weight_norms,
       out_adam_norms,
       out_adam_steps,
       clip_global_grad_norm,
       grad_averaging,
       beta1,
       beta2,
       epsilon,
       bias_correction1,
       bias_correction2,
       weight_decay}};

  hpu_op.set_eager_op_info(
      {habana::eager::eagerOpKind::Inplace,
       "hpu::optimizer_lamb_phase1",
       decltype(habana::eager::EagerOpMetaData::out_indices_){2, 3, 4, 5, 6}});

  std::vector<at::TensorList> tensorlists = {
      exp_avg, exp_avg_sq, out_weight_norms, out_adam_norms, out_adam_steps};

  return hpu_op.call(tensorlists);
}

void optimizer_lamb_phase2(
    at::TensorList weights,
    const at::TensorList adam_norms,
    const at::TensorList weight_norms,
    const at::TensorList adam_steps,
    const at::Tensor& neg_step,
    const double weight_decay,
    const bool use_lamb) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "optimizer_lamb_phase2:",
      DUMP_7ARGS(
          weights,
          adam_norms,
          weight_norms,
          adam_steps,
          neg_step,
          weight_decay,
          use_lamb));

  habana::eager::EagerOp<void> hpu_op{
      "hpu::optimizer_lamb_phase2",
      {weights,
       adam_norms,
       weight_norms,
       adam_steps,
       neg_step,
       weight_decay,
       use_lamb}};
  hpu_op.set_eager_op_info(
      {habana::eager::eagerOpKind::Inplace,
       "hpu::optimizer_lamb_phase2",
       decltype(habana::eager::EagerOpMetaData::out_indices_){0}});
  return hpu_op.call(weights);
}

void optimizer_ema(
    const at::TensorList model_inputs,
    at::TensorList updated_ema,
    const at::Tensor& decay) {
  PT_EAGER_TRACE;
  PT_OP_INFO(" optimizer_ema :", DUMP_3ARGS(model_inputs, updated_ema, decay));

  habana::eager::EagerOp<void> hpu_op{
      "hpu::optimizer_ema", {model_inputs, updated_ema, decay}};
  hpu_op.call(updated_ema);
}

void optimizer_adamw(
    const at::TensorList gradient_vec,
    at::TensorList weight_vec,
    at::TensorList exp_avg_vec,
    at::TensorList exp_avg_sq_vec,
    const at::Tensor& neg_step_t,
    const double beta1,
    const double beta2,
    const double epsilon,
    const at::Tensor& weight_decay,
    const bool has_weight_decay,
    std::optional<at::TensorList> exp_avg_scales = std::nullopt,
    std::optional<at::TensorList> exp_avg_sq_scales = std::nullopt) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "optimizer_adamw :",
      DUMP_12ARGS(
          gradient_vec,
          weight_vec,
          exp_avg_vec,
          exp_avg_sq_vec,
          neg_step_t,
          beta1,
          beta2,
          epsilon,
          weight_decay,
          has_weight_decay,
          exp_avg_scales,
          exp_avg_sq_scales));

  HABANA_ASSERT(
      (weight_vec.size() > 0),
      "optimizer_adamw : can not process empty weight vector");
  HABANA_ASSERT(
      exp_avg_scales.has_value() == exp_avg_sq_scales.has_value(),
      "optimizer_adamw : expects both or neighter scales to be set");

  habana::eager::EagerOp<void> hpu_op{
      "hpu::optimizer_adamw",
      {gradient_vec,
       weight_vec,
       exp_avg_vec,
       exp_avg_sq_vec,
       neg_step_t,
       beta1,
       beta2,
       epsilon,
       weight_decay,
       has_weight_decay,
       exp_avg_scales,
       exp_avg_sq_scales}};

  hpu_op.set_eager_op_info(
      {habana::eager::eagerOpKind::Inplace,
       "hpu::optimizer_adamw",
       {1, 2, 3, 10, 11}});

  std::vector<at::TensorList> tensorlists = {
      weight_vec, exp_avg_vec, exp_avg_sq_vec};
  if (exp_avg_scales.has_value()) {
    tensorlists.push_back(exp_avg_scales.value());
    tensorlists.push_back(exp_avg_sq_scales.value());
  }
  hpu_op.call(tensorlists);
}

at::Tensor fused_clip_norm(
    at::TensorList grad,
    const at::Tensor& max_norm,
    double norm_type) {
  PT_EAGER_TRACE;
  PT_OP_INFO("fused_clip_norm :", DUMP_3ARGS(grad, max_norm, norm_type));

  HABANA_ASSERT(
      (grad.size() > 0),
      "fused_clip_norm : can not process empty grad vector (eager)");

  habana::eager::EagerOp<void> hpu_op{
      "hpu::fused_clip_norm", {grad, max_norm, norm_type}};

  hpu_op.set_eager_op_info(
      {habana::eager::eagerOpKind::Inplace,
       "hpu::fused_clip_norm",
       decltype(habana::eager::EagerOpMetaData::out_indices_){0}});

  hpu_op.call(grad);

  // return the total_norm result from the end of the grad vector
  return grad.back();
}

void optimizer_sgd(
    const at::TensorList gradients,
    at::TensorList weights,
    at::Tensor& lr,
    double wd,
    double mom,
    double damp,
    bool nesterov) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      " optimizer_sgd:",
      DUMP_7ARGS(gradients, weights, lr, wd, mom, damp, nesterov));
  HABANA_ASSERT(
      (weights.size() > 0),
      "optimizer_sgd : can not process empty weight vector");
  habana::eager::EagerOp<void> hpu_op{
      "hpu::optimizer_sgd", {gradients, weights, lr, wd, mom, damp, nesterov}};
  hpu_op.set_eager_op_info(
      {habana::eager::eagerOpKind::Inplace,
       "hpu::optimizer_sgd",
       decltype(habana::eager::EagerOpMetaData::out_indices_){1}});
  hpu_op.call({weights});
}

void optimizer_sgd_momentum(
    const at::TensorList gradients,
    at::TensorList weights,
    at::TensorList momentum,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    at::Tensor& mom,
    double wd,
    double damp,
    bool nesterov) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      " optimizer_sgd_momentum:",
      DUMP_9ARGS(
          gradients,
          weights,
          momentum,
          epoch_num,
          lr,
          mom,
          wd,
          damp,
          nesterov));
  HABANA_ASSERT(
      (weights.size() > 0),
      "optimizer_sgd_momentum : can not process empty weight vector");
  habana::eager::EagerOp<void> hpu_op{
      "hpu::optimizer_sgd_momentum",
      {gradients, weights, momentum, epoch_num, lr, mom, wd, damp, nesterov}};
  hpu_op.set_eager_op_info(
      {habana::eager::eagerOpKind::Inplace,
       "hpu::optimizer_sgd_momentum",
       decltype(habana::eager::EagerOpMetaData::out_indices_){1, 2}});
  hpu_op.call({weights, momentum});
}

at::Tensor kv_reorder(
    const at::Tensor& self,
    const at::Tensor& start,
    const at::Tensor& end,
    const at::Tensor& beam_idx) {
  PT_EAGER_TRACE;
  PT_OP_INFO("kv_reorder :", DUMP_4ARGS(self, start, end, beam_idx));

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::kv_reorder", {self, start, end, beam_idx}, {{self.sizes().vec()}}};
  return hpu_op.call();
}

at::Tensor in_place_interleave(const at::Tensor& self) {
  PT_EAGER_TRACE;
  PT_OP_INFO("in_place_interleave :", DUMP_ARG(self));

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::in_place_interleave", {self}, {{self.sizes().vec()}}};
  return hpu_op.call();
}

at::Tensor slice_ds(
    const at::Tensor& self,
    c10::SymInt dim,
    c10::SymInt start,
    c10::SymInt end,
    c10::SymInt step,
    [[maybe_unused]] std::optional<c10::SymIntArrayRef> size) {
  PT_EAGER_TRACE;
  PT_OP_INFO("slice_ds :", DUMP_5ARGS(self, size, dim, start, end));
  return at::native::slice(
      self,
      dim.expect_int(),
      start.expect_int(),
      end.expect_int(),
      step.expect_int());
}

at::Tensor constant_pad_nd_ds(
    const at::Tensor& self,
    c10::SymIntArrayRef pad,
    const c10::Scalar& value,
    [[maybe_unused]] std::optional<c10::SymIntArrayRef> size) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "constant_pad_ds :",
      DUMP_3ARGS(self, c10::asIntArrayRefUnchecked(pad), value));
  return at::native::constant_pad_nd(
      self, c10::asIntArrayRefUnchecked(pad), value);
}

// accumulate_grads_ is a wrapper for native inductor.accumulate_grad_ op.
// It extracts gradients from variables and assigns respective new_grads to them
// or increment by them, depending if gradients are defined.
void accumulate_grads_(
    at::TensorList variables,
    const at::TensorList new_grads) {
  PT_EAGER_TRACE;
  PT_OP_INFO("accumulate_grads_ :", DUMP_2ARGS(variables, new_grads));

  HABANA_ASSERT(
      variables.size() == new_grads.size(),
      "Inputs to hpu::accumulate_grads_ must be of the same size, got: ",
      variables.size(),
      " and ",
      new_grads.size());

  if (variables.empty()) {
    PT_BRIDGE_WARN("hpu::accumulate_grads_ received empty inputs.");
    return;
  }

  if (variables[0].mutable_grad().defined()) {
    std::vector<at::Tensor> current_grads_list;
    current_grads_list.reserve(variables.size());
    for (auto& variable : variables) {
      current_grads_list.push_back(variable.mutable_grad());
    }
    habana::eager::EagerOp<void> hpu_op{
        "hpu::custom_foreach_add_", {current_grads_list, new_grads}};
    hpu_op.call(current_grads_list);
  } else {
    for (size_t i = 0; i < variables.size(); ++i) {
      variables[i].mutable_grad() = new_grads[i];
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> sdpa_recomp_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const std::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    const bool requires_backward,
    std::string_view softmax_mode,
    const std::optional<at::Tensor>& valid_seq_len,
    std::string_view seq_padding_type) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "sdpa_recomp_fwd :",
      DUMP_11ARGS(
          q,
          k,
          v,
          attention_mask,
          p,
          scale,
          is_causal,
          requires_backward,
          softmax_mode,
          valid_seq_len,
          seq_padding_type));

  if (p > 0.0) {
    int seed = habana::get_seed_hpu(std::nullopt);
    at::TensorOptions o;
    o = o.dtype(at::kInt).device(at::kHPU);
    at::Tensor seed_t = at::tensor(seed, o);
    habana::eager::EagerOp<
        std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>>
        hpu_op{
            "hpu::sdpa_recomp_fwd_dropout_seed",
            {seed_t,
             q,
             k,
             v,
             attention_mask,
             p,
             scale,
             is_causal,
             requires_backward,
             softmax_mode,
             valid_seq_len,
             seq_padding_type},
            habana::SDPARecompFwdOutputShape};
    hpu_op.set_scalar_types(
        {q.scalar_type(),
         q.scalar_type(),
         c10::ScalarType::Float,
         c10::ScalarType::Int});
    return hpu_op.call();

  } else {
    habana::eager::EagerOp<
        std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>>
        hpu_op{
            "hpu::sdpa_recomp_fwd",
            {q,
             k,
             v,
             attention_mask,
             p,
             scale,
             is_causal,
             requires_backward,
             softmax_mode,
             valid_seq_len,
             seq_padding_type},
            habana::SDPARecompFwdOutputShape};
    auto linvType = c10::ScalarType::Float;

    if ((softmax_mode == "fast") &&
        (q.scalar_type() == c10::ScalarType::BFloat16)) {
      linvType = c10::ScalarType::BFloat16;
    }
    hpu_op.set_scalar_types(
        {q.scalar_type(), q.scalar_type(), linvType, c10::ScalarType::Int});
    return hpu_op.call();
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const std::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    std::string_view softmax_mode,
    const std::optional<at::Tensor>& valid_seq_len,
    std::string_view seq_padding_type) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "sdpa_fwd :",
      DUMP_10ARGS(
          q,
          k,
          v,
          attention_mask,
          p,
          scale,
          is_causal,
          softmax_mode,
          valid_seq_len,
          seq_padding_type));
  if (p > 0.0) {
    int seed = habana::get_seed_hpu(std::nullopt);
    at::TensorOptions o;
    o = o.dtype(at::kInt).device(at::kHPU);
    at::Tensor seed_t = at::tensor(seed, o);
    habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>>
        hpu_op{
            "hpu::sdpa_fwd_dropout_seed",
            {seed_t,
             q,
             k,
             v,
             attention_mask,
             p,
             scale,
             is_causal,
             softmax_mode,
             valid_seq_len,
             seq_padding_type},
            habana::SDPAFwdOutputShape};
    hpu_op.set_scalar_types(
        {q.scalar_type(), q.scalar_type(), c10::ScalarType::Char});
    return hpu_op.call();

  } else {
    habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>>
        hpu_op{
            "hpu::sdpa_fwd",
            {q,
             k,
             v,
             attention_mask,
             p,
             scale,
             is_causal,
             softmax_mode,
             valid_seq_len,
             seq_padding_type},
            habana::SDPAFwdOutputShape};
    hpu_op.set_scalar_types(
        {q.scalar_type(), q.scalar_type(), c10::ScalarType::Char});
    return hpu_op.call();
  }
}

at::Tensor weight_permutation(const at::Tensor& weight) {
  habana::graph::PermuteWeightTensor t(weight);
  t.PermuteIfNeeded();
  return weight;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fp8_sdpa_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const std::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    std::string_view softmax_mode,
    const std::optional<at::Tensor>& d_scale_q,
    const std::optional<at::Tensor>& d_scale_k,
    const std::optional<at::Tensor>& d_scale_v,
    const std::optional<at::Tensor>& q_scale_s,
    const std::optional<at::Tensor>& q_scale_o,
    const std::optional<at::Tensor>& d_scale_s,
    const bool is_amax_s,
    const std::optional<at::Tensor>& valid_seq_len,
    std::string_view seq_padding_type) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "fp8_sdpa_fwd :",
      DUMP_17ARGS(
          q,
          k,
          v,
          attention_mask,
          p,
          scale,
          is_causal,
          softmax_mode,
          d_scale_q,
          d_scale_k,
          d_scale_v,
          q_scale_s,
          q_scale_o,
          d_scale_s,
          is_amax_s,
          valid_seq_len,
          seq_padding_type));
  auto fwdOutType = q.scalar_type();
  auto sfmxType = q.scalar_type();
  // if (is_amax_s) {
  // sfmxType = c10::ScalarType::BFloat16;
  // }

  // Normally SDPA FWD Fp8 dtype is e4m3. Supporting
  // e5m2 for experiments
  if (q.scalar_type() == at::ScalarType::Float8_e4m3fn ||
      q.scalar_type() == at::ScalarType::Float8_e5m2) {
    if (q_scale_o.has_value()) {
      fwdOutType = q.scalar_type();
    } else {
      fwdOutType = at::ScalarType::BFloat16;
    }
  }

  if (p > 0.0) {
    int seed = habana::get_seed_hpu(std::nullopt);
    at::TensorOptions o;
    o = o.dtype(at::kInt).device(at::kHPU);
    at::Tensor seed_t = at::tensor(seed, o);
    habana::eager::EagerOp<
        std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>>
        hpu_op{
            "hpu::fp8_sdpa_fwd_dropout_seed",
            {seed,
             q,
             k,
             v,
             attention_mask,
             p,
             scale,
             is_causal,
             softmax_mode,
             d_scale_q,
             d_scale_k,
             d_scale_v,
             q_scale_s,
             q_scale_o,
             d_scale_s,
             is_amax_s,
             valid_seq_len,
             seq_padding_type},
            habana::Fp8SDPAFwdOutputShape};
    hpu_op.set_scalar_types(
        {fwdOutType, sfmxType, c10::ScalarType::Char, c10::ScalarType::Float});

    return hpu_op.call();

  } else {
    habana::eager::EagerOp<
        std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>>
        hpu_op{
            "hpu::fp8_sdpa_fwd",
            {q,
             k,
             v,
             attention_mask,
             p,
             scale,
             is_causal,
             softmax_mode,
             d_scale_q,
             d_scale_k,
             d_scale_v,
             q_scale_s,
             q_scale_o,
             d_scale_s,
             is_amax_s,
             valid_seq_len,
             seq_padding_type},
            habana::Fp8SDPAFwdOutputShape};
    hpu_op.set_scalar_types(
        {fwdOutType, sfmxType, c10::ScalarType::Char, c10::ScalarType::Float});
    return hpu_op.call();
  }
}

template <class T>
static std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
fp8_sdpa_recomp_fwd_common(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const std::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    const bool requires_backward,
    std::string_view softmax_mode,
    T d_scale_q,
    T d_scale_k,
    T d_scale_v,
    T q_scale_s,
    T q_scale_o,
    T d_scale_s,
    const bool is_amax_s,
    const bool is_amax_o,
    const std::optional<at::Tensor>& valid_seq_len,
    std::string_view seq_padding_type,
    c10::ScalarType fwdOutType) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "fp8_sdpa_recomp_fwd :",
      DUMP_19ARGS(
          q,
          k,
          v,
          attention_mask,
          p,
          scale,
          is_causal,
          requires_backward,
          softmax_mode,
          d_scale_q,
          d_scale_k,
          d_scale_v,
          q_scale_s,
          q_scale_o,
          d_scale_s,
          is_amax_s,
          is_amax_o,
          valid_seq_len,
          seq_padding_type));

  auto linvType = c10::ScalarType::Float;

  if ((softmax_mode == "fast") &&
      (q.scalar_type() == c10::ScalarType::BFloat16)) {
    linvType = c10::ScalarType::BFloat16;
  }
  if (q.scalar_type() == at::ScalarType::Float8_e4m3fn) {
    linvType = c10::ScalarType::BFloat16;
  }

  auto mType = q.scalar_type();
  if (q.scalar_type() == at::ScalarType::Float8_e4m3fn) {
    mType = c10::ScalarType::BFloat16;
  }

  if (p > 0.0) {
    int seed = habana::get_seed_hpu(std::nullopt);
    at::TensorOptions o;
    o = o.dtype(at::kInt).device(at::kHPU);
    at::Tensor seed_t = at::tensor(seed, o);
    habana::eager::EagerOp<std::tuple<
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor>>
        hpu_op{
            "hpu::fp8_sdpa_recomp_fwd_dropout_seed",
            {seed,
             q,
             k,
             v,
             attention_mask,
             p,
             scale,
             is_causal,
             requires_backward,
             softmax_mode,
             d_scale_q,
             d_scale_k,
             d_scale_v,
             q_scale_s,
             q_scale_o,
             d_scale_s,
             is_amax_s,
             is_amax_o,
             valid_seq_len,
             seq_padding_type},
            habana::Fp8SDPARecompFwdOutputShape};
    hpu_op.set_scalar_types(
        {fwdOutType,
         mType,
         linvType,
         c10::ScalarType::Int,
         c10::ScalarType::Float,
         c10::ScalarType::Float});

    return hpu_op.call();
  } else {
    habana::eager::EagerOp<std::tuple<
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor>>
        hpu_op{
            "hpu::fp8_sdpa_recomp_fwd",
            {q,
             k,
             v,
             attention_mask,
             p,
             scale,
             is_causal,
             requires_backward,
             softmax_mode,
             d_scale_q,
             d_scale_k,
             d_scale_v,
             q_scale_s,
             q_scale_o,
             d_scale_s,
             is_amax_s,
             is_amax_o,
             valid_seq_len,
             seq_padding_type},
            habana::Fp8SDPARecompFwdOutputShape};
    hpu_op.set_scalar_types(
        {fwdOutType,
         mType,
         linvType,
         c10::ScalarType::Int,
         c10::ScalarType::Float,
         c10::ScalarType::Float});

    return hpu_op.call();
  }
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
fp8_sdpa_recomp_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const std::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    const bool requires_backward,
    std::string_view softmax_mode,
    const std::optional<at::Tensor>& d_scale_q,
    const std::optional<at::Tensor>& d_scale_k,
    const std::optional<at::Tensor>& d_scale_v,
    const std::optional<at::Tensor>& q_scale_s,
    const std::optional<at::Tensor>& q_scale_o,
    const std::optional<at::Tensor>& d_scale_s,
    const bool is_amax_s,
    const bool is_amax_o,
    const std::optional<at::Tensor>& valid_seq_len,
    std::string_view seq_padding_type) {
  PT_EAGER_TRACE;
  auto fwdOutType = q.scalar_type();

  if (q.scalar_type() == at::ScalarType::Float8_e4m3fn &&
      (!q_scale_o.has_value()))
    fwdOutType = at::ScalarType::BFloat16;
  return fp8_sdpa_recomp_fwd_common<std::optional<at::Tensor>>(
      q,
      k,
      v,
      attention_mask,
      p,
      scale,
      is_causal,
      requires_backward,
      softmax_mode,
      d_scale_q,
      d_scale_k,
      d_scale_v,
      q_scale_s,
      q_scale_o,
      d_scale_s,
      is_amax_s,
      is_amax_o,
      valid_seq_len,
      seq_padding_type,
      fwdOutType);
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
fp8_sdpa_recomp_scalar_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const std::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    const bool requires_backward,
    std::string_view softmax_mode,
    const double d_scale_q,
    const double d_scale_k,
    const double d_scale_v,
    const double q_scale_s,
    const double q_scale_o,
    const double d_scale_s,
    const bool is_amax_s,
    const bool is_amax_o,
    const std::optional<at::Tensor>& valid_seq_len,
    std::string_view seq_padding_type) {
  PT_EAGER_TRACE;
  auto fwdOutType = q.scalar_type();
  if (q.scalar_type() == at::ScalarType::Float8_e4m3fn && (q_scale_o == 0.))
    fwdOutType = at::ScalarType::BFloat16;

  return fp8_sdpa_recomp_fwd_common<double>(
      q,
      k,
      v,
      attention_mask,
      p,
      scale,
      is_causal,
      requires_backward,
      softmax_mode,
      d_scale_q,
      d_scale_k,
      d_scale_v,
      q_scale_s,
      q_scale_o,
      d_scale_s,
      is_amax_s,
      is_amax_o,
      valid_seq_len,
      seq_padding_type,
      fwdOutType);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fp8_sdpa_recomp_bwd(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const std::optional<at::Tensor>& attention_mask,
    const at::Tensor& m,
    const at::Tensor& linv,
    const std::optional<at::Tensor>& seed,
    const bool is_causal,
    const double p,
    const double scale,
    std::string_view softmax_mode,
    const std::optional<at::Tensor>& d_scale_q,
    const std::optional<at::Tensor>& d_scale_k,
    const std::optional<at::Tensor>& d_scale_v,
    const std::optional<at::Tensor>& d_scale_s,
    const std::optional<at::Tensor>& d_scale_do,
    const std::optional<at::Tensor>& d_scale_ds,
    const std::optional<at::Tensor>& q_scale_s,
    const std::optional<at::Tensor>& q_scale_ds,
    const bool is_amax_ds,
    const at::Tensor& fwd_out) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "fp8_sdpa_recomp_bwd :",
      DUMP_22ARGS(
          grad,
          q,
          k,
          v,
          attention_mask,
          m,
          linv,
          seed,
          is_causal,
          p,
          scale,
          softmax_mode,
          d_scale_q,
          d_scale_k,
          d_scale_v,
          d_scale_s,
          d_scale_do,
          d_scale_ds,
          q_scale_s,
          q_scale_ds,
          is_amax_ds,
          fwd_out));

  habana::eager::EagerOp<
      std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>>
      hpu_op{
          "hpu::fp8_sdpa_recomp_bwd",
          {grad,           q,          k,         v,
           attention_mask, m,          linv,      seed,
           is_causal,      p,          scale,     softmax_mode,
           d_scale_q,      d_scale_k,  d_scale_v, d_scale_s,
           d_scale_do,     d_scale_ds, q_scale_s, q_scale_ds,
           is_amax_ds,     fwd_out},
          habana::Fp8SDPARecompBwdOutputShape};

  // Set grad type to BF16 for now
  auto gradType = c10::ScalarType::BFloat16;
  hpu_op.set_scalar_types(
      {gradType, // dQ
       gradType, // dK
       gradType, // dV
       c10::ScalarType::Float}); // amax_ds

  return hpu_op.call();
}
/***********************************************************************************
 * Native ops
 **********************************************************************************/

at::Tensor nms(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    double iou_threshold) {
  PT_EAGER_TRACE;
  PT_OP_INFO("nms :", DUMP_3ARGS(boxes, scores, iou_threshold));

  // max_classes set for COCO dataset for now, can be increased in future
  // based on requirement. larger max_classes => smaller max size for
  // num_boxes allowed because of memory trade-off.
  int max_classes = 81;

  auto indices = at::zeros_like(scores, torch::kInt32);
  const int64_t box_id_out_shape{scores.sizes()[0] * max_classes};
  const int64_t shape_tensor_shape{5};

  habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor>> hpu_op{
      "hpu::batched_nms_eager",
      {boxes, scores, indices, iou_threshold, max_classes},
      {{box_id_out_shape}, {shape_tensor_shape}}};

  hpu_op.set_scalar_types({torch::kLong, torch::kInt});

  auto [output_nms, shape_tensor] = hpu_op.call();
  const int64_t output_numel = shape_tensor[0].item<int64_t>();

  const auto output = output_nms.slice(0, 0, output_numel, 1);

  return output;
}

at::Tensor roi_align(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t output_h,
    int64_t output_w,
    int64_t sampling_ratio,
    bool aligned) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "roi_align :",
      DUMP_7ARGS(
          input,
          rois,
          spatial_scale,
          output_h,
          output_w,
          sampling_ratio,
          aligned));

  std::vector<int64_t> output_shape{
      rois.size(0), input.size(1), output_h, output_w};

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "torchvision::roi_align",
      {input, rois, spatial_scale, output_h, output_w, sampling_ratio, aligned},
      {{output_shape}}};
  return hpu_op.call();
}

at::Tensor roi_align_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t sampling_ratio,
    bool aligned) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "_roi_align_backward :",
      DUMP_11ARGS(
          grad,
          rois,
          spatial_scale,
          pooled_height,
          pooled_width,
          batch_size,
          channels,
          height,
          width,
          sampling_ratio,
          aligned));

  std::vector<int64_t> output_shape{batch_size, channels, height, width};

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "torchvision::_roi_align_backward",
      {grad,
       rois,
       spatial_scale,
       pooled_height,
       pooled_width,
       batch_size,
       channels,
       height,
       width,
       sampling_ratio,
       aligned},
      {{output_shape}}};
  return hpu_op.call();
}

at::Tensor dropout(const at::Tensor& input, double p, bool train) {
  PT_EAGER_TRACE;
  PT_OP_INFO("dropout :", DUMP_3ARGS(input, p, train));

  return std::get<0>(at::native_dropout(input, p, train));
}

struct DropoutFunction : public torch::autograd::Function<DropoutFunction> {
  static constexpr bool is_traceable = true;

  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor input,
      double p,
      bool train) {
    ctx->saved_data["p"] = train ? p : 0.0;
    if ((p == 0) || !train)
      return input.clone();

    at::Tensor result1, result2;
    std::tie(result1, result2) = at::native_dropout(input, p, train);
    ctx->save_for_backward({result2});

    return result1;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    auto p = ctx->saved_data["p"].toDouble();
    if (p == 0) {
      return {grad_output[0], torch::Tensor(), torch::Tensor()};
    } else if (p == 1) {
      return {grad_output[0] * 0.0, torch::Tensor(), torch::Tensor()};
    }

    torch::autograd::variable_list saved_vars = ctx->get_saved_variables();
    auto mask = saved_vars[0];
    auto scale = 1.0 / (1.0 - p);
    at::Tensor result = grad_output[0] * mask * scale;

    return {result, torch::Tensor(), torch::Tensor()};
  }
};

at::Tensor dropout_wrap(const at::Tensor& input, double p, bool train) {
  PT_EAGER_TRACE;
  PT_OP_INFO("dropout :", DUMP_3ARGS(input, p, train));

  FALLBACK_IF_UNSUPPORTED_OP(dropout, PARAMS1(input), PARAMS2(input, p, train))
  if (input.dim() > 5) {
    return dispatch_fallback<ATEN_OP(dropout)>::call(
        OpSupportLevel::Value::unsupported_rank, PARAMS2(input, p, train));
  }
  return DropoutFunction::apply(input, p, train);
}

// pytorch decomposes this op to at::_euclidean_dist in some cases
// For hpu we prefer to call _cdist_forward in all cases
at::Tensor cdist(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const double p,
    std::optional<int64_t> compute_mode) {
  PT_EAGER_TRACE;
  PT_OP_INFO("cdist :", DUMP_4ARGS(x1, x2, p, compute_mode));

  return _cdist_forward(x1, x2, p, compute_mode);
}

habana::CheckNodeWithSharedLayerValidator validator_one_hot(
    "one_hot",
    habana::OneHotSharedMeta,
    habana_helpers::HabanaExecutionMode::EAGER);

at::Tensor one_hot_forward(const at::Tensor& self, int64_t num_classes) {
  PT_EAGER_TRACE;
  PT_OP_INFO("one_hot: ", DUMP_2ARGS(self, num_classes));
  [[maybe_unused]] bool require_h2d = false;
  [[maybe_unused]] bool require_st = false;
  VAL_CUSTOM_FALLBACK_IF_UNSUPPORTED_DTYPE(one_hot, false, self, num_classes)
  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::one_hot", {self, num_classes}};
  hpu_op.SetOutputMetaFn(habana::OneHotMeta);
  return hpu_op.call();
}

void fsdp_split_with_sizes_copy(
    const at::Tensor& self,
    at::IntArrayRef split_sizes,
    int64_t dim,
    at::TensorList out) {
  auto tmp = self.split_with_sizes(split_sizes, dim);

  TORCH_CHECK(
      out.size() == tmp.size(),
      "split_with_sizes_copy_out() expected an out= argument of size ",
      tmp.size(),
      ", got size ",
      out.size());
  for (const auto i : c10::irange(out.size())) {
    if (at::native::resize_output_check(out[i], tmp[i].sizes())) {
      out[i].resize_(tmp[i].sizes());
    }
    TORCH_CHECK(
        out[i].dtype() == tmp[i].dtype(),
        "Expected out tensor to have dtype ",
        tmp[i].dtype(),
        ", but got ",
        out[i].dtype(),
        " instead");
    TORCH_CHECK(
        out[i].device() == tmp[i].device(),
        "Expected out tensor to have device ",
        tmp[i].device(),
        ", but got ",
        out[i].device(),
        " instead");
    // out[i].copy_(tmp[i]);
    // above line is replaced by below line to avoid the following run-time
    // error RuntimeError: one of the variables needed for gradient computation
    // has been modified by an inplace operation:
    const_cast<at::Tensor&>(out[i]) = tmp[i].clone();
  }
}

void fsdp_chunk_cat_out(
    at::TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    at::Tensor& out) {
  at::native::_chunk_cat_out(tensors, dim, num_chunks, out);
  return;
}

at::Tensor dequantize_nf4_impl(
    const at::Tensor& input,
    const at::Tensor& absmax,
    c10::SymInt blocksize,
    at::IntArrayRef out_shape,
    at::ScalarType out_dtype) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "dequantize_nf4: ",
      DUMP_5ARGS(input, absmax, blocksize, out_shape, out_dtype));
  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::dequantize_nf4",
      {input, absmax, blocksize, out_shape, out_dtype},
      {out_shape.vec()}};
  hpu_op.set_scalar_types({out_dtype});
  return hpu_op.call();
}

void amp_foreach_non_finite_check_and_unscale_inplace(
    at::TensorList self,
    at::Tensor& found_inf,
    const at::Tensor& inv_scale) {
  // In this op implementation we will multiply by inv_scale and if any inf
  // value is there inside a tensor we will return found_inf as [1.0] and
  // modified TensorList

  std::vector<at::Tensor> has_inf;
  for (auto& tensor : self) {
    // we can compare 1d bool tensor using logical_or
    auto temp = torch::logical_or(
        torch::any(torch::isinf(tensor)), torch::any(torch::isnan(tensor)));
    has_inf.emplace_back(temp);
    tensor.mul_(inv_scale);
  }
  c10::ArrayRef<at::Tensor> tensor_array_ref(has_inf);
  auto inf_ref = torch::any(torch::stack(tensor_array_ref), 0, true);
  found_inf.copy_(inf_ref);
  return;
}

// Refer CPU Implementation:
// https://github.com/pytorch/pytorch/blob/v2.6.0/torch/csrc/distributed/c10d/quantization/quantization.cpp#L44
at::Tensor _float_to_bfloat16_hpu(const at::Tensor& input) {
  PT_EAGER_TRACE;
  PT_OP_INFO("_FloatToBfloat16Quantized: ", DUMP_ARG(input));

  // Currently it supports 2D inputs
  HABANA_ASSERT(
      input.ndimension() == 2,
      "Tensor input must have 2 dimension(s). "
      "Found ",
      input.ndimension());

  return input.to(at::ScalarType::BFloat16).view(at::ScalarType::Half);
}

at::Tensor _bfloat16_to_float_hpu(const at::Tensor& input) {
  PT_EAGER_TRACE;
  PT_OP_INFO("_Bfloat16QuantizedToFloat: ", DUMP_ARG(input));

  // Currently it supports 2D inputs
  HABANA_ASSERT(
      input.ndimension() == 2,
      "Tensor input must have 2 dimension(s). "
      "Found ",
      input.ndimension());

  return input.view(at::ScalarType::BFloat16).to(at::ScalarType::Float);
}

} // namespace

namespace habana::eager {

TORCH_LIBRARY(hpu, m) {
  m.def(
      "hpu::batch_as_strided(Tensor[] inputs, int[][] sizes, int[][] strides, int[]? storage_offsets=None) -> Tensor[]");
  m.def("control_edge_(Tensor(a) self)-> Tensor(a)");
  m.def(
      "hpu::dequantize_nf4(Tensor input, Tensor absmax, SymInt blocksize, int[] out_shape, ScalarType out_dtype) -> Tensor");
  m.def("hpu::in_place_interleave(Tensor self) -> Tensor");
  m.def(
      "hpu::kv_reorder(Tensor self, Tensor start, Tensor end, Tensor beam_idx) -> Tensor");
  m.def(
      "hpu::optimizer_adamw(Tensor[] gradient_vec, Tensor(a!)[] weight_vec, Tensor(b!)[] exp_avg_vec, Tensor(c!)[] exp_avg_sq_vec, Tensor neg_step_t, float beta1, float beta2, float epsilon, Tensor weight_decay, bool has_weight_decay, Tensor(d!)[]? exp_avg_scales = None, Tensor(e!)[]? exp_avg_sq_scales = None) -> ()");
  m.def(
      "hpu::optimizer_ema(Tensor[] model_inputs, Tensor(a!)[] updated_ema, Tensor decay) -> ()");
  m.def(
      "hpu::optimizer_lamb_fused_norm(Tensor[] grad, float max_norm) -> Tensor");
  m.def(
      "hpu::fused_clip_norm(Tensor(a!)[] grad, Tensor max_norm, float norm_type) -> Tensor");
  m.def(
      "hpu::optimizer_lamb_phase1(Tensor[] gradients, Tensor[] weights, Tensor(a!)[] exp_avg, Tensor(b!)[] exp_avg_sq, Tensor(c!)[] out_weight_norms, Tensor(d!)[] out_adam_norms, Tensor(e!)[] out_adam_steps, Tensor clip_global_grad_norm, int grad_averaging, float beta1, float beta2, float epsilon, Tensor bias_correction1, Tensor bias_correction2, float weight_decay) -> ()");
  m.def(
      "hpu::optimizer_lamb_phase2(Tensor(a!)[] weights, Tensor[] adam_norms, Tensor[] weight_norms, Tensor[] adam_steps, Tensor neg_step, float wd, bool use_lamb) -> ()");
  m.def(
      "hpu::optimizer_lars(Tensor[] params, Tensor(a!)[] grads, int[] skip_masks, float eeta, float weight_decay, float eps, Tensor lr) -> ()");
  m.def(
      "hpu::optimizer_resource_apply_momentum(Tensor(a!)[] params_momentum_buf_list, Tensor[] dp_list, float momentum) -> ()");
  m.def(
      "hpu::optimizer_sgd(Tensor[] gradients, Tensor(a!)[] weights_in, Tensor(b!) learning_rate, float wd, float mom, float damp, bool nesterov) -> ()");
  m.def(
      "hpu::optimizer_sgd_momentum(Tensor[] gradients, Tensor(a!)[] weights_in, Tensor(b!)[] momentum_in, Tensor epoch_num, Tensor(c!) learning_rate, Tensor(d!) mom, float wd, float damp, bool nesterov) -> ()");
  m.def("hpu::repeat_ht(Tensor self, Tensor result_shape) -> Tensor");
  m.def(
      "hpu::expand_ds(Tensor(a) self, Tensor shape, *, bool implicit=False) -> Tensor(a)");
  m.def(
      "hpu::mixture_of_experts(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w1, Tensor[] w2, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max, *, bool? recomp=True) -> Tensor");
  m.def(
      "hpu::mixture_of_experts.fused_weights(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w12, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max, *, bool? recomp=True) -> Tensor");
  m.def(
      "hpu::mixture_of_experts.fp8_measurement(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w1, Tensor[] w2, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max, bool measurement_mode) -> (Tensor, Tensor)");
  m.def(
      "hpu::mixture_of_experts.fp8_measurement_fused_weights(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w12, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max, bool measurement_mode) -> (Tensor, Tensor)");
  m.def(
      "hpu::mixture_of_experts_fp8_measurement(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w1, Tensor[] w2, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max, bool measurement_mode) -> (Tensor, Tensor)");
  m.def(
      "hpu::mixture_of_experts_fp8_measurement.fused_weights(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w12, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max, bool measurement_mode) -> (Tensor, Tensor)");
  m.def(
      "hpu::mixture_of_experts_compile(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w1, Tensor[] w2, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max, *, bool? recomp=True) -> Tensor");
  m.def(
      "hpu::mixture_of_experts_compile.fused_weights(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w12, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max, *, bool? recomp=True) -> Tensor");
  m.def(
      "hpu::mixture_of_experts_fwd(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w1, Tensor[] w2, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max) -> Tensor[]");
  m.def(
      "hpu::mixture_of_experts_recomp_fwd(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w1, Tensor[] w2, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max) -> Tensor");
  m.def(
      "hpu::mixture_of_experts_fwd.fused_weights(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w12, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max) -> Tensor[]");
  m.def(
      "hpu::mixture_of_experts_recomp_fwd.fused_weights(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w12, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max) -> Tensor");
  m.def(
      "hpu::mixture_of_experts_bwd(Tensor grad_tokens_in, Tensor chunks_input, Tensor token_to_chunk, Tensor token_in_chunk, Tensor chunks_routing_table, Tensor chunks_routing_weights, Tensor gemm1_out, Tensor gemm2_out, Tensor activation_out, Tensor mult_out, Tensor mlp_out, Tensor[] w1, Tensor[] w2, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max, *, int[] router_weights_size) -> Tensor[]");
  m.def(
      "hpu::mixture_of_experts_recomp_bwd(Tensor grad_tokens_in, Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w1, Tensor[] w2, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max) -> Tensor[]");
  m.def(
      "hpu::mixture_of_experts_bwd.fused_weights(Tensor grad_tokens_in, Tensor chunks_input, Tensor token_to_chunk, Tensor token_in_chunk, Tensor chunks_routing_table, Tensor chunks_routing_weights, Tensor gemm12_out, Tensor activation_out, Tensor mult_out, Tensor mlp_out, Tensor[] w12, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max, *, int[] router_weights_size) -> Tensor[]");
  m.def(
      "hpu::mixture_of_experts_recomp_bwd.fused_weights(Tensor grad_tokens_in, Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w12, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max) -> Tensor[]");
  m.def("hpu::view(Tensor input, Tensor shape) -> Tensor");
  m.def("hpu::view_neg(Tensor input, Tensor shape, int[] shape) -> Tensor");
  m.def("hpu::slice_ht(Tensor input, Tensor shape, Tensor shape) -> Tensor");
  m.def(
      "hpu::slice_ds(Tensor input, SymInt dim, SymInt start, SymInt end, SymInt step, SymInt[]? size=None) -> Tensor");
  m.def(
      "strided_insert_orig_ds(Tensor self, Tensor other, Tensor stride, Tensor offset) -> (Tensor)");
  m.def(
      "strided_insert_orig_ds_h2d(Tensor self, Tensor other, Tensor stride) -> (Tensor)");
  m.def(
      "strided_view_ds_h2d(Tensor self, Tensor size, Tensor stride, Tensor offset) -> (Tensor)");
  m.def(
      "hpu::select_scatter(Tensor self, Tensor src, Tensor dim, Tensor index) -> (Tensor)");
  m.def(
      "hpu::slice_scatter(Tensor self, Tensor src, Tensor dim = None, Tensor? start = None, Tensor? end = None, Tensor step = None) -> (Tensor)");
  m.def(
      "hpu::slice_scatter_ds(Tensor self, Tensor src, Tensor step = None, Tensor start = None) -> (Tensor)");
  m.def(
      "hpu::as_strided_scatter(Tensor self, Tensor src, Tensor stride, Tensor? storage_offset = None) -> (Tensor)");
  m.def(
      "hpu::as_strided_scatter_orig(Tensor self, Tensor src, Tensor stride) -> (Tensor)");
  m.def(
      "strided_view_orig_ds_h2d(Tensor self, Tensor size, Tensor stride) -> (Tensor)");
  m.def(
      "hpu::sdpa_recomp_fwd(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::sdpa_recomp_fwd_dropout(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor, Tensor)");
  // for torch.compile to insert seed
  m.def(
      "hpu::sdpa_recomp_fwd_non_dropout(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_modem, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::sdpa_recomp_fwd_dropout_seed(Tensor seed, Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::sdpa_fwd(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, str softmax_mode, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::sdpa_fwd_dropout(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, str softmax_mode, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::sdpa_fwd_non_dropout(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, str softmax_mode, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::sdpa_fwd_dropout_seed(Tensor seed, Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, str softmax_mode, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_fwd(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? q_scale_s, Tensor? q_scale_o, Tensor? d_scale_s, bool is_amax_s, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_fwd_dropout(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? q_scale_s, Tensor? q_scale_o, Tensor? d_scale_s, bool is_amax_s, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_fwd_non_dropout(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? q_scale_s, Tensor? q_scale_o, Tensor? d_scale_s, bool is_amax_s, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_fwd_dropout_seed(Tensor seed, Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? q_scale_s, Tensor? q_scale_o, Tensor? d_scale_s, bool is_amax_s, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? q_scale_s, Tensor? q_scale_o, Tensor? d_scale_s, bool is_amax_s, bool is_amax_0, Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd_non_dropout(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? q_scale_s, Tensor? q_scale_o, Tensor? d_scale_s, bool is_amax_s, bool is_amax_0, Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd_dropout(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? q_scale_s, Tensor? q_scale_o, Tensor? d_scale_s, bool is_amax_s, bool is_amax_0, Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd_dropout_seed(Tensor seed, Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? q_scale_s, Tensor? q_scale_o, Tensor? d_scale_s, bool is_amax_s, bool is_amax_o, Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd.scalar(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, float d_scale_q, float d_scale_k, float d_scale_v, float q_scale_s, float q_scale_o, float d_scale_s, bool is_amax_s, bool is_amax_0, Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd_non_dropout.scalar(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, float d_scale_q, float d_scale_k, float d_scale_v, float q_scale_s, float q_scale_o, float d_scale_s, bool is_amax_s, bool is_amax_0,  Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd_dropout.scalar(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, float d_scale_q, float d_scale_k, float d_scale_v, float q_scale_s, float q_scale_o, float d_scale_s, bool is_amax_s, bool is_amax_0,  Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd_dropout_seed.scalar(Tensor seed, Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, float d_scale_q, float d_scale_k, float d_scale_v, float q_scale_s, float q_scale_o, float d_scale_s, bool is_amax_s, bool is_amax_o,  Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  m.def(
      "hpu::fp8_sdpa_recomp_bwd(Tensor grad, Tensor q, Tensor k, Tensor v, Tensor? attention_mask, Tensor m, Tensor linv, Tensor ? seed, bool is_causal, float p, float scale, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? d_scale_s, Tensor? d_scale_do, Tensor? d_scale_ds, Tensor? q_scale_s, Tensor? q_scale_ds, bool is_amax_ds, Tensor fwd_out) -> (Tensor, Tensor, Tensor, Tensor)");

  m.def("hpu::accumulate_grads_(Tensor[] variables, Tensor[] new_grads) -> ()");
  m.def("hpu::custom_foreach_add_(Tensor(a!)[] self, Tensor[] other) -> ()");
  m.def(
      "hpu::batched_nms_eager(Tensor boxes, Tensor scores, Tensor indexes, float iou_threshold, int max_classes) -> (Tensor, Tensor)");
  m.def(
      "hpu::habana_randperm_ht(Tensor seed, Tensor h2d_tensor, Tensor shape_tensor, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor");
  m.def(
      "hpu::habana_rand_st(Tensor seed, Tensor shape_tensor, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor");
  m.def(
      "hpu::habana_randn_st(Tensor seed, Tensor shape_tensor, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor");
  m.def(
      "hpu::habana_randint_st(Tensor seed, SymInt low, SymInt high, Tensor shape_tensor, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor");
  m.def(
      "hpu::full_ds(Tensor size, Scalar fill_value, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor");
  m.def(
      "hpu::empty_ds(Tensor size,  ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor");
  m.def(
      "hpu::constant_pad_nd(Tensor input, Tensor pad_tensor, Tensor output_shape_tensor, Scalar value) -> Tensor");
  m.def(
      "hpu::constant_pad_nd_ds(Tensor input, SymInt[] pad, Scalar value, SymInt[]? size=None) -> Tensor");
  m.def("hpu::weight_permutation(Tensor input) -> Tensor");
  m.def(
      "hpu::custom_bernoulli.Size(SymInt[] size, float p, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor");
  m.def(
      "hpu::habana_seed_generator(Tensor seed, Tensor counter, int size) -> Tensor");
  HABANA_RANDOM_DEF_CHECKPOINT(bernoulli, "Tensor seed, Tensor self")
  HABANA_RANDOM_DEF_CHECKPOINT(poisson, "Tensor seed, Tensor self")
  HABANA_RANDOM_DEF_CHECKPOINT(
      rand,
      "Tensor seed, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None")
  HABANA_RANDOM_DEF_CHECKPOINT(
      randn,
      "Tensor seed, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None")
  HABANA_RANDOM_DEF_CHECKPOINT(
      randint,
      "Tensor seed, SymInt low, SymInt high, SymInt[] size, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None")
  HABANA_RANDOM_DEF_CHECKPOINT(
      multinomial,
      "Tensor seed, Tensor self, int num_samples, bool replacement=False")
  HABANA_RANDOM_DEF_CHECKPOINT(
      uniform, "Tensor seed, Tensor self, float from=0, float to=1")
  HABANA_RANDOM_DEF_CHECKPOINT(
      randperm,
      "Tensor seed, SymInt n, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None")
  HABANA_RANDOM_DEF_CHECKPOINT_2_OUTS(
      native_dropout, "Tensor seed, Tensor input, float p, bool? train")
  m.def("hpu::one_hot(Tensor self, int num_classes=-1) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu, HPU, m) {
  m.impl("hpu::accumulate_grads_", accumulate_grads_);
  m.impl("hpu::in_place_interleave", in_place_interleave);
  m.impl("hpu::kv_reorder", kv_reorder);
  m.impl("hpu::optimizer_adamw", optimizer_adamw);
  m.impl("hpu::optimizer_ema", optimizer_ema);
  m.impl("hpu::optimizer_lamb_fused_norm", optimizer_lamb_norm);
  m.impl("hpu::optimizer_lamb_phase1", optimizer_lamb_phase1);
  m.impl("hpu::optimizer_lamb_phase2", optimizer_lamb_phase2);
  m.impl("hpu::optimizer_lars", optimizer_lars);
  m.impl(
      "hpu::optimizer_resource_apply_momentum",
      optimizer_resource_apply_momentum);
  m.impl("hpu::mixture_of_experts", mixture_of_experts);
  m.impl(
      "hpu::mixture_of_experts.fused_weights",
      mixture_of_experts_fused_weights);
  m.impl(
      "hpu::mixture_of_experts.fp8_measurement",
      mixture_of_experts_fp8_measurement);
  m.impl(
      "hpu::mixture_of_experts.fp8_measurement_fused_weights",
      mixture_of_experts_fp8_measurement_fused_weights);
  m.impl("hpu::mixture_of_experts.fp8", mixture_of_experts_fp8);
  m.impl(
      "hpu::mixture_of_experts.fp8_fused_weights",
      mixture_of_experts_fp8_fused_weights);
  m.impl("hpu::mixture_of_experts.fp8_scalars", mixture_of_experts_fp8_scalars);
  m.impl(
      "hpu::mixture_of_experts.fp8_fused_weights_scalars",
      mixture_of_experts_fp8_fused_weights_scalars);
  m.impl("hpu::mixture_of_experts.fp8_dynamic", mixture_of_experts_fp8_dynamic);
  m.impl(
      "hpu::mixture_of_experts.fp8_fused_weights_dynamic",
      mixture_of_experts_fp8_fused_weights_dynamic);
  m.impl(
      "hpu::mixture_of_experts.fp8_scalars_dynamic",
      mixture_of_experts_fp8_scalars_dynamic);
  m.impl(
      "hpu::mixture_of_experts.fp8_fused_weights_scalars_dynamic",
      mixture_of_experts_fp8_fused_weights_scalars_dynamic);
  m.impl(
      "hpu::mixture_of_experts.fp8_blockwise",
      mixture_of_experts_fp8_blockwise);
  m.impl(
      "hpu::mixture_of_experts.fp8_fused_weights_blockwise",
      mixture_of_experts_fp8_fused_weights_blockwise);
  m.impl("hpu::optimizer_sgd", optimizer_sgd);
  m.impl("hpu::optimizer_sgd_momentum", optimizer_sgd_momentum);
  m.impl("hpu::sdpa_recomp_fwd", sdpa_recomp_fwd);
  m.impl("hpu::sdpa_recomp_fwd_non_dropout", sdpa_recomp_fwd);
  m.impl("hpu::sdpa_fwd", sdpa_fwd);
  m.impl("hpu::sdpa_fwd_non_dropout", sdpa_fwd);
  m.impl("hpu::sdpa_fwd_dropout", sdpa_fwd);
  m.impl("hpu::fp8_sdpa_fwd", fp8_sdpa_fwd);
  m.impl("hpu::fp8_sdpa_fwd_non_dropout", fp8_sdpa_fwd);
  m.impl("hpu::fp8_sdpa_fwd_dropout", fp8_sdpa_fwd);
  m.impl("hpu::fp8_sdpa_recomp_fwd", fp8_sdpa_recomp_fwd);
  m.impl("hpu::fp8_sdpa_recomp_fwd_non_dropout", fp8_sdpa_recomp_fwd);
  m.impl("hpu::fp8_sdpa_recomp_fwd_dropout", fp8_sdpa_recomp_fwd);
  m.impl("hpu::fp8_sdpa_recomp_fwd.scalar", fp8_sdpa_recomp_scalar_fwd);
  m.impl(
      "hpu::fp8_sdpa_recomp_fwd_non_dropout.scalar",
      fp8_sdpa_recomp_scalar_fwd);
  m.impl("hpu::fp8_sdpa_recomp_fwd_dropout.scalar", fp8_sdpa_recomp_scalar_fwd);
  m.impl("hpu::fp8_sdpa_recomp_bwd", fp8_sdpa_recomp_bwd);
  m.impl("hpu::slice_ds", slice_ds);
  m.impl("hpu::constant_pad_nd_ds", constant_pad_nd_ds);
  m.impl("hpu::fused_clip_norm", fused_clip_norm);
  m.impl("hpu::weight_permutation", weight_permutation);
  m.impl("hpu::one_hot", one_hot_forward);
  m.impl("hpu::dequantize_nf4", dequantize_nf4_impl);
}

TORCH_LIBRARY_IMPL(aten, HPU, m) {
  m.impl("cdist", cdist);
  m.impl("dropout", dropout);
  m.impl("one_hot", one_hot_forward);
  m.impl(
      "_amp_foreach_non_finite_check_and_unscale_",
      amp_foreach_non_finite_check_and_unscale_inplace);
}

TORCH_LIBRARY_IMPL(aten, AutogradHPU, m) {
  m.impl("dropout", dropout_wrap);
}

TORCH_LIBRARY_IMPL(fsdp, HPU, m) {
  m.impl("split_with_sizes_copy", fsdp_split_with_sizes_copy);
  m.impl("chunk_cat", fsdp_chunk_cat_out);
}

TORCH_LIBRARY_IMPL(torchvision, HPU, m) {
  m.impl("roi_align", roi_align);
  m.impl("_roi_align_backward", roi_align_backward);
  m.impl("nms", nms);
}

TORCH_LIBRARY_IMPL(hpu, Autograd, m) {
  m.impl("hpu::mixture_of_experts_compile", mixture_of_experts_fwd_autograd);
  m.impl(
      "hpu::mixture_of_experts_compile.fused_weights",
      mixture_of_experts_fwd_fused_weights_autograd);
  m.impl("hpu::mixture_of_experts", mixture_of_experts);
  m.impl(
      "hpu::mixture_of_experts.fused_weights",
      mixture_of_experts_fused_weights);
}

TORCH_LIBRARY_IMPL(quantization, HPU, m) {
  m.impl("_FloatToBfloat16Quantized", _float_to_bfloat16_hpu);
  m.impl("_Bfloat16QuantizedToFloat", _bfloat16_to_float_hpu);
}

} // namespace habana::eager

namespace {
// Inplace ops must be additionally registered to Functionalize backend
// to be handled in torch.compile
// https://gist.github.com/bdhirsh/7dadbf6296f8f7d1abcf4c482f438aaa
template <class T>
T get_functional_tensor(const T& tensor) {
  if (at::functionalization::impl::isFunctionalTensor(tensor)) {
    at::functionalization::impl::sync(tensor);
    return at::functionalization::impl::from_functional_tensor(tensor);
  } else {
    return tensor;
  }
}

at::Tensor& kv_reorder_functionalization_glue(
    at::Tensor& self,
    const at::Tensor& start,
    const at::Tensor& end,
    const at::Tensor& beam_idx) {
  auto self_ = get_functional_tensor(self);
  auto start_ = get_functional_tensor(start);
  auto end_ = get_functional_tensor(end);
  auto beam_idx_ = get_functional_tensor(beam_idx);

  static auto op_handle = c10::Dispatcher::singleton()
                              .findSchemaOrThrow("hpu::kv_reorder", "")
                              .typed<at::Tensor(
                                  const at::Tensor&,
                                  const at::Tensor&,
                                  const at::Tensor&,
                                  const at::Tensor&)>();

  at::Tensor tmp_output;
  {
    at::AutoDispatchSkipFunctionalize guard;
    tmp_output = op_handle.call(self_, start_, end_, beam_idx_);
  }

  at::functionalization::impl::replace_(self, tmp_output);
  at::functionalization::impl::commit_update(self);
  at::functionalization::impl::sync(self);
  return self;
}

at::Tensor& in_place_interleave_functionalization_glue(at::Tensor& self) {
  auto self_ = get_functional_tensor(self);

  static auto op_handle = c10::Dispatcher::singleton()
                              .findSchemaOrThrow("hpu::in_place_interleave", "")
                              .typed<at::Tensor(const at::Tensor&)>();

  at::Tensor tmp_output;
  {
    at::AutoDispatchSkipFunctionalize guard;
    tmp_output = op_handle.call(self_);
  }

  at::functionalization::impl::replace_(self, tmp_output);
  at::functionalization::impl::commit_update(self);
  at::functionalization::impl::sync(self);
  return self;
}

at::Tensor& fp8_gemm_functionalization_glue(
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    const at::Tensor& D,
    at::ScalarType out_dtype,
    const std::optional<at::Tensor>& A_scale_inv,
    const std::optional<at::Tensor>& B_scale_inv,
    const std::optional<at::Tensor>& bias,
    bool accumulate,
    at::Tensor& out) {
  auto A_ = get_functional_tensor(A);
  auto B_ = get_functional_tensor(B);
  auto D_ = get_functional_tensor(D);
  auto A_scale_inv_ = get_functional_tensor(A_scale_inv);
  auto B_scale_inv_ = get_functional_tensor(B_scale_inv);
  auto bias_ = get_functional_tensor(bias);
  auto out_ = get_functional_tensor(out);

  static auto op_handle = c10::Dispatcher::singleton()
                              .findSchemaOrThrow("hpu::fp8_gemm_v2", "")
                              .typed<at::Tensor(
                                  const at::Tensor&,
                                  bool,
                                  const at::Tensor&,
                                  bool,
                                  const std::optional<at::Tensor>&,
                                  at::ScalarType,
                                  const std::optional<at::Tensor>&,
                                  const std::optional<at::Tensor>&,
                                  const std::optional<at::Tensor>&,
                                  bool,
                                  at::OptionalIntArrayRef)>();

  at::Tensor tmp_output;
  {
    at::AutoDispatchSkipFunctionalize guard;
    tmp_output = op_handle.call(
        A_,
        trans_A,
        B_,
        trans_B,
        D_,
        out_dtype,
        A_scale_inv_,
        B_scale_inv_,
        bias_,
        accumulate,
        std::nullopt);
  }

  at::functionalization::impl::replace_(out, tmp_output);
  at::functionalization::impl::commit_update(out);
  at::functionalization::impl::sync(out);
  return out;
}
} // namespace

namespace habana::eager {

TORCH_LIBRARY_IMPL(hpu, Functionalize, m) {
  m.impl("kv_reorder_", kv_reorder_functionalization_glue);
  m.impl("in_place_interleave_", in_place_interleave_functionalization_glue);
  m.impl("fp8_gemm", fp8_gemm_functionalization_glue);
}

} // namespace habana::eager
