/**
 * Copyright (c) 2025 Intel Corporation
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

#include "fused_sdpa.h"
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include "backend/random.h"
#include "common/dump_args.h"
#include "common/warning_suppress.h"
#include "generated/backend/sdpa_bwd.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_helpers/logging.h"
#include "hpu_ops/op_logger.h"
#include "hpu_ops/sdpa_gen.h"

namespace habana::eager {

int64_t fused_sdp_choice_hpu(
    [[maybe_unused]] const at::Tensor& query,
    [[maybe_unused]] const at::Tensor& key,
    [[maybe_unused]] const at::Tensor& value,
    [[maybe_unused]] const ::std::optional<at::Tensor>& attn_mask,
    [[maybe_unused]] double dropout_p,
    [[maybe_unused]] bool is_causal,
    [[maybe_unused]] ::std::optional<double> scale,
    [[maybe_unused]] bool enable_gqa) {
  // there are five availble SDPBackend (math, flash_attention,
  // efficient_attention, cudnn_attention, overrideable)
  return static_cast<int64_t>(sdp::SDPBackend::math);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> dispatch_sdpa_fwd_wrap(
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
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("hpu::sdpa_fwd", "")
                       .typed<decltype(dispatch_sdpa_fwd_wrap)>();
  return op.call(
      q,
      k,
      v,
      attention_mask,
      p,
      scale,
      is_causal,
      softmax_mode,
      valid_seq_len,
      seq_padding_type);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> dispatch_sdpa_bwd_wrap(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& P,
    const std::optional<at::Tensor>& dm,
    const bool is_causal,
    const double p,
    const double scale,
    const at::Tensor& fwd_out) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "sdpa_bwd :",
      DUMP_10ARGS(grad, q, k, v, P, dm, is_causal, p, scale, fwd_out));

  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("hpu::sdpa_bwd", "")
                       .typed<decltype(dispatch_sdpa_bwd_wrap)>();
  return op.call(grad, q, k, v, P, dm, is_causal, p, scale, fwd_out);
}

at::Tensor gqa_output_reshape(at::Tensor input_tensor) {
  auto size = input_tensor.sizes().vec();
  return input_tensor.reshape({size[0], size[1] * size[2], size[3], size[4]});
}

std::vector<at::Tensor> gqa_input_reshape_fwd(
    at::Tensor& query,
    at::Tensor& key,
    at::Tensor& value,
    ::std::optional<at::Tensor>& attn_mask) {
  auto q_size = query.sizes().vec();
  auto k_size = key.sizes().vec();
  auto v_size = value.sizes().vec();

  auto q_heads = q_size[1];
  auto kv_heads = k_size[1];

  auto q_heads_per_group = q_heads / kv_heads;
  auto groups = kv_heads;

  query = query.reshape(
      {q_size[0], groups, q_heads_per_group, q_size[2], q_size[3]});
  key = key.reshape({k_size[0], groups, 1, k_size[2], k_size[3]});
  value = value.reshape({v_size[0], groups, 1, v_size[2], v_size[3]});

  if (attn_mask.has_value()) {
    auto a_size = attn_mask.value().sizes().vec();
    if (q_heads ==
        a_size[1]) { // attention mask shape = [batch size, q_heads, *, *]
      attn_mask = attn_mask.value().reshape(
          {a_size[0], groups, q_heads_per_group, a_size[2], a_size[3]});
    } else { // attention mask shape = [batch size, 1, *, *]
      attn_mask = attn_mask.value().unsqueeze(1);
    }
  }
  std::vector out{query, key, value};
  out.push_back(attn_mask.has_value() ? attn_mask.value() : torch::Tensor());
  return out;
}

at::Tensor gqa_input_reshape_bwd(
    at::Tensor& query,
    at::Tensor& value,
    at::Tensor grad) {
  auto q_size = query.sizes().vec();
  auto v_size = value.sizes().vec();
  return grad.reshape(
      {q_size[0], q_size[1], q_size[2], q_size[3], v_size.back()});
}

class FusedSDPAAutogradHPU
    : public torch::autograd::Function<FusedSDPAAutogradHPU> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& query,
      const at::Tensor& key,
      const at::Tensor& value,
      const ::std::optional<at::Tensor>& attn_mask,
      double dropout_p,
      bool is_causal,
      ::std::optional<double> scale,
      bool enable_gqa) {
    PT_EAGER_TRACE;
    auto softmax_mode = "None";
    auto seq_padding_type = "left";
    double scale_;
    if (!scale.has_value())
      scale_ =
          (query.sizes().size() > 0) ? (1. / sqrt(query.sizes().back())) : 1.;
    else
      scale_ = scale.value();
    auto valid_seq_len = std::optional<at::Tensor>();
    ctx->saved_data["dropout_p"] = dropout_p;
    ctx->saved_data["scale"] = scale_;
    ctx->saved_data["is_causal"] = is_causal;
    ctx->saved_data["enable_gqa"] = enable_gqa;
    const bool has_attn_mask =
        attn_mask.has_value() && attn_mask.value().defined();
    bool mask_requires_grad = false;
    if (has_attn_mask) {
      mask_requires_grad = attn_mask.value().requires_grad();
    }
    ctx->saved_data["mask_requires_grad"] = mask_requires_grad;

    at::Tensor query_n = query;
    at::Tensor key_n = key;
    at::Tensor value_n = value;
    ::std::optional<at::Tensor> attn_mask_n = attn_mask;
    if (enable_gqa) {
      auto gqa_out =
          gqa_input_reshape_fwd(query_n, key_n, value_n, attn_mask_n);
      query_n = gqa_out[0];
      key_n = gqa_out[1];
      value_n = gqa_out[2];
      attn_mask_n = has_attn_mask ? gqa_out[3] : attn_mask_n;
    }

    // output (out, P, dm)
    auto output = dispatch_sdpa_fwd_wrap(
        query_n,
        key_n,
        value_n,
        attn_mask_n,
        dropout_p,
        scale_,
        is_causal,
        softmax_mode,
        valid_seq_len,
        seq_padding_type);
    auto out = std::get<0>(output);
    auto P = std::get<1>(output);
    auto dm = std::get<2>(output);
    if (enable_gqa) {
      out = gqa_output_reshape(out);
    }
    ctx->save_for_backward(
        {query_n,
         key_n,
         value_n,
         mask_requires_grad ? attn_mask_n.value() : torch::Tensor(),
         P,
         dm,
         out});
    return out;
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    PT_EAGER_TRACE;
    torch::autograd::variable_list saved_vars = ctx->get_saved_variables();
    auto grad_out = grad_output[0];
    auto query = saved_vars[0];
    auto key = saved_vars[1];
    auto value = saved_vars[2];
    auto attn_mask = saved_vars[3];
    auto P = saved_vars[4];
    auto dm = saved_vars[5];
    auto fwd_out = saved_vars[6];
    auto enable_gqa = ctx->saved_data["enable_gqa"].toBool();
    auto mask_requires_grad = ctx->saved_data["mask_requires_grad"].toBool();

    if (enable_gqa) {
      grad_out = gqa_input_reshape_bwd(query, value, grad_out);
      fwd_out = gqa_input_reshape_bwd(query, value, fwd_out);
    }

    auto output = dispatch_sdpa_bwd_wrap(
        grad_out,
        query,
        key,
        value,
        P,
        dm,
        ctx->saved_data["is_causal"].toBool(),
        ctx->saved_data["dropout_p"].toScalar().toFloat(),
        ctx->saved_data["scale"].toScalar().toFloat(),
        fwd_out);
    auto query_grad = std::get<0>(output);
    auto key_grad = std::get<1>(output);
    auto value_grad = std::get<2>(output);
    if (enable_gqa) {
      query_grad = gqa_output_reshape(query_grad);
      key_grad = gqa_output_reshape(key_grad);
      value_grad = gqa_output_reshape(value_grad);
    }

    return {
        query_grad,
        key_grad,
        value_grad,
        // if attn_mask requires grad, pass it to backward as it is
        // because attn_mask grad computation is not supported yet in hpu
        mask_requires_grad ? attn_mask : torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor()};
  }
};

at::Tensor fused_sdpa_autograd_wrap(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const ::std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    ::std::optional<double> scale,
    bool enable_gqa) {
  PT_EAGER_TRACE;
  return FusedSDPAAutogradHPU::apply(
      query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
}

// When below flag is enabled, aten.scaled_dot_product_attention is used
// math backend in torch.compile and eager mode
// When this flag is not enabled then aten.scaled_dot_product_attention
// will be overwritten by default in torch.compile and eager mode
static const bool ATEN_FSDPA = GET_ENV_FLAG_NEW(PT_HPU_USE_ATEN_SDPA);

TORCH_LIBRARY_IMPL(aten, AutogradHPU, m) {
  if (!ATEN_FSDPA) {
    m.impl("scaled_dot_product_attention", fused_sdpa_autograd_wrap);
  }
}

TORCH_LIBRARY_IMPL(aten, HPU, m) {
  if (ATEN_FSDPA) {
    m.impl("_fused_sdp_choice", fused_sdp_choice_hpu);
  }
}

} // namespace habana::eager
