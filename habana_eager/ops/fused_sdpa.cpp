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
  // This function defines the priority order of the different sdp backends
  // 1. Math fallback (default)
  // 2. Overrideable backend (when explicitly enabled)
  auto& ctx = at::globalContext();

  if (!ctx.userEnabledMathSDP() && !ctx.userEnabledOverrideableSDP()) {
    return static_cast<int64_t>(sdp::SDPBackend::error);
  }

  const std::array<sdp::SDPBackend, 2> priority_order{
      sdp::SDPBackend::math,
      sdp::SDPBackend::overrideable,
  };

  for (auto& backend : priority_order) {
    switch (backend) {
      case sdp::SDPBackend::overrideable:
        if (ctx.userEnabledOverrideableSDP()) {
          PT_OP_DEBUG(
              "fused_sdp_choice_hpu: Choosing the overrideable SDPA kernel backend");
          return static_cast<int64_t>(sdp::SDPBackend::overrideable);
        }
        break;
      case sdp::SDPBackend::math:
        if (ctx.userEnabledMathSDP()) {
          PT_OP_DEBUG(
              "fused_sdp_choice_hpu: Choosing the math backend (default)");
          return static_cast<int64_t>(sdp::SDPBackend::math);
        }
        break;
      default:
        TORCH_CHECK(false, "Invalid backend");
    }
  }

  // If we have gotten to this point then two things have happened:
  // 1. use_overrideable_hpu did not satisfy the constraints to be ran
  // 2. The user has explicitly disabled the math kernel
  // We then re-run the kernel checks with debug enabled to print out the
  // reason why the kernel was not selected
  return static_cast<int64_t>(sdp::SDPBackend::error);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
dispatch_sdpa_recomp_fwd_wrap(
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
    std::string_view seq_padding_type,
    c10::SymIntArrayRef window_size,
    const std::optional<at::Tensor>& sink) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "sdpa_recomp_fwd :",
      DUMP_12ARGS(
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
          seq_padding_type,
          window_size));
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("hpu::sdpa_recomp_fwd", "")
                       .typed<decltype(dispatch_sdpa_recomp_fwd_wrap)>();
  return op.call(
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
      seq_padding_type,
      window_size,
      sink);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> dispatch_sdpa_recomp_bwd_wrap(
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
    const at::Tensor& fwd_out) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "sdpa_recomp_bwd :",
      DUMP_12ARGS(
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
          softmax_mode));

  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("hpu::sdpa_recomp_bwd", "")
                       .typed<decltype(dispatch_sdpa_recomp_bwd_wrap)>();

  return op.call(
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
      fwd_out);
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
    if (scale.has_value())
      scale_ = scale.value();
    else
      scale_ = !query.sizes().empty() ? (1. / sqrt(query.sizes().back())) : 1.;
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

    // output (out, m, linv, seed)
    auto output = dispatch_sdpa_recomp_fwd_wrap(
        query_n,
        key_n,
        value_n,
        attn_mask_n,
        dropout_p,
        scale_,
        is_causal,
        true, // requires_backward = true
        softmax_mode,
        valid_seq_len,
        seq_padding_type);
    auto out = std::get<0>(output);
    auto m = std::get<1>(output);
    auto linv = std::get<2>(output);
    auto seed = std::get<3>(output);
    if (enable_gqa) {
      out = gqa_output_reshape(out);
    }
    ctx->save_for_backward(
        {query_n,
         key_n,
         value_n,
         mask_requires_grad ? attn_mask_n.value() : torch::Tensor(),
         m,
         linv,
         seed,
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
    auto m = saved_vars[4];
    auto linv = saved_vars[5];
    auto seed = saved_vars[6];
    auto fwd_out = saved_vars[7];
    auto enable_gqa = ctx->saved_data["enable_gqa"].toBool();
    auto mask_requires_grad = ctx->saved_data["mask_requires_grad"].toBool();

    if (enable_gqa) {
      grad_out = gqa_input_reshape_bwd(query, value, grad_out);
      fwd_out = gqa_input_reshape_bwd(query, value, fwd_out);
    }

    auto output = dispatch_sdpa_recomp_bwd_wrap(
        grad_out,
        query,
        key,
        value,
        mask_requires_grad ? std::make_optional(attn_mask) : std::nullopt,
        m,
        linv,
        seed.defined() ? std::make_optional(seed) : std::nullopt,
        ctx->saved_data["is_causal"].toBool(),
        ctx->saved_data["dropout_p"].toScalar().toFloat(),
        ctx->saved_data["scale"].toScalar().toFloat(),
        "None", // softmax_mode - using default
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

class FusedSDPAOverrideableAutogradHPU
    : public torch::autograd::Function<FusedSDPAOverrideableAutogradHPU> {
 public:
  static std::vector<at::Tensor> forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& query,
      const at::Tensor& key,
      const at::Tensor& value,
      const std::optional<at::Tensor>& attn_bias,
      double dropout_p,
      bool is_causal,
      bool return_debug_mask,
      std::optional<double> scale) {
    PT_EAGER_TRACE;

    // Calculate scale if not provided
    double scale_ =
        scale.has_value() ? scale.value() : (1.0 / std::sqrt(query.size(-1)));

    // Save parameters for backward
    ctx->saved_data["dropout_p"] = dropout_p;
    ctx->saved_data["scale"] = scale_;
    ctx->saved_data["is_causal"] = is_causal;
    ctx->saved_data["return_debug_mask"] = return_debug_mask;

    const bool has_attn_bias =
        attn_bias.has_value() && attn_bias.value().defined();
    bool bias_requires_grad = false;
    if (has_attn_bias) {
      bias_requires_grad = attn_bias.value().requires_grad();
    }
    ctx->saved_data["bias_requires_grad"] = bias_requires_grad;

    // Use only dispatch_sdpa_recomp_fwd_wrap for forward computation
    auto softmax_mode = "None";
    auto seq_padding_type = "left";
    auto valid_seq_len = std::optional<at::Tensor>();

    // Call recompute forward to get output, m, linv, seed
    auto recomp_output = dispatch_sdpa_recomp_fwd_wrap(
        query,
        key,
        value,
        attn_bias,
        dropout_p,
        scale_,
        is_causal,
        true, // requires_backward = true
        softmax_mode,
        valid_seq_len,
        seq_padding_type);

    auto out = std::get<0>(recomp_output); // out tensor
    auto m = std::get<1>(recomp_output); // m tensor
    auto linv = std::get<2>(recomp_output); // linv tensor
    auto seed = std::get<3>(recomp_output); // seed tensor

    // Compute logsumexp from m and linv: logsumexp = m - log(linv)
    auto logsumexp = m.squeeze(-1) - torch::log(linv).squeeze(-1);

    // Create placeholder tensors for compatibility with overrideable interface
    auto cum_seq_q = torch::empty({0}, query.options().dtype(torch::kInt32));
    auto cum_seq_k = torch::empty({0}, key.options().dtype(torch::kInt32));
    auto philox_seed = torch::empty({0}, query.options().dtype(torch::kInt64));
    auto philox_offset =
        torch::empty({0}, query.options().dtype(torch::kInt64));
    auto debug_attn_mask = return_debug_mask
        ? torch::empty_like(query.select(2, 0).select(2, 0))
        : torch::empty({0}, query.options());

    ctx->save_for_backward({
        query,
        key,
        value,
        bias_requires_grad ? attn_bias.value() : torch::Tensor(),
        out,
        m, // m tensor from recompute
        linv, // linv tensor from recompute
        seed // seed tensor from recompute
    });

    // Return tensor outputs for autograd
    return {
        out, // out
        logsumexp, // logsumexp computed from m + log(linv)
        cum_seq_q, // cum_seq_q (placeholder)
        cum_seq_k, // cum_seq_k (placeholder)
        philox_seed, // philox_seed (placeholder)
        philox_offset, // philox_offset (placeholder)
        debug_attn_mask // debug_attn_mask (placeholder)
    };
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    PT_EAGER_TRACE;

    torch::autograd::variable_list saved_vars = ctx->get_saved_variables();
    auto grad_out = grad_output[0]; // Only the main output has gradients
    auto query = saved_vars[0];
    auto key = saved_vars[1];
    auto value = saved_vars[2];
    auto attn_bias = saved_vars[3];
    auto out = saved_vars[4];
    auto m = saved_vars[5]; // m tensor from recompute
    auto linv = saved_vars[6]; // linv tensor from recompute
    auto seed = saved_vars[7]; // seed tensor from recompute

    auto bias_requires_grad = ctx->saved_data["bias_requires_grad"].toBool();

    // Call the recompute backward function using m, linv, seed
    auto grad_result = dispatch_sdpa_recomp_bwd_wrap(
        grad_out,
        query,
        key,
        value,
        bias_requires_grad ? std::make_optional(attn_bias) : std::nullopt,
        m,
        linv,
        seed.defined() ? std::make_optional(seed) : std::nullopt,
        ctx->saved_data["is_causal"].toBool(),
        ctx->saved_data["dropout_p"].toScalar().toFloat(),
        ctx->saved_data["scale"].toScalar().toDouble(),
        "None", // softmax_mode - using default
        out);

    auto grad_query = std::get<0>(grad_result);
    auto grad_key = std::get<1>(grad_result);
    auto grad_value = std::get<2>(grad_result);
    // Note: dispatch_sdpa_recomp_bwd_wrap returns (grad_q, grad_k, grad_v)
    // For attn_bias grad, we need to handle it separately if needed
    auto grad_attn_bias = torch::Tensor(); // Currently not supported

    return {
        grad_query,
        grad_key,
        grad_value,
        bias_requires_grad ? grad_attn_bias : torch::Tensor(),
        torch::Tensor(), // dropout_p
        torch::Tensor(), // is_causal
        torch::Tensor(), // return_debug_mask
        torch::Tensor() // scale
    };
  }
};

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    c10::SymInt,
    c10::SymInt,
    at::Tensor,
    at::Tensor,
    at::Tensor>
fused_sdpa_overrideable_autograd_wrap(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  PT_EAGER_TRACE;

  // Always use the autograd version, just like fused_sdpa_autograd_wrap
  auto tensor_results = FusedSDPAOverrideableAutogradHPU::apply(
      query,
      key,
      value,
      attn_bias,
      dropout_p,
      is_causal,
      return_debug_mask,
      scale);

  // Reconstruct the full tuple with SymInts
  // We need to extract the SymInts from the query and key tensors
  c10::SymInt max_q = query.sym_size(2); // seq_len_q
  c10::SymInt max_k = key.sym_size(2); // seq_len_kv

  return std::make_tuple(
      tensor_results[0], // out
      tensor_results[1], // logsumexp
      tensor_results[2], // cum_seq_q
      tensor_results[3], // cum_seq_k
      max_q, // max_q
      max_k, // max_k
      tensor_results[4], // philox_seed
      tensor_results[5], // philox_offset
      tensor_results[6] // debug_attn_mask
  );
}

// Overrideable backend is always available when explicitly requested
// with the python context sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE])
// and autograd override will take over through
// fused_sdpa_overrideable_autograd_wrap

TORCH_LIBRARY_IMPL(aten, AutogradHPU, m) {
  m.impl(
      "_scaled_dot_product_fused_attention_overrideable",
      fused_sdpa_overrideable_autograd_wrap);
}

// By default, F.scaled_dot_product_attention uses
// math backend in torch.compile and eager mode

TORCH_LIBRARY_IMPL(aten, HPU, m) {
  // Register choice function for standard backend selection
  m.impl("_fused_sdp_choice", fused_sdp_choice_hpu);
}

} // namespace habana::eager
