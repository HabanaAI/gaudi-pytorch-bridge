/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
#include "generated/backend/softmax_fp8.h"

namespace habana {

namespace {

void addOptionalTensor(
    const c10::optional<OpBackend::TensorsPair>& input_opt,
    std::vector<synTensor>& syn_inputs,
    const at::ScalarType dtype,
    const std::string& input_name) {
  if (input_opt) {
    TORCH_CHECK(
        input_opt->pt_t.scalar_type() == dtype,
        "Input ",
        input_name,
        " must be of dtype: ",
        dtype);
    syn_inputs.push_back(input_opt->syn_t);
  } else {
    syn_inputs.push_back(nullptr);
  }
}

} // namespace

OutputMetaDataVector SoftmaxFp8Meta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = stack_tensor(stack, 0).sizes().vec();
  meta.dtype = stack[2].isTensor() ? at::ScalarType::Float8_e4m3fn
                                   : at::ScalarType::BFloat16;
  return {meta};
}

void SoftmaxFp8::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "SoftmaxFp8::AddNode");
  auto self = getNextInput<TensorsPair>(stackGetter);
  int dim = getNextInput<int>(stackGetter);
  auto input_scale_opt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto output_scale_opt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto inv_attn_heads_opt =
      getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto fused_add_opt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto rank = self.pt_t.dim();
  dim = at::maybe_wrap_dim(dim, rank, /*wrap_scalar=*/true);
  auto out_meta = SoftmaxFp8Meta(stack)[0];

  TORCH_CHECK(
      self.pt_t.scalar_type() == at::ScalarType::BFloat16,
      "Input tensor must be of torch.bfloat16 dtype.");

  TORCH_CHECK(
      (input_scale_opt && output_scale_opt) ||
          (!input_scale_opt && !output_scale_opt),
      "Output and input scales must be both given or None.");

  if (fused_add_opt) {
    TORCH_CHECK((input_scale_opt), "FusedAdd available only for Float8 output");
    TORCH_CHECK(
        (rank == fused_add_opt->pt_t.dim()),
        "FusedAdd tensor must have the same rank as the input tensor");
    TORCH_CHECK(
        (self.pt_t.sizes()[0] == fused_add_opt->pt_t.sizes()[0] &&
         self.pt_t.sizes()[rank - 1] == fused_add_opt->pt_t.sizes()[rank - 1]),
        "FusedAdd tensor must have the same first and last dim as the input tensor");
    for (int dim_id = 1; dim_id < rank - 1; dim_id++)
      TORCH_CHECK(
          (fused_add_opt->pt_t.sizes()[dim_id] == 1 ||
           fused_add_opt->pt_t.sizes()[dim_id] == self.pt_t.sizes()[dim_id]),
          "FusedAdd tensor's dim other than first and last should be equal to 1 or the same as the input tensor");
  }

  ns_Softmax::ParamsV7 params{};
  params.dim = static_cast<int>(rank - dim - 1);
  int mode = input_scale_opt ? SoftmaxMode_t::SOFTMAX_HF8_1B
                             : SoftmaxMode_t::SOFTMAX_HF8_1C;
  if (fused_add_opt)
    mode |= SoftmaxMode_t::FUSED_ADD;
  params.mode = static_cast<SoftmaxMode_t>(mode);

  // valid count, max tensor, and reciprocal sum of EXP are optional tensors
  // and are not used by fp8 version but need to be passed as nullptr to place
  // rest optional inputs on correct positions.
  std::vector<synTensor> syn_inputs{self.syn_t, nullptr, nullptr, nullptr};
  addOptionalTensor(
      input_scale_opt, syn_inputs, at::ScalarType::Float, "input_scale");
  addOptionalTensor(
      inv_attn_heads_opt, syn_inputs, at::ScalarType::Float, "inv_attn_heads");
  addOptionalTensor(
      output_scale_opt, syn_inputs, at::ScalarType::Float, "output_scale");
  addOptionalTensor(
      fused_add_opt, syn_inputs, at::ScalarType::BFloat16, "fused_add");

  auto result = OpBackend::BuildNode(
      this,
      graph,
      {"softmax_fwd_hf8",
       std::move(syn_inputs),
       {{out_meta.shape, out_meta.dtype, 0}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(result[0]);
}

} // namespace habana
