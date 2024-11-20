/**
* Copyright (c) 2021-2024 Intel Corporation
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
#include "generated/backend/softmax_fp8.h"

namespace sh = synapse_helpers;

namespace habana {

namespace {

void addOptionalTensor(
    const c10::optional<TensorsPair>& input_opt,
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

void handleScale(
    habana::OpBackend* op,
    sh::graph& graph,
    const VariantWrapper<TensorsPair, c10::IValue>& scale,
    std::vector<sh::tensor>& const_scales,
    std::vector<synTensor>& syn_inputs) {
  // If scale is a Tensor, add respective synTensor to the node inputs.
  if (scale.isTensorsPair()) {
    addOptionalTensor(
        scale.toTensorsPair(), syn_inputs, at::ScalarType::Float, "scale");
    return;
  }

  // If scale is a Scalar, create a const tensor first.
  const auto scale_value = scale.toIValue();
  if (scale_value.isDouble()) {
    const_scales.emplace_back(
        op->BuildConstantTensor(op, graph, scale_value.toDouble()));
    syn_inputs.push_back(const_scales.back().get());
  } else {
    syn_inputs.push_back(nullptr);
  }
}

} // namespace

OutputMetaDataVector SoftmaxFp8Meta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = stack_tensor(stack, 0).sizes().vec();
  meta.dtype = stack[2].isNone() ? at::ScalarType::BFloat16
                                 : at::ScalarType::Float8_e4m3fn;
  return {meta};
}

void SoftmaxFp8::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "SoftmaxFp8::AddNode");
  auto self = stackGetter.getNextInput<TensorsPair>();
  int dim = stackGetter.getNextInput<int>();
  auto input_scale_opt =
      stackGetter.getNextInput<std::variant<TensorsPair, c10::IValue>>();
  auto output_scale_opt =
      stackGetter.getNextInput<std::variant<TensorsPair, c10::IValue>>();
  auto inv_attn_heads_opt =
      stackGetter.getNextInput<std::variant<TensorsPair, c10::IValue>>();
  auto fused_add_opt = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  auto rank = self.pt_t.dim();
  dim = at::maybe_wrap_dim(dim, rank, /*wrap_scalar=*/true);
  auto out_meta = SoftmaxFp8Meta(stack)[0];

  const auto& self_dtype = self.pt_t.scalar_type();
  TORCH_CHECK(
      self_dtype == at::ScalarType::BFloat16 ||
          self_dtype == at::ScalarType::Float8_e4m3fn,
      "Input tensor must be of torch.bfloat16 or torch.float8_e4m3fn dtype.");

  const bool is_input_scale =
      input_scale_opt.isTensorsPair() or input_scale_opt.toIValue().isDouble();
  const bool is_output_scale = output_scale_opt.isTensorsPair() or
      output_scale_opt.toIValue().isDouble();

  TORCH_CHECK(
      is_input_scale == is_output_scale,
      "Output and input scales must be both given or None");

  TORCH_CHECK(
      !(self_dtype == at::ScalarType::Float8_e4m3fn) or is_input_scale,
      "If Input is of torch.float8_e4m3fn dtype then input scale must be given.");

  if (fused_add_opt) {
    TORCH_CHECK(
        is_input_scale,
        "FusedAdd available only for Float8 output, but input scale is not given.");
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
  int mode = is_input_scale ? self_dtype == at::ScalarType::Float8_e4m3fn
          ? SoftmaxMode_t::SOFTMAX_HF8_2B
          : SoftmaxMode_t::SOFTMAX_HF8_1B
                            : SoftmaxMode_t::SOFTMAX_HF8_1C;
  if (fused_add_opt)
    mode |= SoftmaxMode_t::FUSED_ADD;
  params.mode = static_cast<SoftmaxMode_t>(mode);

  // valid count, max tensor, and reciprocal sum of EXP are optional tensors
  // and are not used by fp8 version but need to be passed as nullptr to place
  // rest optional inputs on correct positions.
  std::vector<synTensor> syn_inputs{self.syn_t, nullptr, nullptr, nullptr};
  std::vector<sh::tensor> const_scales;
  handleScale(this, graph, input_scale_opt, const_scales, syn_inputs);
  handleScale(this, graph, inv_attn_heads_opt, const_scales, syn_inputs);
  handleScale(this, graph, output_scale_opt, const_scales, syn_inputs);
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
