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

#include "hpu_ops/fp8_ops.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {

CastToFp8::CastToFp8(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "cast_to_fp8", scalar_type, {}, {}, {}, true) {
  SetNumOutTensors(2);
}

void CastToFp8::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 5, "CastToFp8 must have 5 input arguments");

  auto self = stack_tensor(stack, 0);
  auto scale = stack[1].toOptional<torch::Tensor>().value_or(torch::Tensor());
  bool stochastic_rounding = stack[2].toBool();
  auto src_type = self.scalar_type();
  auto dst_type = at::ScalarType::Char;
  auto sizes = self.sizes();
  auto out = stack_tensor(stack, 3);
  auto amax = stack_tensor(stack, 4);

  bool is_amax = amax.numel() != 0;

  TORCH_CHECK(
      sizes == out.sizes(), "Input and output must have the same shape");

  std::string guid = src_type == at::ScalarType::Float ? "convert_to_fp8_f32"
                                                       : "convert_to_fp8_bf16";

  ns_CastKernel::Params params{};
  params.round_mode = stochastic_rounding ? CAST_ROUND_SR : CAST_ROUND_HALF_NE;

  std::vector<synTensor> syn_inputs{syn_in(0)};
  if (scale.defined()) {
    syn_inputs.push_back(syn_in(1));
  }
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {sizes, dst_type, 0, DATA_TENSOR, fp8_syn_type}};
  if (is_amax) {
    output_attrs.push_back({amax.sizes(), at::ScalarType::Float, 1});
  }

  auto casted = OpBackend::BuildNode(
      this, graph, {guid, syn_inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(casted[0]);
  if (is_amax) {
    syn_out(1) = std::move(casted[1]);
  }
}

sizes_vec CastToFp8V2OutputShape(const at::Stack& stack) {
  auto input_sv = stack[0].toTensor().sizes().vec();
  bool is_amax = stack[3].toBool();
  return {input_sv, std::vector<int64_t>{is_amax ? 1 : 0}};
}

std::shared_ptr<void> FillCastToFp8V2Params(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_CastKernel::Params);
  bool stochastic_rounding = stack[2].toBool();
  params->round_mode = stochastic_rounding ? CAST_ROUND_SR : CAST_ROUND_HALF_NE;
  return params;
}

CastToFp8V2::CastToFp8V2(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "cast_to_fp8_v2",
          scalar_type,
          {0, 0},
          {},
          {},
          false) {
  SetComputeOutputShapes(CastToFp8V2OutputShape);
  SetFillParams(FillCastToFp8V2Params);
}

void CastToFp8V2::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 4, "CastToFp8V2 must have 4 input arguments");

  auto self = stack_tensor(stack, 0);
  auto scale = stack[1].toOptional<torch::Tensor>().value_or(torch::Tensor());
  bool stochastic_rounding = stack[2].toBool();
  bool is_amax = stack[3].toBool();
  auto src_type = self.scalar_type();
  auto dst_type = at::ScalarType::Char;

  std::string guid = src_type == at::ScalarType::Float ? "convert_to_fp8_f32"
                                                       : "convert_to_fp8_bf16";

  auto out_shapes = CastToFp8V2OutputShape(stack);
  std::vector<synTensor> syn_inputs{syn_in(0)};
  if (scale.defined()) {
    syn_inputs.push_back(syn_in(1));
  }
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {out_shapes[0], dst_type, 0, DATA_TENSOR, fp8_syn_type}};
  if (is_amax) {
    output_attrs.push_back({out_shapes[1], at::ScalarType::Float, 1});
  }

  size_t size; // Will be initialized by below call
  const auto& params = FillCastToFp8V2Params(stack, size);

  auto casted = OpBackend::BuildNode(
      this, graph, {guid, syn_inputs, output_attrs, params.get(), size});

  syn_out(0) = std::move(casted[0]);
  if (is_amax) {
    syn_out(1) = std::move(casted[1]);
  }
}

Fp8CastTranspose::Fp8CastTranspose(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "fp8_cast_transpose",
          scalar_type,
          {},
          {},
          {},
          true) {
  SetNumOutTensors(3);
}

void Fp8CastTranspose::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(
      stack.size() == 6, "Fp8CastTranspose must have 6 input arguments");

  auto self = stack_tensor(stack, 0);
  auto scale = stack[1].toOptional<torch::Tensor>().value_or(torch::Tensor());
  bool stochastic_rounding = stack[2].toBool();
  auto dst_type = at::ScalarType::Char;
  auto sizes = self.sizes();
  auto out = stack_tensor(stack, 3);
  auto transposed = stack_tensor(stack, 4);
  auto amax = stack_tensor(stack, 5);

  bool is_amax = amax.numel() != 0;

  TORCH_CHECK(
      sizes == out.sizes(), "Input and output must have the same shape");

  std::string guid =
      get_guid_with_precision("convert_to_fp8_transpose", self.scalar_type());

  ns_CastKernel::Params params{};
  params.round_mode = stochastic_rounding ? CAST_ROUND_SR : CAST_ROUND_HALF_NE;

  std::vector<synTensor> syn_inputs{syn_in(0)};
  if (scale.defined()) {
    syn_inputs.push_back(syn_in(1));
  }
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {sizes, dst_type, 0, DATA_TENSOR, fp8_syn_type},
      {transposed.sizes(), dst_type, 1, DATA_TENSOR, fp8_syn_type}};
  if (is_amax) {
    output_attrs.push_back({amax.sizes(), at::ScalarType::Float, 2});
  }

  auto casted = OpBackend::BuildNode(
      this, graph, {guid, syn_inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(casted[0]);
  syn_out(1) = std::move(casted[1]);
  if (is_amax) {
    syn_out(2) = std::move(casted[2]);
  }
}

Fp8CastTransposeBgrad::Fp8CastTransposeBgrad(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "fp8_cast_transpose_bgrad",
          scalar_type,
          {},
          {},
          {},
          true) {
  SetNumOutTensors(4);
}

void Fp8CastTransposeBgrad::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(
      stack.size() == 7, "Fp8CastTransposeBgrad must have 7 input arguments");

  auto self = stack_tensor(stack, 0);
  auto scale = stack[1].toOptional<torch::Tensor>().value_or(torch::Tensor());
  bool stochastic_rounding = stack[2].toBool();
  auto dst_type = at::ScalarType::Char;
  auto sizes = self.sizes();
  auto out = stack_tensor(stack, 3);
  auto transposed = stack_tensor(stack, 4);
  auto bgrad = stack_tensor(stack, 5);
  auto amax = stack_tensor(stack, 6);

  bool is_amax = amax.numel() != 0;

  TORCH_CHECK(
      sizes == out.sizes(), "Input and output must have the same shape");

  std::string guid = get_guid_with_precision(
      "convert_to_fp8_transpose_bgrad", self.scalar_type());

  ns_CastKernel::Params params{};
  params.round_mode = stochastic_rounding ? CAST_ROUND_SR : CAST_ROUND_HALF_NE;

  std::vector<synTensor> syn_inputs{syn_in(0)};
  if (scale.defined()) {
    syn_inputs.push_back(syn_in(1));
  }
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {sizes, dst_type, 0, DATA_TENSOR, fp8_syn_type},
      {transposed.sizes(), dst_type, 1, DATA_TENSOR, fp8_syn_type},
      {bgrad.sizes(), self.scalar_type(), 2}};
  if (is_amax) {
    output_attrs.push_back({amax.sizes(), at::ScalarType::Float, 3});
  }

  auto casted = OpBackend::BuildNode(
      this, graph, {guid, syn_inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(casted[0]);
  syn_out(1) = std::move(casted[1]);
  syn_out(2) = std::move(casted[2]);
  if (is_amax) {
    syn_out(3) = std::move(casted[3]);
  }
}

Fp8CastTransposeBgradDgelu::Fp8CastTransposeBgradDgelu(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "fp8_cast_transpose_bgrad_dgelu",
          scalar_type,
          {},
          {},
          {},
          true) {
  SetNumOutTensors(4);
}

void Fp8CastTransposeBgradDgelu::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(
      stack.size() == 9,
      "Fp8CastTransposeBgradDgelu must have 9 input arguments");

  auto self = stack_tensor(stack, 0);
  auto scale = stack[2].toOptional<torch::Tensor>().value_or(torch::Tensor());
  auto retain = stack[3].toOptional<torch::Tensor>().value_or(torch::Tensor());
  bool stochastic_rounding = stack[4].toBool();
  auto dst_type = at::ScalarType::Char;
  auto sizes = self.sizes();
  auto out = stack_tensor(stack, 5);
  auto transposed = stack_tensor(stack, 6);
  auto bgrad = stack_tensor(stack, 7);
  auto amax = stack_tensor(stack, 8);

  bool is_amax = amax.numel() != 0;

  TORCH_CHECK(
      sizes == out.sizes(), "Input and output must have the same shape");

  std::string guid = get_guid_with_precision(
      "convert_to_fp8_transpose_bgrad_dgelu", self.scalar_type());

  ns_CastKernel::Params params{};
  params.round_mode = stochastic_rounding ? CAST_ROUND_SR : CAST_ROUND_HALF_NE;

  std::vector<synTensor> syn_inputs{syn_in(0), syn_in(1)};
  int retain_id = 3;
  if (scale.defined()) {
    syn_inputs.push_back(syn_in(2));
  } else {
    syn_inputs.push_back(nullptr);
    retain_id = 2;
  }
  if (retain.defined()) {
    syn_inputs.push_back(syn_in(retain_id));
  }
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {sizes, dst_type, 0, DATA_TENSOR, fp8_syn_type},
      {transposed.sizes(), dst_type, 1, DATA_TENSOR, fp8_syn_type},
      {bgrad.sizes(), self.scalar_type(), 2}};
  if (is_amax) {
    output_attrs.push_back({amax.sizes(), at::ScalarType::Float, 3});
  }

  auto casted = OpBackend::BuildNode(
      this, graph, {guid, syn_inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(casted[0]);
  syn_out(1) = std::move(casted[1]);
  syn_out(2) = std::move(casted[2]);
  if (is_amax) {
    syn_out(3) = std::move(casted[3]);
  }
}

CastFromFp8::CastFromFp8(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "cast_from_fp8", scalar_type, {0}, {}, {}, false) {}

void CastFromFp8::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 3, "CastFromFp8 must have 3 input arguments");

  auto self = stack_tensor(stack, 0);
  auto scale = stack[1].toOptional<torch::Tensor>().value_or(torch::Tensor());
  auto dst_type = stack[2].toScalarType();
  auto sizes = self.sizes();

  std::string guid = dst_type == at::ScalarType::Float
      ? "convert_from_fp8_f32"
      : "convert_from_fp8_bf16";

  std::vector<synTensor> syn_inputs{syn_in(0)};
  if (scale.defined()) {
    syn_inputs.push_back(syn_in(1));
  }

  auto casted = OpBackend::BuildNode(
      this, graph, {guid, syn_inputs, {{sizes, dst_type, 0}}});

  syn_out(0) = std::move(casted[0]);
}

sizes_vec Fp8DropoutOutputShape(const at::Stack& stack) {
  std::vector<int64_t> amax_size{1};
  auto size = stack_tensor(stack, 0).sizes().vec();
  return {size, size, amax_size};
}

Fp8Dropout::Fp8Dropout(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "fp8_dropout",
          scalar_type,
          {0, 0, 0},
          {},
          {},
          false) {
  SetComputeOutputShapes(Fp8DropoutOutputShape);
}

void Fp8Dropout::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 5, "Fp8Dropout must have 5 input arguments");

  StackGetter stackGetter(stack, "Fp8Dropout::AddNode");
  auto self = getNextInput<TensorsPair>(stackGetter);
  double p = getNextInput<double>(stackGetter);
  auto scaleOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  bool stochastic_rounding = getNextInput<bool>(stackGetter);
  bool is_amax = getNextInput<bool>(stackGetter);
  auto dst_type = at::ScalarType::Char;
  auto sizes = self.pt_t.sizes().vec();
  std::vector<int64_t> amax_size{1};

  std::string guid =
      get_guid_with_precision("dropout_fp8", self.pt_t.scalar_type());

  ns_DropoutFp8::Params params{};
  params.round_mode = stochastic_rounding ? CAST_ROUND_SR : CAST_ROUND_HALF_NE;
  params.ratio = static_cast<float>(p);
  params.seed = habana::get_seed_hpu(c10::nullopt);

  std::vector<synTensor> syn_inputs{self.syn_t};
  if (scaleOpt) {
    syn_inputs.push_back(scaleOpt->syn_t);
  }
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {sizes, dst_type, 0, DATA_TENSOR, fp8_syn_type}, {sizes, dst_type, 1}};
  if (is_amax) {
    output_attrs.push_back({amax_size, at::ScalarType::Float, 2});
  }

  auto dropout = OpBackend::BuildNode(
      this, graph, {guid, syn_inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(dropout[0]);
  syn_out(1) = std::move(dropout[1]);
  if (is_amax) {
    syn_out(2) = std::move(dropout[2]);
  }
}

Fp8Gelu::Fp8Gelu(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_gelu", scalar_type, {}, {}, {}, true) {
  SetNumOutTensors(3);
}

void Fp8Gelu::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 6, "Fp8Gelu must have 6 input arguments");

  auto self = stack_tensor(stack, 0);
  auto scale = stack[1].toOptional<torch::Tensor>().value_or(torch::Tensor());
  bool stochastic_rounding = stack[2].toBool();
  auto src_type = self.scalar_type();
  auto dst_type = at::ScalarType::Char;
  auto sizes = self.sizes();
  auto out = stack_tensor(stack, 3);
  auto retain = stack_tensor(stack, 4);
  auto amax = stack_tensor(stack, 5);

  bool is_amax = amax.numel() != 0;

  TORCH_CHECK(
      sizes == out.sizes(), "Input and output must have the same shape");

  std::string guid =
      src_type == at::ScalarType::Float ? "fp8_gelu_f32" : "fp8_gelu_bf16";

  ns_CastKernel::Params params{};
  params.round_mode = stochastic_rounding ? CAST_ROUND_SR : CAST_ROUND_HALF_NE;

  std::vector<synTensor> syn_inputs{syn_in(0)};
  if (scale.defined()) {
    syn_inputs.push_back(syn_in(1));
  }
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {sizes, dst_type, 0, DATA_TENSOR, fp8_syn_type},
      {retain.sizes(), src_type, 1}};
  if (is_amax) {
    output_attrs.push_back({amax.sizes(), at::ScalarType::Float, 2});
  }

  auto gelu = OpBackend::BuildNode(
      this, graph, {guid, syn_inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(gelu[0]);
  syn_out(1) = std::move(gelu[1]);
  if (is_amax) {
    syn_out(2) = std::move(gelu[2]);
  }
}

sizes_vec Fp8GeluV2OutputShape(const at::Stack& stack) {
  std::vector<int64_t> amax_size{1};
  auto size = stack_tensor(stack, 0).sizes().vec();
  return {size, size, amax_size};
}

Fp8GeluV2::Fp8GeluV2(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "fp8_gelu_v2",
          scalar_type,
          {0, 0, 0},
          {},
          {},
          false) {
  SetComputeOutputShapes(Fp8GeluV2OutputShape);
}

void Fp8GeluV2::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 4, "Fp8GeluV2 must have 4 input arguments");

  StackGetter stackGetter(stack, "Fp8GeluV2::AddNode");
  auto self = getNextInput<TensorsPair>(stackGetter);
  auto scaleOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  bool stochastic_rounding = getNextInput<bool>(stackGetter);
  bool is_amax = getNextInput<bool>(stackGetter);

  auto src_dtype = self.pt_t.scalar_type();

  std::string guid = get_guid_with_precision("fp8_gelu", src_dtype);

  ns_CastKernel::Params params{};
  params.round_mode = stochastic_rounding ? CAST_ROUND_SR : CAST_ROUND_HALF_NE;

  auto output_shapes = Fp8GeluV2OutputShape(stack);

  std::vector<synTensor> syn_inputs{self.syn_t};
  if (scaleOpt) {
    syn_inputs.push_back(scaleOpt->syn_t);
  }
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {output_shapes[0], at::ScalarType::Char, 0, DATA_TENSOR, fp8_syn_type},
      {output_shapes[1], src_dtype, 1}};
  if (is_amax) {
    output_attrs.push_back({output_shapes[2], at::ScalarType::Float, 2});
  }

  auto gelu = OpBackend::BuildNode(
      this, graph, {guid, syn_inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(gelu[0]);
  syn_out(1) = std::move(gelu[1]);
  if (is_amax) {
    syn_out(2) = std::move(gelu[2]);
  }
}

sizes_vec Fp8BgradDgeluOutputShape(const at::Stack& stack) {
  std::vector<int64_t> amax_size{1};
  auto out_size = stack_tensor(stack, 0).sizes().vec();
  auto bgrad_size = std::vector<int64_t>{out_size[1]};
  return {out_size, bgrad_size, amax_size};
}

Fp8BgradDgelu::Fp8BgradDgelu(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "fp8_bgrad_dgelu",
          scalar_type,
          {0, 0, 0},
          {},
          {},
          false) {}

void Fp8BgradDgelu::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 6, "Fp8BgradDgelu must have 6 input arguments");

  StackGetter stackGetter(stack, "Fp8eBgradDgelu::AddNode");
  auto grad = getNextInput<TensorsPair>(stackGetter);
  auto input = getNextInput<TensorsPair>(stackGetter);
  auto scaleOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto retainOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto stochastic_rounding = getNextInput<bool>(stackGetter);
  auto is_amax = getNextInput<bool>(stackGetter);

  auto out_sizes = Fp8BgradDgeluOutputShape(stack);

  std::string guid = get_guid_with_precision("fp8_bgrad_dgelu", ScalarType());

  ns_CastKernel::Params params{};
  params.round_mode = stochastic_rounding ? CAST_ROUND_SR : CAST_ROUND_HALF_NE;

  std::vector<synTensor> syn_inputs{grad.syn_t, input.syn_t};
  if (scaleOpt) {
    syn_inputs.push_back(scaleOpt->syn_t);
  } else {
    syn_inputs.push_back(nullptr);
  }
  if (retainOpt) {
    syn_inputs.push_back(retainOpt->syn_t);
  }
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {out_sizes[0], at::ScalarType::Char, 0, DATA_TENSOR, fp8_syn_type},
      {out_sizes[1], ScalarType(), 1}};
  if (is_amax) {
    output_attrs.push_back({out_sizes[2], at::ScalarType::Float, 2});
  }

  auto result = OpBackend::BuildNode(
      this, graph, {guid, syn_inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(result[0]);
  syn_out(1) = std::move(result[1]);
  if (is_amax) {
    syn_out(2) = std::move(result[2]);
  }
}

sizes_vec Fp8FastSoftmaxOutputShape(const at::Stack& stack) {
  std::vector<int64_t> amax_size{1};
  auto out_size = stack_tensor(stack, 0).sizes().vec();
  return {out_size, amax_size};
}

Fp8FastSoftmax::Fp8FastSoftmax(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "fp8_fast_softmax",
          scalar_type,
          {0, 0},
          {},
          {},
          false) {}

void Fp8FastSoftmax::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 6, "Fp8FastSoftmax must have 6 input arguments");

  StackGetter stackGetter(stack, "Fp8FastSoftmax::AddNode");
  auto input = getNextInput<TensorsPair>(stackGetter);
  auto mask = getNextInput<TensorsPair>(stackGetter);
  auto scale_opt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto softmax_scale = static_cast<float>(getNextInput<double>(stackGetter));
  auto stochastic_rounding = getNextInput<bool>(stackGetter);
  auto is_amax = getNextInput<bool>(stackGetter);

  auto out_sizes = Fp8FastSoftmaxOutputShape(stack);

  std::string guid = get_guid_with_precision("fp8_fast_softmax", ScalarType());

  // reuse params structure from LayerNormFp8
  ns_LayerNormFp8::Params params{};
  params.round_mode = stochastic_rounding ? CAST_ROUND_SR : CAST_ROUND_HALF_NE;
  params.eps = softmax_scale;

  std::vector<synTensor> syn_inputs{input.syn_t, mask.syn_t};
  if (scale_opt) {
    syn_inputs.push_back(scale_opt->syn_t);
  }
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {out_sizes[0], at::ScalarType::Char, 0, DATA_TENSOR, fp8_syn_type}};
  if (is_amax) {
    output_attrs.push_back({out_sizes[1], at::ScalarType::Float, 1});
  }

  auto result = OpBackend::BuildNode(
      this, graph, {guid, syn_inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(result[0]);
  if (is_amax) {
    syn_out(1) = std::move(result[1]);
  }
}

Fp8Layernorm::Fp8Layernorm(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_layernorm", scalar_type, {}, {}, {}, true) {
  SetNumOutTensors(4);
}

void Fp8Layernorm::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 10, "Fp8Layernorm must have 10 input arguments");

  auto self = stack_tensor(stack, 0);
  float eps = static_cast<float>(stack[3].toDouble());
  auto scale = stack[4].toOptional<torch::Tensor>().value_or(torch::Tensor());
  bool stochastic_rounding = stack[5].toBool();
  auto dst_type = at::ScalarType::Char;
  auto sizes = self.sizes();
  auto out = stack_tensor(stack, 6);
  auto mean = stack_tensor(stack, 7);
  auto istd = stack_tensor(stack, 8);
  auto amax = stack_tensor(stack, 9);

  bool is_amax = amax.numel() != 0;

  TORCH_CHECK(
      sizes == out.sizes(), "Input and output must have the same shape");

  std::string guid =
      get_guid_with_precision("layer_norm_fp8_fwd", self.scalar_type());

  ns_LayerNormFp8::Params params{};
  params.round_mode = stochastic_rounding ? CAST_ROUND_SR : CAST_ROUND_HALF_NE;
  params.eps = eps;

  std::vector<synTensor> syn_inputs{syn_in(0), syn_in(1), syn_in(2)};
  if (scale.defined()) {
    syn_inputs.push_back(syn_in(3));
  }
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {sizes, dst_type, 0, DATA_TENSOR, fp8_syn_type},
      {mean.sizes(), at::ScalarType::Float, 1},
      {istd.sizes(), at::ScalarType::Float, 2}};
  if (is_amax) {
    output_attrs.push_back({amax.sizes(), at::ScalarType::Float, 3});
  }

  auto layernorm = OpBackend::BuildNode(
      this, graph, {guid, syn_inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(layernorm[0]);
  syn_out(1) = std::move(layernorm[1]);
  syn_out(2) = std::move(layernorm[2]);
  if (is_amax) {
    syn_out(3) = std::move(layernorm[3]);
  }
}

Fp8Gemm::Fp8Gemm(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_gemm", scalar_type, {}, {}, {}, true) {}

void Fp8Gemm::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 11, "Fp8Gemm must have 11 input arguments");

  StackGetter stackGetter(stack, "Fp8Gemm::AddNode");
  auto A = getNextInput<TensorsPair>(stackGetter);
  bool trans_A = getNextInput<bool>(stackGetter);
  auto B = getNextInput<TensorsPair>(stackGetter);
  bool trans_B = getNextInput<bool>(stackGetter);
  auto D = getNextInput<TensorsPair>(stackGetter);
  auto out_type = getNextInput<c10::ScalarType>(stackGetter);
  auto scaleAOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto scaleBOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto biasOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  bool accumulate = getNextInput<bool>(stackGetter);

  int64_t rank = A.pt_t.dim();
  std::vector<int64_t> A_shape = A.pt_t.sizes().vec();
  std::vector<int64_t> B_shape = B.pt_t.sizes().vec();
  std::vector<int64_t> out_shape{A_shape.begin(), A_shape.begin() + rank - 2};
  int A_dim = rank - 2 + (trans_A ? 1 : 0);
  int B_dim = rank - 2 + (trans_B ? 0 : 1);
  out_shape.push_back(A_shape[A_dim]);
  out_shape.push_back(B_shape[B_dim]);

  std::string guid = get_guid_with_precision("fp8_gemm", out_type);

  std::vector<synTensor> syn_inputs = {A.syn_t, B.syn_t};
  if (scaleAOpt) {
    syn_inputs.push_back(scaleAOpt->syn_t);
  } else {
    syn_inputs.push_back(nullptr);
  }
  if (scaleBOpt) {
    syn_inputs.push_back(scaleBOpt->syn_t);
  } else {
    syn_inputs.push_back(nullptr);
  }
  if (biasOpt) {
    syn_inputs.push_back(biasOpt->syn_t);
  } else {
    syn_inputs.push_back(nullptr);
  }
  if (accumulate) {
    syn_inputs.push_back(D.syn_t);
  }

  synGEMMParams params{trans_A, trans_B};

  auto gemm = OpBackend::BuildNode(
      this,
      graph,
      {guid, syn_inputs, {{out_shape, out_type, 0}}, &params, sizeof(params)});

  syn_out(0) = std::move(gemm[0]);
}

sizes_vec Fp8GemmV2OutputShape(const at::Stack& stack) {
  auto A = stack_tensor(stack, 0);
  bool trans_A = stack[1].toBool();
  auto B = stack_tensor(stack, 2);
  bool trans_B = stack[3].toBool();

  int64_t rank = A.dim();
  std::vector<int64_t> A_shape = A.sizes().vec();
  std::vector<int64_t> B_shape = B.sizes().vec();
  std::vector<int64_t> out_shape{A_shape.begin(), A_shape.begin() + rank - 2};
  int A_dim = rank - 2 + (trans_A ? 1 : 0);
  int B_dim = rank - 2 + (trans_B ? 0 : 1);
  out_shape.push_back(A_shape[A_dim]);
  out_shape.push_back(B_shape[B_dim]);

  return {out_shape};
}

Fp8GemmV2::Fp8GemmV2(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_gemm_v2", scalar_type, {0}, {}, {}, false) {
  SetComputeOutputShapes(Fp8GemmV2OutputShape);
}

void Fp8GemmV2::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 10, "Fp8GemmV2 must have 10 input arguments");

  StackGetter stackGetter(stack, "Fp8Gemm::AddNode");
  auto A = getNextInput<TensorsPair>(stackGetter);
  bool trans_A = getNextInput<bool>(stackGetter);
  auto B = getNextInput<TensorsPair>(stackGetter);
  bool trans_B = getNextInput<bool>(stackGetter);
  auto DOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto out_type = getNextInput<c10::ScalarType>(stackGetter);
  auto scaleAOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto scaleBOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto biasOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  bool accumulate = getNextInput<bool>(stackGetter);

  std::string guid = get_guid_with_precision("fp8_gemm", out_type);

  std::vector<synTensor> syn_inputs = {A.syn_t, B.syn_t};
  if (scaleAOpt) {
    syn_inputs.push_back(scaleAOpt->syn_t);
  } else {
    syn_inputs.push_back(nullptr);
  }
  if (scaleBOpt) {
    syn_inputs.push_back(scaleBOpt->syn_t);
  } else {
    syn_inputs.push_back(nullptr);
  }
  if (biasOpt) {
    syn_inputs.push_back(biasOpt->syn_t);
  } else {
    syn_inputs.push_back(nullptr);
  }
  if (accumulate) {
    TORCH_CHECK(
        DOpt,
        "Accumulation tensor must be provided at index 4 for Fp8GemmV2, when accumulate is true");
    syn_inputs.push_back(DOpt->syn_t);
  }

  synGEMMParams params{trans_A, trans_B};

  auto out_shapes = Fp8GemmV2OutputShape(stack);

  auto gemm = OpBackend::BuildNode(
      this,
      graph,
      {guid,
       syn_inputs,
       {{out_shapes[0], out_type, 0}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(gemm[0]);
}

Fp8Transpose::Fp8Transpose(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_transpose", scalar_type, {}, {}, {}, true) {}

void Fp8Transpose::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto out = stack_tensor(stack, 1);

  synTransposeParams params{};
  params.tensorDim = 2;
  params.permutation[0] = static_cast<TransposePermutationDim>(1);
  params.permutation[1] = static_cast<TransposePermutationDim>(0);

  auto transpose = OpBackend::BuildNode(
      this,
      graph,
      {"transpose",
       {syn_in(0)},
       {{out.sizes(), at::ScalarType::Char, 0, DATA_TENSOR, fp8_syn_type}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(transpose[0]);
}

Fp8Permute::Fp8Permute(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_permute", scalar_type, {}, {}, {}, true) {}

void Fp8Permute::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto dims = stack[1].toIntList().vec();
  auto out = stack_tensor(stack, 2);
  auto dims_size = dims.size();

  synTransposeParams params{};
  params.tensorDim = dims_size;
  for (int i = 0; i < dims_size; i++) {
    params.permutation[i] = static_cast<TransposePermutationDim>(
        dims_size - dims[dims_size - i - 1] - 1);
  }

  auto transpose = OpBackend::BuildNode(
      this,
      graph,
      {"transpose",
       {syn_in(0)},
       {{out.sizes(), at::ScalarType::Char, 0, DATA_TENSOR, fp8_syn_type}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(transpose[0]);
}

sizes_vec Fp8ReshapeOutputShape(const at::Stack& stack) {
  return {stack[1].toIntList().vec()};
}

Fp8Reshape::Fp8Reshape(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_reshape", scalar_type, {0}, {}, {}, false) {
  SetComputeOutputShapes(Fp8ReshapeOutputShape);
}

void Fp8Reshape::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto shape = stack[1].toIntList().vec();

  std::vector<synTensor> inputs = {syn_in(0)};
  CreateShapeTensorInput(graph, at::ScalarType::Char, shape, inputs);

  auto reshape = BuildNode(
      this,
      graph,
      {"reshape",
       inputs,
       {{shape, at::ScalarType::Char, 0, DATA_TENSOR, fp8_syn_type}}});

  syn_out(0) = std::move(reshape[0]);
}

Fp8Copy_::Fp8Copy_(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_copy_", scalar_type, {}, {0}, {}, false) {}

void Fp8Copy_::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 2, "Fp8Copy_ must have 2 input arguments");

  std::string guid_suffix = fp8_syn_type == syn_type_fp8_143 ? "hf8" : "f8";
  auto shape = stack[0].toTensor().sizes().vec();
  auto copy = BuildNode(
      this,
      graph,
      {"memcpy_" + guid_suffix,
       {syn_in(1)},
       {{shape, at::ScalarType::Char, 0, DATA_TENSOR, fp8_syn_type}}});

  syn_out(0) = std::move(copy[0]);
}

Fp8KvReorder::Fp8KvReorder(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_kv_reorder", scalar_type, {}, {0}, {}, false) {}

void Fp8KvReorder::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 4, "Fp8KvReorder must have 4 input arguments");

  StackGetter stackGetter(stack, "Fp8KvReorder::AddNode");
  auto self = getNextInput<TensorsPair>(stackGetter);
  auto start = getNextInput<TensorsPair>(stackGetter);
  auto end = getNextInput<TensorsPair>(stackGetter);
  auto beam_idx = getNextInput<TensorsPair>(stackGetter);

  TORCH_CHECK(
      start.pt_t.dtype() == c10::ScalarType::Int,
      "Start tensor must be of type Int32");
  TORCH_CHECK(
      end.pt_t.dtype() == c10::ScalarType::Int,
      "End tensor must be of type Int32");
  TORCH_CHECK(
      beam_idx.pt_t.dtype() == c10::ScalarType::Byte,
      "Beam_idx tensor must be of type UInt8");
  TORCH_CHECK(start.pt_t.dim() == 1, "Start tensor must have dimensions 1");
  TORCH_CHECK(end.pt_t.dim() == 1, "End tensor must have dimensions 1");
  TORCH_CHECK(
      beam_idx.pt_t.dim() == 1, "Beam_idx tensor must have dimensions 1");

  std::string guid_suffix = fp8_syn_type == syn_type_fp8_143 ? "hf8" : "f8";
  auto shape = self.pt_t.sizes().vec();
  auto selective_gather = BuildNode(
      this,
      graph,
      {"selective_gather_fwd_" + guid_suffix,
       {self.syn_t, start.syn_t, end.syn_t, beam_idx.syn_t},
       {{shape, at::ScalarType::Char, 0, DATA_TENSOR, fp8_syn_type}}});

  syn_out(0) = std::move(selective_gather[0]);
}

Fp8IndexCopy_::Fp8IndexCopy_(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_index_copy_", scalar_type, {}, {0}, {}, false) {
}

void Fp8IndexCopy_::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 4, "Fp8IndexCopy_ must have 4 input arguments");

  StackGetter stackGetter(stack, "IndexCopy::AddNode");
  auto self = getNextInput<TensorsPair>(stackGetter);
  int dim = getNextInput<int>(stackGetter);
  auto index = getNextInput<TensorsPair>(stackGetter);
  auto source = getNextInput<TensorsPair>(stackGetter);

  std::string guid_suffix = fp8_syn_type == syn_type_fp8_143 ? "hf8" : "f8";
  auto shape = self.pt_t.sizes().vec();

  ns_IndexCopy::Params params{};
  params.axis = dim;

  auto copy = BuildNode(
      this,
      graph,
      {"index_copy_fwd_" + guid_suffix,
       {self.syn_t, index.syn_t, source.syn_t},
       {{shape, at::ScalarType::Char, 0, DATA_TENSOR, fp8_syn_type}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(copy[0]);
}

sizes_vec Fp8RepeatV2OutputShape(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto repeats = stack[1].toIntList();

  int64_t num_new_dimensions = repeats.size() - self.dim();
  std::vector<int64_t> padded_size(num_new_dimensions, 1);
  padded_size.insert(
      padded_size.end(), self.sizes().begin(), self.sizes().end());
  std::vector<int64_t> outshape(repeats.size());
  for (size_t i = 0; i < repeats.size(); ++i) {
    outshape[i] = padded_size[i] * repeats[i];
  }

  return {outshape};
}

Fp8RepeatV2::Fp8RepeatV2(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_repeat_v2", scalar_type, {0}, {}, {}, false) {
  SetComputeOutputShapes(Fp8RepeatV2OutputShape);
}

void Fp8RepeatV2::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 2, "Fp8RepeatV2 must have 2 input arguments");

  StackGetter stackGetter(stack, "Fp8RepeatV2::AddNode");
  auto self = getNextInput<TensorsPair>(stackGetter);
  auto repeats = getNextInput<std::vector<int64_t>>(stackGetter);

  std::string guid_suffix = fp8_syn_type == syn_type_fp8_143 ? "hf8" : "f8";

  synTensor self_reshaped_st = self.syn_t;
  std::optional<synapse_helpers::tensor> self_reshaped_storage;
  if (repeats.size() > self.pt_t.ndimension()) {
    int64_t num_new_dimensions = repeats.size() - self.pt_t.dim();
    std::vector<int64_t> padded_size(num_new_dimensions, 1);
    padded_size.insert(
        padded_size.end(), self.pt_t.sizes().begin(), self.pt_t.sizes().end());

    std::vector<synTensor> inputs = {self.syn_t};
    CreateShapeTensorInput(graph, at::ScalarType::Char, padded_size, inputs);

    self_reshaped_storage = std::move(BuildNode(
                                          this,
                                          graph,
                                          {"reshape",
                                           inputs,
                                           {{padded_size,
                                             at::ScalarType::Char,
                                             0,
                                             DATA_TENSOR,
                                             fp8_syn_type}}})
                                          .at(0));

    self_reshaped_st = (*self_reshaped_storage).get();
  }

  ns_TileKernel::ParamsV2 params{};
  for (int64_t i = 0; i < repeats.size(); ++i) {
    params.repeat[repeats.size() - i - 1] = repeats[i];
  }

  auto out_shapes = Fp8RepeatV2OutputShape(stack);

  auto result = OpBackend::BuildNode(
      this,
      graph,
      {"tile_fwd_" + guid_suffix,
       {self_reshaped_st},
       {{out_shapes[0], at::ScalarType::Char, 0, DATA_TENSOR, fp8_syn_type}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(result[0]);
}

sizes_vec Fp8IndexSelectV2OutputShape(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  int dim = stack[1].toInt();
  auto index = stack_tensor(stack, 2);
  dim = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);

  std::vector<int64_t> outshape(self.sizes().vec());
  if (self.dim() > 0) {
    outshape[dim] = index.numel();
  }

  return {outshape};
}

Fp8IndexSelectV2::Fp8IndexSelectV2(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "fp8_index_select_v2",
          scalar_type,
          {0},
          {},
          {},
          false) {
  SetComputeOutputShapes(Fp8IndexSelectV2OutputShape);
}

void Fp8IndexSelectV2::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(
      stack.size() == 3, "Fp8IndexSelectV2 must have 3 input arguments");

  StackGetter stackGetter(stack, "FpIndexSelectV2::AddNode");
  auto self = getNextInput<TensorsPair>(stackGetter);
  int dim = getNextInput<int>(stackGetter);
  auto index = getNextInput<TensorsPair>(stackGetter);
  dim = at::maybe_wrap_dim(dim, self.pt_t.dim(), /*wrap_scalar=*/true);

  std::string guid_suffix = fp8_syn_type == syn_type_fp8_143 ? "hf8" : "f8";

  auto out_shapes = Fp8IndexSelectV2OutputShape(stack);

  ns_GatherKernel::Params params{};
  params.axis = self.pt_t.dim() - dim - 1;

  auto result = OpBackend::BuildNode(
      this,
      graph,
      {"gather_fwd_" + guid_suffix,
       {self.syn_t, index.syn_t},
       {{out_shapes[0], at::ScalarType::Char, 0, DATA_TENSOR, fp8_syn_type}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(result[0]);
}

} // namespace habana

static const auto& CastKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::cast_to_fp8", KERNEL_FN_GLOBAL(habana::CastToFp8))
        .add("hpu::cast_to_fp8_v2", KERNEL_FN_GLOBAL(habana::CastToFp8V2))
        .add(
            "hpu::fp8_cast_transpose",
            KERNEL_FN_GLOBAL(habana::Fp8CastTranspose))
        .add(
            "hpu::fp8_cast_transpose_bgrad",
            KERNEL_FN_GLOBAL(habana::Fp8CastTransposeBgrad))
        .add(
            "hpu::fp8_cast_transpose_bgrad_dgelu",
            KERNEL_FN_GLOBAL(habana::Fp8CastTransposeBgradDgelu))
        .add("hpu::cast_from_fp8", KERNEL_FN_GLOBAL(habana::CastFromFp8))
        .add("hpu::fp8_dropout", KERNEL_FN_GLOBAL(habana::Fp8Dropout))
        .add("hpu::fp8_gelu", KERNEL_FN_GLOBAL(habana::Fp8Gelu))
        .add("hpu::fp8_gelu_v2", KERNEL_FN_GLOBAL(habana::Fp8GeluV2))
        .add("hpu::fp8_bgrad_dgelu", KERNEL_FN_GLOBAL(habana::Fp8BgradDgelu))
        .add("hpu::fp8_fast_softmax", KERNEL_FN_GLOBAL(habana::Fp8FastSoftmax))
        .add("hpu::fp8_layernorm", KERNEL_FN_GLOBAL(habana::Fp8Layernorm))
        .add("hpu::fp8_gemm", KERNEL_FN_GLOBAL(habana::Fp8Gemm))
        .add("hpu::fp8_gemm_v2", KERNEL_FN_GLOBAL(habana::Fp8GemmV2))
        .add("hpu::fp8_transpose", KERNEL_FN_GLOBAL(habana::Fp8Transpose))
        .add("hpu::fp8_permute", KERNEL_FN_GLOBAL(habana::Fp8Permute))
        .add("hpu::fp8_reshape", KERNEL_FN_GLOBAL(habana::Fp8Reshape))
        .add("hpu::fp8_copy_", KERNEL_FN_GLOBAL(habana::Fp8Copy_))
        .add("hpu::fp8_kv_reorder_", KERNEL_FN_GLOBAL(habana::Fp8KvReorder))
        .add("hpu::fp8_index_copy_", KERNEL_FN_GLOBAL(habana::Fp8IndexCopy_))
        .add("hpu::fp8_repeat_v2", KERNEL_FN_GLOBAL(habana::Fp8RepeatV2))
        .add(
            "hpu::fp8_index_select_v2",
            KERNEL_FN_GLOBAL(habana::Fp8IndexSelectV2));
