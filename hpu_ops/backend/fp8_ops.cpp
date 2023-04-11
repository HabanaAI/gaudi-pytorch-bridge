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
    : OpBackend(device_id, "cast_to_fp8_", scalar_type, {}, {}, {}, true) {
  SetNumOutTensors(2);
}

void CastToFp8::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 5, "CastToFp8 must have 5 input arguments");

  auto self = stack_tensor(stack, 0);
  bool stochastic_rounding = stack[2].toBool();
  auto src_type = self.scalar_type();
  auto dst_type = at::ScalarType::Char;
  auto sizes = self.sizes();
  auto out = stack_tensor(stack, 3);
  auto amax = stack_tensor(stack, 4);

  TORCH_CHECK(
      sizes == out.sizes(), "Input and output must have the same shape");

  std::string guid = src_type == at::ScalarType::Float ? "convert_to_fp8_f32"
                                                       : "convert_to_fp8_bf16";

  ns_CastKernel::Params params{};
  params.round_mode = stochastic_rounding ? CAST_ROUND_SR : CAST_ROUND_HALF_NE;

  auto casted = OpBackend::BuildNode(
      this,
      graph,
      {guid,
       {syn_in(0), syn_in(1)},
       {{sizes, dst_type, 0, DATA_TENSOR, syn_type_fp8_152},
        {amax.sizes(), at::ScalarType::Float, 1}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(casted[0]);
  syn_out(1) = std::move(casted[1]);
}

Fp8CastTranspose::Fp8CastTranspose(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "fp8_cast_transpose_",
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
  bool stochastic_rounding = stack[2].toBool();
  auto dst_type = at::ScalarType::Char;
  auto sizes = self.sizes();
  auto out = stack_tensor(stack, 3);
  auto amax = stack_tensor(stack, 4);
  auto transposed = stack_tensor(stack, 5);

  TORCH_CHECK(
      sizes == out.sizes(), "Input and output must have the same shape");

  std::string guid = "convert_to_fp8_transpose_" +
      habana_helpers::name_suffix_from_type(self.scalar_type());

  ns_CastKernel::Params params{};
  params.round_mode = stochastic_rounding ? CAST_ROUND_SR : CAST_ROUND_HALF_NE;

  auto casted = OpBackend::BuildNode(
      this,
      graph,
      {guid,
       {syn_in(0), syn_in(1)},
       {{sizes, dst_type, 0, DATA_TENSOR, syn_type_fp8_152},
        {amax.sizes(), at::ScalarType::Float, 1},
        {transposed.sizes(), dst_type, 2, DATA_TENSOR, syn_type_fp8_152}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(casted[0]);
  syn_out(1) = std::move(casted[1]);
  syn_out(2) = std::move(casted[2]);
}

Fp8CastTransposeBgrad::Fp8CastTransposeBgrad(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "fp8_cast_transpose_bgrad_",
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
  bool stochastic_rounding = stack[2].toBool();
  auto dst_type = at::ScalarType::Char;
  auto sizes = self.sizes();
  auto out = stack_tensor(stack, 3);
  auto amax = stack_tensor(stack, 4);
  auto transposed = stack_tensor(stack, 5);
  auto bgrad = stack_tensor(stack, 6);

  TORCH_CHECK(
      sizes == out.sizes(), "Input and output must have the same shape");

  std::string guid = "convert_to_fp8_transpose_bgrad_" +
      habana_helpers::name_suffix_from_type(self.scalar_type());

  ns_CastKernel::Params params{};
  params.round_mode = stochastic_rounding ? CAST_ROUND_SR : CAST_ROUND_HALF_NE;

  auto casted = OpBackend::BuildNode(
      this,
      graph,
      {guid,
       {syn_in(0), syn_in(1)},
       {{sizes, dst_type, 0, DATA_TENSOR, syn_type_fp8_152},
        {amax.sizes(), at::ScalarType::Float, 1},
        {transposed.sizes(), dst_type, 2, DATA_TENSOR, syn_type_fp8_152},
        {bgrad.sizes(), self.scalar_type(), 3}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(casted[0]);
  syn_out(1) = std::move(casted[1]);
  syn_out(2) = std::move(casted[2]);
  syn_out(3) = std::move(casted[3]);
}

Fp8CastTransposeBgradDgelu::Fp8CastTransposeBgradDgelu(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "fp8_cast_transpose_bgrad_dgelu_",
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
  auto retain = stack[3].toOptional<torch::Tensor>().value_or(torch::Tensor());
  bool stochastic_rounding = stack[4].toBool();
  auto dst_type = at::ScalarType::Char;
  auto sizes = self.sizes();
  auto out = stack_tensor(stack, 5);
  auto amax = stack_tensor(stack, 6);
  auto transposed = stack_tensor(stack, 7);
  auto bgrad = stack_tensor(stack, 8);

  TORCH_CHECK(
      sizes == out.sizes(), "Input and output must have the same shape");

  std::string guid = "convert_to_fp8_transpose_bgrad_dgelu_" +
      habana_helpers::name_suffix_from_type(self.scalar_type());

  ns_CastKernel::Params params{};
  params.round_mode = stochastic_rounding ? CAST_ROUND_SR : CAST_ROUND_HALF_NE;

  std::vector<synTensor> syn_inputs = {syn_in(0), syn_in(1), syn_in(2)};
  if (retain.defined()) {
    syn_inputs.push_back(syn_in(3));
  }

  auto casted = OpBackend::BuildNode(
      this,
      graph,
      {guid,
       syn_inputs,
       {{sizes, dst_type, 0, DATA_TENSOR, syn_type_fp8_152},
        {amax.sizes(), at::ScalarType::Float, 1},
        {transposed.sizes(), dst_type, 2, DATA_TENSOR, syn_type_fp8_152},
        {bgrad.sizes(), self.scalar_type(), 3}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(casted[0]);
  syn_out(1) = std::move(casted[1]);
  syn_out(2) = std::move(casted[2]);
  syn_out(3) = std::move(casted[3]);
}

CastFromFp8::CastFromFp8(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "cast_from_fp8_", scalar_type, {0}, {}, {}, false) {}

void CastFromFp8::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 3, "CastFromFp8 must have 3 input arguments");

  auto self = stack_tensor(stack, 0);
  auto dst_type = stack[2].toScalarType();
  auto sizes = self.sizes();

  std::string guid = dst_type == at::ScalarType::Float
      ? "convert_from_fp8_f32"
      : "convert_from_fp8_bf16";

  auto casted = OpBackend::BuildNode(
      this, graph, {guid, {syn_in(0), syn_in(1)}, {{sizes, dst_type, 0}}});

  syn_out(0) = std::move(casted[0]);
}

Fp8Dropout::Fp8Dropout(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "fp8_dropout_",
          scalar_type,
          {0, 0, 2},
          {},
          {},
          false) {}

void Fp8Dropout::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 4, "Fp8Dropout must have 4 input arguments");

  auto self = stack_tensor(stack, 0);
  float p = static_cast<float>(stack[1].toDouble());
  bool stochastic_rounding = stack[3].toBool();
  auto src_type = self.scalar_type();
  auto dst_type = at::ScalarType::Char;
  auto sizes = self.sizes();

  std::string guid = "dropout_fp8_" +
      habana_helpers::name_suffix_from_type(self.scalar_type());

  ns_DropoutFp8::Params params{};
  params.round_mode = stochastic_rounding ? CAST_ROUND_SR : CAST_ROUND_HALF_NE;
  params.ratio = p;
  params.seed = habana::get_seed_hpu(c10::nullopt);

  auto dropout = OpBackend::BuildNode(
      this,
      graph,
      {guid,
       {syn_in(0), syn_in(1)},
       {{sizes, dst_type, 0, DATA_TENSOR, syn_type_fp8_152},
        {sizes, dst_type, 1},
        {{1}, at::ScalarType::Float, 2}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(dropout[0]);
  syn_out(1) = std::move(dropout[1]);
  syn_out(2) = std::move(dropout[2]);
}

Fp8Gelu::Fp8Gelu(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_gelu_", scalar_type, {}, {}, {}, true) {
  SetNumOutTensors(3);
}

void Fp8Gelu::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 6, "Fp8Gelu must have 6 input arguments");

  auto self = stack_tensor(stack, 0);
  bool stochastic_rounding = stack[2].toBool();
  auto src_type = self.scalar_type();
  auto dst_type = at::ScalarType::Char;
  auto sizes = self.sizes();
  auto out = stack_tensor(stack, 3);
  auto amax = stack_tensor(stack, 4);
  auto retain = stack_tensor(stack, 5);

  TORCH_CHECK(
      sizes == out.sizes(), "Input and output must have the same shape");

  std::string guid =
      src_type == at::ScalarType::Float ? "fp8_gelu_f32" : "fp8_gelu_bf16";

  ns_CastKernel::Params params{};
  params.round_mode = stochastic_rounding ? CAST_ROUND_SR : CAST_ROUND_HALF_NE;

  auto gelu = OpBackend::BuildNode(
      this,
      graph,
      {guid,
       {syn_in(0), syn_in(1)},
       {{sizes, dst_type, 0, DATA_TENSOR, syn_type_fp8_152},
        {amax.sizes(), at::ScalarType::Float, 1},
        {retain.sizes(), src_type, 2}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(gelu[0]);
  syn_out(1) = std::move(gelu[1]);
  syn_out(2) = std::move(gelu[2]);
}

Fp8Layernorm::Fp8Layernorm(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_layernorm_", scalar_type, {}, {}, {}, true) {
  SetNumOutTensors(4);
}

void Fp8Layernorm::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 10, "Fp8Layernorm must have 10 input arguments");

  auto self = stack_tensor(stack, 0);
  float eps = static_cast<float>(stack[3].toDouble());
  bool stochastic_rounding = stack[5].toBool();
  auto dst_type = at::ScalarType::Char;
  auto sizes = self.sizes();
  auto out = stack_tensor(stack, 6);
  auto amax = stack_tensor(stack, 7);
  auto mean = stack_tensor(stack, 8);
  auto istd = stack_tensor(stack, 9);

  TORCH_CHECK(
      sizes == out.sizes(), "Input and output must have the same shape");

  std::string guid = "layer_norm_fp8_fwd_" +
      habana_helpers::name_suffix_from_type(self.scalar_type());

  ns_LayerNormFp8::Params params{};
  params.round_mode = stochastic_rounding ? CAST_ROUND_SR : CAST_ROUND_HALF_NE;
  params.eps = eps;

  auto layernorm = OpBackend::BuildNode(
      this,
      graph,
      {guid,
       {syn_in(0), syn_in(1), syn_in(2), syn_in(3)},
       {{sizes, at::ScalarType::Char, 0, DATA_TENSOR, syn_type_fp8_152},
        {amax.sizes(), at::ScalarType::Float, 1},
        {mean.sizes(), at::ScalarType::Float, 2},
        {istd.sizes(), at::ScalarType::Float, 3}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(layernorm[0]);
  syn_out(1) = std::move(layernorm[1]);
  syn_out(2) = std::move(layernorm[2]);
  syn_out(3) = std::move(layernorm[3]);
}

Fp8Gemm::Fp8Gemm(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_gemm_", scalar_type, {}, {}, {}, true) {}

void Fp8Gemm::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 11, "Fp8Gemm must have 11 input arguments");

  auto A = stack_tensor(stack, 0);
  bool trans_A = stack[2].toBool();
  auto B = stack_tensor(stack, 3);
  bool trans_B = stack[5].toBool();
  auto out_type = stack[7].toScalarType();
  auto bias = stack[8].toOptional<torch::Tensor>().value_or(torch::Tensor());
  bool accumulate = stack[9].toBool();

  int64_t rank = A.dim();
  std::vector<int64_t> A_shape = A.sizes().vec();
  std::vector<int64_t> B_shape = B.sizes().vec();
  std::vector<int64_t> out_shape{A_shape.begin(), A_shape.begin() + rank - 2};
  int A_dim = rank - 2 + (trans_A ? 1 : 0);
  int B_dim = rank - 2 + (trans_B ? 0 : 1);
  out_shape.push_back(A_shape[A_dim]);
  out_shape.push_back(B_shape[B_dim]);

  std::string guid =
      out_type == at::ScalarType::Float ? "fp8_gemm_f32" : "fp8_gemm_bf16";

  std::vector<synTensor> syn_inputs = {
      syn_in(0), syn_in(2), syn_in(1), syn_in(3)};
  if (bias.defined()) {
    syn_inputs.push_back(syn_in(5));
  } else {
    syn_inputs.push_back(nullptr);
  }
  if (accumulate) {
    syn_inputs.push_back(syn_in(4));
  }

  synGEMMParams params{trans_A, trans_B};

  auto gemm = OpBackend::BuildNode(
      this,
      graph,
      {guid, syn_inputs, {{out_shape, out_type, 0}}, &params, sizeof(params)});

  syn_out(0) = std::move(gemm[0]);
}

Fp8Transpose::Fp8Transpose(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_transpose_", scalar_type, {}, {}, {}, true) {}

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
       {{out.sizes(), at::ScalarType::Char, 0, DATA_TENSOR, syn_type_fp8_152}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(transpose[0]);
}

Fp8Permute::Fp8Permute(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_permute_", scalar_type, {}, {}, {}, true) {}

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
       {{out.sizes(), at::ScalarType::Char, 0, DATA_TENSOR, syn_type_fp8_152}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(transpose[0]);
}

sizes_vec Fp8ReshapeOutputShape(const at::Stack& stack) {
  return {stack[1].toIntList().vec()};
}

Fp8Reshape::Fp8Reshape(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_reshape_", scalar_type, {0}, {}, {}, false) {
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
       {{shape, at::ScalarType::Char, 0, DATA_TENSOR, syn_type_fp8_152}}});

  syn_out(0) = std::move(reshape[0]);
}

} // namespace habana

static const auto& CastKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::cast_to_fp8", KERNEL_FN_GLOBAL(habana::CastToFp8))
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
        .add("hpu::fp8_layernorm", KERNEL_FN_GLOBAL(habana::Fp8Layernorm))
        .add("hpu::fp8_gemm", KERNEL_FN_GLOBAL(habana::Fp8Gemm))
        .add("hpu::fp8_transpose", KERNEL_FN_GLOBAL(habana::Fp8Transpose))
        .add("hpu::fp8_permute", KERNEL_FN_GLOBAL(habana::Fp8Permute))
        .add("hpu::fp8_reshape", KERNEL_FN_GLOBAL(habana::Fp8Reshape));
