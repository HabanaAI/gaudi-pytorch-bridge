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

#include "hpu_ops/lazy_fp8_ops.h"

namespace habana {

LazyCastToFp8::LazyCastToFp8(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "cast_to_fp8_", scalar_type, {}, {}, {}, true) {
  SetNumOutTensors(2);
}

void LazyCastToFp8::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
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

LazyCastFromFp8::LazyCastFromFp8(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "cast_from_fp8_", scalar_type, {0}, {}, {}, false) {}

void LazyCastFromFp8::AddNode(
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

LazyFp8Gemm::LazyFp8Gemm(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_gemm_", scalar_type, {}, {}, {}, true) {}

void LazyFp8Gemm::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
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

LazyFp8Transpose::LazyFp8Transpose(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_transpose_", scalar_type, {}, {}, {}, true) {}

void LazyFp8Transpose::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto A = stack_tensor(stack, 0);
  std::vector<int64_t> out_shape = {A.sizes().vec()[1], A.sizes().vec()[0]};

  synTransposeParams params{};
  params.tensorDim = 2;
  params.permutation[0] = static_cast<TransposePermutationDim>(1);
  params.permutation[1] = static_cast<TransposePermutationDim>(0);

  auto transpose = OpBackend::BuildNode(
      this,
      graph,
      {"transpose",
       {syn_in(0)},
       {{out_shape, at::ScalarType::Char, 0, DATA_TENSOR, syn_type_fp8_152}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(transpose[0]);
}

} // namespace habana

static const auto& CastKernelRegistry =
    habana::KernelRegistry()
        .add(
            "hpu::habana_cast_to_fp8_te",
            KERNEL_FN_GLOBAL(habana::LazyCastToFp8))
        .add(
            "hpu::habana_cast_from_fp8",
            KERNEL_FN_GLOBAL(habana::LazyCastFromFp8))
        .add("hpu::habana_fp8_gemm", KERNEL_FN_GLOBAL(habana::LazyFp8Gemm))
        .add(
            "hpu::habana_fp8_transpose",
            KERNEL_FN_GLOBAL(habana::LazyFp8Transpose));
