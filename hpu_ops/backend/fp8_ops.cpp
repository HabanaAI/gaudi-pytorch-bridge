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

#include "hpu_ops/fp8_ops.h"
#include "habana_kernels/random_gen_kernels.h"
#include "hpu_ops/backend/reduction_template.h"
#include "hpu_ops/common/batched_matmul_output_shape.h"
#include "hpu_ops/custom_op_outshape.h"

namespace sh = synapse_helpers;

namespace habana {

namespace {
auto GetFp8Dtypes(const at::ScalarType& dtype) {
  auto syn_dtype = dtype == at::ScalarType::Char
      ? fp8_syn_type
      : habana_helpers::pytorch_to_synapse_type(dtype);

  return std::make_pair(dtype, syn_dtype);
}

auto GetFp8Dtypes(const at::IValue& dtype) {
  return GetFp8Dtypes(
      dtype.toOptional<at::ScalarType>().value_or(at::ScalarType::Char));
}

void ValidateScaleShape(
    const c10::IValue& scale,
    const c10::IValue& scale_shape) {
  if (scale.isNone() or scale.isDouble() or scale_shape.isNone()) {
    return;
  }

  int64_t scale_numel = 0;
  int64_t shape_numel = 1;
  if (scale.isTensor()) {
    scale_numel = scale.toTensor().numel();
  } else if (scale.isDoubleList()) {
    scale_numel = scale.toDoubleVector().size();
  }
  for (auto d : scale_shape.toIntVector()) {
    shape_numel *= d;
  }
  TORCH_CHECK(
      scale_numel == shape_numel,
      "Number of scale elements (",
      scale_numel,
      ") is not equal to number of scale_shape elements (",
      shape_numel,
      ").");
}

void HandleScaleTensor(
    habana::OpBackend* op,
    sh::graph& graph,
    const at::Tensor& scale,
    synTensor syn_scale,
    std::vector<sh::tensor>& maybe_reshaped_scale,
    std::vector<synTensor>& syn_inputs,
    const c10::IValue& scale_shape_ival = c10::IValue{}) {
  if (scale.numel() > 1 and not scale_shape_ival.isNone() and
      scale.sizes().vec() != scale_shape_ival.toIntVector()) {
    maybe_reshaped_scale.emplace_back(OpBackend::BuildReshape(
        op,
        graph,
        syn_scale,
        scale_shape_ival.toIntVector(),
        scale.scalar_type()));
    syn_inputs.push_back(maybe_reshaped_scale.back().get());
  } else {
    syn_inputs.push_back(syn_scale);
  }
}

void HandleScaleScalar(
    habana::OpBackend* op,
    sh::graph& graph,
    const c10::IValue& scale,
    const int device_id,
    std::vector<sh::tensor>& maybe_const_scale,
    std::vector<synTensor>& syn_inputs,
    const c10::IValue& scale_shape_ival = c10::IValue{}) {
  if (scale.isDouble()) {
    maybe_const_scale.emplace_back(
        op->BuildConstantTensor(op, graph, scale.toDouble()));
    syn_inputs.push_back(maybe_const_scale.back().get());
  } else if (scale.isDoubleList() and not op->isOutputInfMode()) {
    maybe_const_scale.emplace_back(op->AllocateConstantSynapseTensor(
        graph,
        device_id,
        scale.toDoubleVector(),
        scale_shape_ival.isNone() ? at::OptionalIntArrayRef{}
                                  : scale_shape_ival.toIntVector()));
    syn_inputs.push_back(maybe_const_scale.back().get());
  } else {
    syn_inputs.push_back(nullptr);
  }
}
} // namespace

ns_CastKernel::Params GetCastParams(
    const bool stochastic,
    const at::ScalarType& from_dtype,
    const at::ScalarType& to_dtype) {
  ns_CastKernel::Params params{};
  if (stochastic) {
    const bool is_sftz_available = is_sr_sftz and
        from_dtype == at::ScalarType::BFloat16 and
        to_dtype == at::ScalarType::Float8_e5m2;
    params.round_mode = is_sftz_available ? CAST_ROUND_SFTZ : CAST_ROUND_SR;
  } else {
    params.round_mode = CAST_ROUND_HALF_NE;
  }
  return params;
}

/********** CastToFp8 **********/

CastToFp8::CastToFp8(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "cast_to_fp8", scalar_type, {}, {}, {}, true) {
  SetNumOutTensors(2);
}

void CastToFp8::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto scale = stack[1].toOptional<torch::Tensor>().value_or(torch::Tensor());
  bool stochastic_rounding = stack[2].toBool();
  auto src_type = self.scalar_type();
  auto sizes = self.sizes();
  auto out = stack_tensor(stack, 3);
  auto amax = stack_tensor(stack, 4);
  auto [dst_type, dst_syn_type] = GetFp8Dtypes(out.scalar_type());

  bool is_amax = amax.numel() != 0;

  TORCH_CHECK(
      src_type == at::ScalarType::Float or src_type == at::ScalarType::BFloat16,
      "CastToFp8 input must be of float or bfloat16 dtype.");
  TORCH_CHECK(
      sizes == out.sizes(), "Input and output must have the same shape");

  std::string guid = src_type == at::ScalarType::Float ? "convert_to_fp8_f32"
                                                       : "convert_to_fp8_bf16";

  auto params = GetCastParams(stochastic_rounding, src_type, dst_type);

  std::vector<synTensor> syn_inputs{syn_in(0)};
  if (scale.defined()) {
    syn_inputs.push_back(syn_in(1));
  }
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {sizes, dst_type, 0, DATA_TENSOR, dst_syn_type}};
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

/********** CastToFp8V2 **********/

sizes_vec CastToFp8V2OutputShape(const at::Stack& stack) {
  auto input_sv = stack[0].toTensor().sizes().vec();
  bool is_amax = stack[3].toBool();
  std::vector<int64_t> amax_shape{};
  if (not is_amax) {
    amax_shape.push_back(0);
  }
  return {input_sv, amax_shape};
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
}

void CastToFp8V2::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto scale = stack[1];
  bool stochastic_rounding = stack[2].toBool();
  bool is_amax = stack[3].toBool();
  auto src_type = self.scalar_type();
  auto [dst_type, dst_syn_type] = GetFp8Dtypes(stack[4]);
  auto scale_shape = stack[5];

  auto is_fp8_input = src_type == at::ScalarType::Float8_e5m2 or
      src_type == at::ScalarType::Float8_e4m3fn;
  TORCH_CHECK(
      is_fp8_input or src_type == at::ScalarType::Float or
          src_type == at::ScalarType::BFloat16,
      "CastToFp8V2 input dtype must be one of [float, bfloat16, float8_e5m2, float8_e4m3fn].");
  if (is_fp8_input) {
    TORCH_CHECK(!is_amax, "CastToFp8V2 must have no amax for float8.");
    TORCH_CHECK(
        src_type == dst_type,
        "CastToFp8V2 input and output must have the same dtype for float8, but are ",
        src_type,
        " and ",
        dst_type);
  }

  ValidateScaleShape(scale, scale_shape);

  auto guid = get_guid_with_precision("convert_to_fp8", src_type);

  auto out_shapes = CastToFp8V2OutputShape(stack);
  std::vector<synTensor> syn_inputs{syn_in(0)};
  std::vector<sh::tensor> adjusted_scale;
  if (scale.isTensor()) {
    HandleScaleTensor(
        this,
        graph,
        scale.toTensor(),
        syn_in(1),
        adjusted_scale,
        syn_inputs,
        scale_shape);
  } else {
    HandleScaleScalar(
        this,
        graph,
        scale,
        p_context_->device_id_,
        adjusted_scale,
        syn_inputs,
        scale_shape);
  }
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {out_shapes[0], dst_type, 0, DATA_TENSOR, dst_syn_type}};
  if (is_amax) {
    output_attrs.push_back({out_shapes[1], at::ScalarType::Float, 1});
  }

  auto params = GetCastParams(stochastic_rounding, src_type, dst_type);

  auto casted = OpBackend::BuildNode(
      this, graph, {guid, syn_inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(casted[0]);
  if (is_amax) {
    syn_out(1) = std::move(casted[1]);
  }
}

/********** Fp8CastTranspose **********/

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

void Fp8CastTranspose::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto src_type = self.scalar_type();
  auto scale = stack[1].toOptional<torch::Tensor>().value_or(torch::Tensor());
  bool stochastic_rounding = stack[2].toBool();
  auto sizes = self.sizes();
  auto out = stack_tensor(stack, 3);
  auto transposed = stack_tensor(stack, 4);
  auto amax = stack_tensor(stack, 5);
  auto [dst_type, dst_syn_type] = GetFp8Dtypes(out.scalar_type());

  bool is_amax = amax.numel() != 0;

  TORCH_CHECK(
      src_type == at::ScalarType::Float or src_type == at::ScalarType::BFloat16,
      "Fp8CastTranspose input must be of float or bfloat16 dtype.");
  TORCH_CHECK(
      sizes == out.sizes(), "Input and output must have the same shape");

  std::string guid =
      get_guid_with_precision("convert_to_fp8_transpose", src_type);

  auto params = GetCastParams(stochastic_rounding, src_type, dst_type);

  std::vector<synTensor> syn_inputs{syn_in(0)};
  if (scale.defined()) {
    syn_inputs.push_back(syn_in(1));
  }
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {sizes, dst_type, 0, DATA_TENSOR, dst_syn_type},
      {transposed.sizes(), dst_type, 1, DATA_TENSOR, dst_syn_type}};
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

/********** Fp8CastTransposeBgrad **********/

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

void Fp8CastTransposeBgrad::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto src_type = self.scalar_type();
  auto scale = stack[1].toOptional<torch::Tensor>().value_or(torch::Tensor());
  bool stochastic_rounding = stack[2].toBool();
  auto sizes = self.sizes();
  auto out = stack_tensor(stack, 3);
  auto transposed = stack_tensor(stack, 4);
  auto bgrad = stack_tensor(stack, 5);
  auto amax = stack_tensor(stack, 6);
  auto [dst_type, dst_syn_type] = GetFp8Dtypes(out.scalar_type());

  bool is_amax = amax.numel() != 0;

  TORCH_CHECK(
      src_type == at::ScalarType::Float or src_type == at::ScalarType::BFloat16,
      "Fp8CastTransposeBgrad input must be of float or bfloat16 dtype.");
  TORCH_CHECK(
      sizes == out.sizes(), "Input and output must have the same shape");

  std::string guid =
      get_guid_with_precision("convert_to_fp8_transpose_bgrad", src_type);

  auto params = GetCastParams(stochastic_rounding, src_type, dst_type);

  std::vector<synTensor> syn_inputs{syn_in(0)};
  if (scale.defined()) {
    syn_inputs.push_back(syn_in(1));
  }
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {sizes, dst_type, 0, DATA_TENSOR, dst_syn_type},
      {transposed.sizes(), dst_type, 1, DATA_TENSOR, dst_syn_type},
      {bgrad.sizes(), src_type, 2}};
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

/********** Fp8CastTransposeBgradDgelu **********/

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
    sh::graph& graph,
    const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto src_type = self.scalar_type();
  auto scale = stack[2].toOptional<torch::Tensor>().value_or(torch::Tensor());
  auto retain = stack[3].toOptional<torch::Tensor>().value_or(torch::Tensor());
  bool stochastic_rounding = stack[4].toBool();
  auto sizes = self.sizes();
  auto out = stack_tensor(stack, 5);
  auto transposed = stack_tensor(stack, 6);
  auto bgrad = stack_tensor(stack, 7);
  auto amax = stack_tensor(stack, 8);
  auto [dst_type, dst_syn_type] = GetFp8Dtypes(out.scalar_type());

  bool is_amax = amax.numel() != 0;

  TORCH_CHECK(
      src_type == at::ScalarType::Float or src_type == at::ScalarType::BFloat16,
      "Fp8CastTransposeBgradDgelu input must be of float or bfloat16 dtype.");
  TORCH_CHECK(
      sizes == out.sizes(), "Input and output must have the same shape");

  std::string guid =
      get_guid_with_precision("convert_to_fp8_transpose_bgrad_dgelu", src_type);

  auto params = GetCastParams(stochastic_rounding, src_type, dst_type);

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
      {sizes, dst_type, 0, DATA_TENSOR, dst_syn_type},
      {transposed.sizes(), dst_type, 1, DATA_TENSOR, dst_syn_type},
      {bgrad.sizes(), src_type, 2}};
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

/********** CastFromFp8 **********/

CastFromFp8::CastFromFp8(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "cast_from_fp8", scalar_type, {0}, {}, {}, false) {}

void CastFromFp8::AddNode(sh::graph& graph, const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 4, "CastFromFp8 must have 4 input arguments");

  auto self = stack_tensor(stack, 0);
  auto scale = stack[1];
  auto dst_type = stack[2].toScalarType();
  auto scale_shape = stack[3];
  auto sizes = self.sizes();

  ValidateScaleShape(scale, scale_shape);

  TORCH_CHECK(
      dst_type == at::ScalarType::Float or dst_type == at::ScalarType::BFloat16,
      "CastFromFp8 output dtype must be equal to float or bfloat16.");

  std::string guid = dst_type == at::ScalarType::Float
      ? "convert_from_fp8_f32"
      : "convert_from_fp8_bf16";

  std::vector<synTensor> syn_inputs{syn_in(0)};
  std::vector<sh::tensor> adjusted_scale;
  if (scale.isTensor()) {
    HandleScaleTensor(
        this,
        graph,
        scale.toTensor(),
        syn_in(1),
        adjusted_scale,
        syn_inputs,
        scale_shape);
  } else {
    HandleScaleScalar(
        this,
        graph,
        scale,
        p_context_->device_id_,
        adjusted_scale,
        syn_inputs,
        scale_shape);
  }

  auto casted = OpBackend::BuildNode(
      this, graph, {guid, syn_inputs, {{sizes, dst_type, 0}}});

  syn_out(0) = std::move(casted[0]);
}

/********** Fp8Dropout **********/

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

void Fp8Dropout::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(stack, "Fp8Dropout::AddNode");
  auto self = getNextInput<TensorsPair>(stackGetter);
  double p = getNextInput<double>(stackGetter);
  auto scaleOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  bool stochastic_rounding = getNextInput<bool>(stackGetter);
  bool is_amax = getNextInput<bool>(stackGetter);
  auto [dst_type, dst_syn_type] = GetFp8Dtypes(stack[5]);
  auto sizes = self.pt_t.sizes().vec();
  std::vector<int64_t> amax_size{1};
  auto src_type = self.pt_t.scalar_type();

  TORCH_CHECK(
      src_type == at::ScalarType::Float or src_type == at::ScalarType::BFloat16,
      "Fp8Dropout input must be of float or bfloat16 dtype.");

  std::string guid = get_guid_with_precision("dropout_fp8", src_type);

  ns_DropoutFp8::Params params{};
  params.round_mode = stochastic_rounding ? CAST_ROUND_SR : CAST_ROUND_HALF_NE;
  params.ratio = static_cast<float>(p);
  params.seed = habana::get_seed_hpu(c10::nullopt);

  std::vector<synTensor> syn_inputs{self.syn_t};
  if (scaleOpt) {
    syn_inputs.push_back(scaleOpt->syn_t);
  }
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {sizes, dst_type, 0, DATA_TENSOR, dst_syn_type},
      {sizes, at::ScalarType::Char, 1}};
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

/********** Fp8Gelu **********/

Fp8Gelu::Fp8Gelu(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_gelu", scalar_type, {}, {}, {}, true) {
  SetNumOutTensors(3);
}

void Fp8Gelu::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto scale = stack[1].toOptional<torch::Tensor>().value_or(torch::Tensor());
  bool stochastic_rounding = stack[2].toBool();
  auto src_type = self.scalar_type();
  auto sizes = self.sizes();
  auto out = stack_tensor(stack, 3);
  auto retain = stack_tensor(stack, 4);
  auto amax = stack_tensor(stack, 5);
  auto [dst_type, dst_syn_type] = GetFp8Dtypes(out.scalar_type());

  bool is_amax = amax.numel() != 0;

  TORCH_CHECK(
      src_type == at::ScalarType::Float or src_type == at::ScalarType::BFloat16,
      "Fp8Gelu input must be of float or bfloat16 dtype.");
  TORCH_CHECK(
      sizes == out.sizes(), "Input and output must have the same shape");

  std::string guid =
      src_type == at::ScalarType::Float ? "fp8_gelu_f32" : "fp8_gelu_bf16";

  auto params = GetCastParams(stochastic_rounding, src_type, dst_type);

  std::vector<synTensor> syn_inputs{syn_in(0)};
  if (scale.defined()) {
    syn_inputs.push_back(syn_in(1));
  }
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {sizes, dst_type, 0, DATA_TENSOR, dst_syn_type},
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

/********** Fp8GeluV2 **********/

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

void Fp8GeluV2::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(stack, "Fp8GeluV2::AddNode");
  auto self = getNextInput<TensorsPair>(stackGetter);
  auto scaleOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  bool stochastic_rounding = getNextInput<bool>(stackGetter);
  bool is_amax = getNextInput<bool>(stackGetter);
  auto [dst_type, dst_syn_type] = GetFp8Dtypes(stack[4]);
  auto src_dtype = self.pt_t.scalar_type();

  TORCH_CHECK(
      src_dtype == at::ScalarType::Float or
          src_dtype == at::ScalarType::BFloat16,
      "Fp8GeluV2 input must be of float or bfloat16 dtype.");

  std::string guid = get_guid_with_precision("fp8_gelu", src_dtype);
  auto params = GetCastParams(stochastic_rounding, src_dtype, dst_type);
  auto output_shapes = Fp8GeluV2OutputShape(stack);

  std::vector<synTensor> syn_inputs{self.syn_t};
  if (scaleOpt) {
    syn_inputs.push_back(scaleOpt->syn_t);
  }
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {output_shapes[0], dst_type, 0, DATA_TENSOR, dst_syn_type},
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

/********** Fp8BgradDgelu **********/

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

void Fp8BgradDgelu::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(stack, "Fp8eBgradDgelu::AddNode");
  auto grad = getNextInput<TensorsPair>(stackGetter);
  auto input = getNextInput<TensorsPair>(stackGetter);
  auto scaleOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto retainOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto stochastic_rounding = getNextInput<bool>(stackGetter);
  auto is_amax = getNextInput<bool>(stackGetter);
  auto [dst_type, dst_syn_type] = GetFp8Dtypes(stack[6]);
  auto src_dtype = ScalarType();

  auto out_sizes = Fp8BgradDgeluOutputShape(stack);

  TORCH_CHECK(
      src_dtype == at::ScalarType::Float or
          src_dtype == at::ScalarType::BFloat16,
      "Fp8BgradDgelu input must be of float or bfloat16 dtype.");

  std::string guid = get_guid_with_precision("fp8_bgrad_dgelu", src_dtype);

  auto params = GetCastParams(stochastic_rounding, src_dtype, dst_type);

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
      {out_sizes[0], dst_type, 0, DATA_TENSOR, dst_syn_type},
      {out_sizes[1], src_dtype, 1}};
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

/********** Fp8FastSoftmaxOutputShape **********/

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

void Fp8FastSoftmax::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(stack, "Fp8FastSoftmax::AddNode");
  auto input = getNextInput<TensorsPair>(stackGetter);
  auto mask = getNextInput<TensorsPair>(stackGetter);
  auto scale_opt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto softmax_scale = static_cast<float>(getNextInput<double>(stackGetter));
  auto stochastic_rounding = getNextInput<bool>(stackGetter);
  auto is_amax = getNextInput<bool>(stackGetter);
  auto [dst_type, dst_syn_type] = GetFp8Dtypes(stack[6]);
  auto src_dtype = ScalarType();

  TORCH_CHECK(
      src_dtype == at::ScalarType::Float or
          src_dtype == at::ScalarType::BFloat16,
      "Fp8FastSoftmax input must be of float or bfloat16 dtype.");

  auto out_sizes = Fp8FastSoftmaxOutputShape(stack);

  std::string guid = get_guid_with_precision("fp8_fast_softmax", src_dtype);

  // reuse params structure from LayerNormFp8
  ns_LayerNormFp8::Params params{};
  params.round_mode = stochastic_rounding ? CAST_ROUND_SR : CAST_ROUND_HALF_NE;
  params.eps = softmax_scale;

  std::vector<synTensor> syn_inputs{input.syn_t, mask.syn_t};
  if (scale_opt) {
    syn_inputs.push_back(scale_opt->syn_t);
  }
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {out_sizes[0], dst_type, 0, DATA_TENSOR, dst_syn_type}};
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

/********** Fp8Layernorm **********/

Fp8Layernorm::Fp8Layernorm(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_layernorm", scalar_type, {}, {}, {}, true) {
  SetNumOutTensors(4);
}

void Fp8Layernorm::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  float eps = static_cast<float>(stack[3].toDouble());
  auto scale = stack[4].toOptional<torch::Tensor>().value_or(torch::Tensor());
  bool stochastic_rounding = stack[5].toBool();
  auto sizes = self.sizes();
  auto out = stack_tensor(stack, 6);
  auto mean = stack_tensor(stack, 7);
  auto istd = stack_tensor(stack, 8);
  auto amax = stack_tensor(stack, 9);
  auto [dst_type, dst_syn_type] = GetFp8Dtypes(out.scalar_type());
  auto src_dtype = self.scalar_type();

  bool is_amax = amax.numel() != 0;

  TORCH_CHECK(
      src_dtype == at::ScalarType::Float or
          src_dtype == at::ScalarType::BFloat16,
      "Fp8Layernorm input must be of float or bfloat16 dtype.");
  TORCH_CHECK(
      sizes == out.sizes(), "Input and output must have the same shape");

  std::string guid = get_guid_with_precision("layer_norm_fp8_fwd", src_dtype);

  ns_LayerNormFp8::Params params{};
  params.round_mode = stochastic_rounding ? CAST_ROUND_SR : CAST_ROUND_HALF_NE;
  params.eps = eps;

  std::vector<synTensor> syn_inputs{syn_in(0), syn_in(1), syn_in(2)};
  if (scale.defined()) {
    syn_inputs.push_back(syn_in(3));
  }
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {sizes, dst_type, 0, DATA_TENSOR, dst_syn_type},
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

/********** Fp8Gemm **********/

Fp8Gemm::Fp8Gemm(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_gemm", scalar_type, {}, {}, {}, true) {}

void Fp8Gemm::AddNode(sh::graph& graph, const at::Stack& stack) {
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

  std::vector<int64_t> out_shape;
  try {
    out_shape = getBatchMatmulOutShape(
        A.pt_t.sizes(), B.pt_t.sizes(), trans_A, trans_B);
  } catch (const std::invalid_argument& e) {
    TORCH_CHECK(false, e.what());
  }

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

/********** Fp8GemmV2 **********/

sym_sizes_vec fp8_gemm_v2_out_shape(
    const std::vector<at::Tensor>& inputs,
    const std::vector<int64_t>& params) {
  TORCH_CHECK(inputs.size() == 2);
  TORCH_CHECK(params.size() == 2);
  return {getBatchMatmulOutShape(
      inputs[0].sym_sizes(),
      inputs[1].sym_sizes(),
      static_cast<bool>(params[0]),
      static_cast<bool>(params[1]))};
}

REGISTER_CUSTOM_OP_OUTSHAPE_FUN(fp8_gemm_v2, fp8_gemm_v2_out_shape);

sizes_vec Fp8GemmV2OutputShape(const at::Stack& stack) {
  auto A = stack_tensor(stack, 0);
  bool trans_A = stack[1].toBool();
  auto B = stack_tensor(stack, 2);
  bool trans_B = stack[3].toBool();

  try {
    return {getBatchMatmulOutShape(A.sizes(), B.sizes(), trans_A, trans_B)};
  } catch (const std::invalid_argument& e) {
    TORCH_CHECK(false, e.what());
    return {};
  }
}

Fp8GemmV2::Fp8GemmV2(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_gemm_v2", scalar_type, {0}, {}, {}, false) {
  SetComputeOutputShapes(Fp8GemmV2OutputShape);
}

void Fp8GemmV2::AddNode(sh::graph& graph, const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 11, "Fp8GemmV2 must have 10 input arguments");

  StackGetter stackGetter(stack, "Fp8Gemm::AddNode");
  auto A = getNextInput<TensorsPair>(stackGetter);
  bool trans_A = getNextInput<bool>(stackGetter);
  auto B = getNextInput<TensorsPair>(stackGetter);
  bool trans_B = getNextInput<bool>(stackGetter);
  auto DOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto out_type = getNextInput<c10::ScalarType>(stackGetter);
  auto scaleAOpt =
      getNextInput<std::variant<TensorsPair, c10::IValue>>(stackGetter);
  auto scaleBOpt =
      getNextInput<std::variant<TensorsPair, c10::IValue>>(stackGetter);
  auto biasOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  bool accumulate = getNextInput<bool>(stackGetter);
  auto scale_shape = getNextInput<c10::IValue>(stackGetter);

  std::string guid = get_guid_with_precision("fp8_gemm", out_type);

  std::vector<synTensor> syn_inputs = {A.syn_t, B.syn_t};
  std::vector<sh::tensor> adjusted_scale;

  if (std::holds_alternative<TensorsPair>(scaleAOpt)) {
    auto scaleA = std::get<TensorsPair>(scaleAOpt);
    HandleScaleTensor(
        this, graph, scaleA.pt_t, scaleA.syn_t, adjusted_scale, syn_inputs);
  } else {
    HandleScaleScalar(
        this,
        graph,
        std::get<c10::IValue>(scaleAOpt),
        p_context_->device_id_,
        adjusted_scale,
        syn_inputs);
  }

  if (std::holds_alternative<TensorsPair>(scaleBOpt)) {
    auto scaleB = std::get<TensorsPair>(scaleBOpt);
    ValidateScaleShape(scaleB.pt_t, scale_shape);
    HandleScaleTensor(
        this,
        graph,
        scaleB.pt_t,
        scaleB.syn_t,
        adjusted_scale,
        syn_inputs,
        scale_shape);
  } else {
    auto scaleB = std::get<c10::IValue>(scaleBOpt);
    ValidateScaleShape(scaleB, scale_shape);
    HandleScaleScalar(
        this,
        graph,
        scaleB,
        p_context_->device_id_,
        adjusted_scale,
        syn_inputs,
        scale_shape);
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
  } else {
    syn_inputs.push_back(nullptr);
  }

  // GC pass FUSE_CONVERT_MME inserts this last input, but it needs
  // it to be explicitly filled with nullptr before.
  syn_inputs.push_back(nullptr);

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

/********** Fp8Transpose **********/

Fp8Transpose::Fp8Transpose(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_transpose", scalar_type, {}, {}, {}, true) {}

void Fp8Transpose::AddNode(sh::graph& graph, const at::Stack& stack) {
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

/********** Fp8Permute **********/

Fp8Permute::Fp8Permute(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_permute", scalar_type, {}, {}, {}, true) {}

void Fp8Permute::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto dims = stack[1].toIntList().vec();
  auto out = stack_tensor(stack, 2);
  auto dims_size = dims.size();

  synTransposeParams params{};
  params.tensorDim = dims_size;
  for (size_t i = 0; i < dims_size; i++) {
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

/********** Fp8Reshape **********/

sizes_vec Fp8ReshapeOutputShape(const at::Stack& stack) {
  return {stack[1].toIntList().vec()};
}

Fp8Reshape::Fp8Reshape(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_reshape", scalar_type, {0}, {}, {}, false) {
  SetComputeOutputShapes(Fp8ReshapeOutputShape);
}

void Fp8Reshape::AddNode(sh::graph& graph, const at::Stack& stack) {
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

/********** Fp8Copy_ **********/

Fp8Copy_::Fp8Copy_(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_copy_", scalar_type, {}, {0}, {}, false) {}

void Fp8Copy_::AddNode(sh::graph& graph, const at::Stack& stack) {
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

/********** Fp8KvReorder **********/

Fp8KvReorder::Fp8KvReorder(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_kv_reorder", scalar_type, {}, {0}, {}, false) {}

void Fp8KvReorder::AddNode(sh::graph& graph, const at::Stack& stack) {
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

/********** Fp8IndexCopy_ **********/

Fp8IndexCopy_::Fp8IndexCopy_(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_index_copy_", scalar_type, {}, {0}, {}, false) {
}

void Fp8IndexCopy_::AddNode(sh::graph& graph, const at::Stack& stack) {
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

/********** Fp8RepeatV2 **********/

template <class RT>
RT Fp8RepeatV2OutputShapeCommon(
    const at::Tensor& self,
    c10::IntArrayRef repeats) {
  int64_t num_new_dimensions = repeats.size() - self.dim();
  using OutVecT = typename RT::value_type;
  OutVecT padded_size(num_new_dimensions, 1);
  if constexpr (std::is_same_v<RT, sym_sizes_vec>)
    padded_size.insert(
        padded_size.end(), self.sym_sizes().begin(), self.sym_sizes().end());
  else
    padded_size.insert(
        padded_size.end(), self.sizes().begin(), self.sizes().end());
  OutVecT outshape(repeats.size());
  for (size_t i = 0; i < repeats.size(); ++i) {
    outshape[i] = padded_size[i] * repeats[i];
  }

  return {outshape};
}

sym_sizes_vec fp8_repeat_v2_out_shape(
    const std::vector<at::Tensor>& inputs,
    const std::vector<int64_t>& params) {
  TORCH_CHECK(inputs.size() == 1);
  return Fp8RepeatV2OutputShapeCommon<sym_sizes_vec>(inputs[0], params);
}

REGISTER_CUSTOM_OP_OUTSHAPE_FUN(fp8_repeat_v2, fp8_repeat_v2_out_shape);

sizes_vec Fp8RepeatV2OutputShape(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto repeats = stack[1].toIntVector();
  return Fp8RepeatV2OutputShapeCommon<sizes_vec>(self, repeats);
}

Fp8RepeatV2::Fp8RepeatV2(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "fp8_repeat_v2", scalar_type, {0}, {}, {}, false) {
  SetComputeOutputShapes(Fp8RepeatV2OutputShape);
}

void Fp8RepeatV2::AddNode(sh::graph& graph, const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 2, "Fp8RepeatV2 must have 2 input arguments");

  StackGetter stackGetter(stack, "Fp8RepeatV2::AddNode");
  auto self = getNextInput<TensorsPair>(stackGetter);
  auto repeats = getNextInput<std::vector<int64_t>>(stackGetter);

  std::string guid_suffix = fp8_syn_type == syn_type_fp8_143 ? "hf8" : "f8";

  synTensor self_reshaped_st = self.syn_t;
  std::optional<sh::tensor> self_reshaped_storage;
  if (static_cast<int64_t>(repeats.size()) > self.pt_t.ndimension()) {
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
  for (size_t i = 0; i < repeats.size(); ++i) {
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

/********** Fp8IndexSelectV2 **********/

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

void Fp8IndexSelectV2::AddNode(sh::graph& graph, const at::Stack& stack) {
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

/********** InPlaceInterleave **********/

struct InPlaceInterleave : InPlaceInterleaveCommon {
  InPlaceInterleave(int device_id, c10::ScalarType scalar_type)
      : InPlaceInterleaveCommon(
            device_id,
            "in_place_interleave",
            scalar_type,
            {0},
            {},
            {},
            false) {}
};

struct InPlaceInterleave_ : InPlaceInterleaveCommon {
  InPlaceInterleave_(int device_id, c10::ScalarType scalar_type)
      : InPlaceInterleaveCommon(
            device_id,
            "in_place_interleave_",
            scalar_type,
            {},
            {0},
            {},
            false) {}
};

void InPlaceInterleaveCommon::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(
      stack.size() == 1, "InPlaceInterleave must have 1 input argument");

  StackGetter stackGetter(stack, "InPlaceInterleave::AddNode");
  auto self = getNextInput<TensorsPair>(stackGetter);
  auto shape = self.pt_t.sizes().vec();
  auto [dst_type, dst_syn_type] = GetFp8Dtypes(self.pt_t.scalar_type());
  TORCH_CHECK(shape.size() == 4, "Input has to be a 4D tensor.");
  TORCH_CHECK(shape[0] % 4 == 0, "Batch size has to be a multiple of 4.");

  std::string guid_suffix =
      sh::graph::name_suffix_from_type(dst_syn_type).data();
  auto output = BuildNode(
      this,
      graph,
      {"in_place_interleave_fwd_" + guid_suffix,
       {self.syn_t},
       {{shape, dst_type, 0}}});

  syn_out(0) = std::move(output[0]);
}

/********** Conv2dFp8 **********/

template <class DimT>
DimT ComputeConv2dOutputDim(
    const DimT input_dim,
    const int64_t padding,
    const int64_t dilation,
    const DimT kernel_size,
    const int64_t stride) {
  return (input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride +
      1;
}

template <class DimT>
std::vector<DimT> ComputeConv2dOutputSize(
    c10::ArrayRef<DimT> shape_in,
    c10::ArrayRef<DimT> shape_wt,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    c10::IntArrayRef dilation) {
  std::vector<DimT> out_shape{shape_in[0], shape_wt[0]};
  for (int i = 0; i < 2; ++i) {
    out_shape.emplace_back(ComputeConv2dOutputDim(
        shape_in[i + 2], padding[i], dilation[i], shape_wt[i + 2], stride[i]));
  }
  return out_shape;
}

sym_sizes_vec conv2d_fp8_out_shape(
    const std::vector<at::Tensor>& inputs,
    const std::vector<int64_t>& params) {
  TORCH_CHECK(inputs.size() == 2);
  TORCH_CHECK(params.size() == 6);
  return {ComputeConv2dOutputSize(
      inputs[0].sym_sizes(),
      inputs[1].sym_sizes(),
      c10::IntArrayRef(params).slice(0, 2),
      c10::IntArrayRef(params).slice(2, 2),
      c10::IntArrayRef(params).slice(4, 2))};
}

REGISTER_CUSTOM_OP_OUTSHAPE_FUN(conv2d_fp8, conv2d_fp8_out_shape);

sizes_vec Conv2dFp8OutputShape(const at::Stack& stack) {
  auto shape_in = stack_tensor(stack, 0).sizes();
  auto shape_wt = stack_tensor(stack, 1).sizes();
  const auto stride = stack[3].toIntList().vec();
  const auto padding = stack[4].toIntList().vec();
  const auto dilation = stack[5].toIntList().vec();

  return {
      ComputeConv2dOutputSize(shape_in, shape_wt, stride, padding, dilation)};
}

static synConvolutionParams FillConv2dFp8Params(
    at::IntArrayRef weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  synConvolutionParams params{};
  params.dH = stride[0];
  params.dW = stride[1];
  params.kH = weight[2];
  params.kW = weight[3];
  params.dilH = dilation[0];
  params.dilW = dilation[1];
  params.setPadT(padding[0]);
  params.setPadB(padding[0]);
  params.setPadL(padding[1]);
  params.setPadR(padding[1]);
  params.nGroups = groups;

  return params;
}

Conv2dFp8::Conv2dFp8(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "conv2d_fp8", scalar_type, {0}, {}, {}, false) {
  SetComputeOutputShapes(Conv2dFp8OutputShape);
}

void Conv2dFp8::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(stack, "Conv2dFp8::AddNode");
  auto input = getNextInput<TensorsPair>(stackGetter);
  auto weight = getNextInput<TensorsPair>(stackGetter);
  auto bias_opt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto stride = getNextInput<std::vector<int64_t>>(stackGetter);
  auto padding = getNextInput<std::vector<int64_t>>(stackGetter);
  auto dilation = getNextInput<std::vector<int64_t>>(stackGetter);
  auto groups = getNextInput<int>(stackGetter);
  auto out_dtype = getNextInput<c10::optional<c10::ScalarType>>(stackGetter)
                       .value_or(at::ScalarType::BFloat16);
  auto scale_input_opt =
      getNextInput<std::variant<TensorsPair, c10::IValue>>(stackGetter);
  auto scale_weight_opt =
      getNextInput<std::variant<TensorsPair, c10::IValue>>(stackGetter);

  TORCH_CHECK(
      input.pt_t.dim() == 4 and weight.pt_t.dim() == 4,
      "Input and weight must be 4D tensors");
  if (bias_opt) {
    TORCH_CHECK(
        bias_opt->pt_t.dim() == 1 and
            bias_opt->pt_t.sizes()[0] == weight.pt_t.sizes()[0],
        "Bias must be 1D tensor with size equal to weight dim0");
  }

  SetSynapseLayouts(
      {sh::layouts::SynapseLayoutFormat::WHCN,
       sh::layouts::SynapseLayoutFormat::SRCK,
       sh::layouts::SynapseLayoutFormat::DONT_CARE,
       sh::layouts::SynapseLayoutFormat::DONT_CARE,
       sh::layouts::SynapseLayoutFormat::DONT_CARE,
       sh::layouts::SynapseLayoutFormat::DONT_CARE},
      {sh::layouts::SynapseLayoutFormat::WHCN});

  auto out_shape = Conv2dFp8OutputShape(stack)[0];
  auto params = FillConv2dFp8Params(
      weight.pt_t.sizes().vec(), stride, padding, dilation, groups);

  std::vector<synTensor> syn_inputs{input.syn_t, weight.syn_t};
  syn_inputs.push_back(bias_opt ? bias_opt->syn_t : nullptr);
  std::vector<sh::tensor> adjusted_scale;

  for (int i = 0; i < 2; ++i) {
    const auto& scale = (i == 0) ? scale_input_opt : scale_weight_opt;
    if (std::holds_alternative<TensorsPair>(scale)) {
      auto scale_tp = std::get<TensorsPair>(scale);
      TORCH_CHECK(
          scale_tp.pt_t.numel() == 1,
          "Multi-element scale tensors are not supported yet.");
      syn_inputs.push_back(scale_tp.syn_t);
    } else {
      HandleScaleScalar(
          this,
          graph,
          std::get<c10::IValue>(scale),
          p_context_->device_id_,
          adjusted_scale,
          syn_inputs);
    }
  }

  // GC pass FUSE_CONVERT_MME inserts this last input, but it needs
  // it to be explicitly filled with nullptr before.
  syn_inputs.push_back(nullptr);

  auto conv = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("conv2d_fp8", out_dtype),
       std::move(syn_inputs),
       {{out_shape, out_dtype, 0}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(conv[0]);
}

/********** SumFp8 **********/

static std::shared_ptr<void> FillSumFp8Params(
    int ndims,
    at::IntArrayRef dims,
    bool keepdim,
    size_t& size) {
  PARAMS_STUB(ns_Reduction::ParamsV2);
  unsigned maskval = 0;
  for (size_t i = 0; i < dims.size(); ++i) {
    auto d = c10::maybe_wrap_dim(dims[i], ndims); // handling negative indices
    maskval = maskval | (1 << (ndims - d - 1)); // (ndims-i-1) is TPC order
  }

  params->reductionDimensionMask = maskval;
  params->keepDim = keepdim;
  return params;
}

sizes_vec SumFp8OutputShape(const at::Stack& stack) {
  auto dims = get_dims(stack, 1);
  at::DimVector shape = at::meta::get_reduction_shape(
      stack_tensor(stack, 0), dims, stack[2].toBool());
  return {std::vector<int64_t>(shape.begin(), shape.end())};
}

SumFp8::SumFp8(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "reduce_sum_multi_dim_fwd",
          scalar_type,
          {0},
          {},
          {},
          false) {
  SetComputeOutputShapes(SumFp8OutputShape);
}

void SumFp8::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto input = stack_tensor(stack, 0);
  auto dims = get_dims(stack, 1);
  auto keepdims = stack[2].toBool();
  auto out_dtype =
      stack[3].toOptional<at::ScalarType>().value_or(input.scalar_type());

  auto out_shape = SumFp8OutputShape(stack)[0];

  size_t size = 0;
  auto params = FillSumFp8Params(input.dim(), dims, keepdims, size);

  auto result = OpBackend::BuildNode(
      this,
      graph,
      {GetGuid(),
       {syn_in(0)},
       {{out_shape, out_dtype, 0}},
       params.get(),
       size});
  syn_out(0) = std::move(result[0]);
}

} // namespace habana

static const auto& CastKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::cast_to_fp8", KERNEL_FN_GLOBAL(habana::CastToFp8))
        .add("hpu::cast_to_fp8_v2", KERNEL_FN_GLOBAL(habana::CastToFp8V2))
        .add(
            "hpu::cast_to_fp8_v2.scalar",
            KERNEL_FN_GLOBAL(habana::CastToFp8V2))
        .add(
            "hpu::cast_to_fp8_v2.scalar_list",
            KERNEL_FN_GLOBAL(habana::CastToFp8V2))
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
        .add("hpu::cast_from_fp8.scalar", KERNEL_FN_GLOBAL(habana::CastFromFp8))
        .add(
            "hpu::cast_from_fp8.scalar_list",
            KERNEL_FN_GLOBAL(habana::CastFromFp8))
        .add("hpu::fp8_dropout", KERNEL_FN_GLOBAL(habana::Fp8Dropout))
        .add("hpu::fp8_gelu", KERNEL_FN_GLOBAL(habana::Fp8Gelu))
        .add("hpu::fp8_gelu_v2", KERNEL_FN_GLOBAL(habana::Fp8GeluV2))
        .add("hpu::fp8_bgrad_dgelu", KERNEL_FN_GLOBAL(habana::Fp8BgradDgelu))
        .add("hpu::fp8_fast_softmax", KERNEL_FN_GLOBAL(habana::Fp8FastSoftmax))
        .add("hpu::fp8_layernorm", KERNEL_FN_GLOBAL(habana::Fp8Layernorm))
        .add("hpu::fp8_gemm", KERNEL_FN_GLOBAL(habana::Fp8Gemm))
        .add("hpu::fp8_gemm_v2", KERNEL_FN_GLOBAL(habana::Fp8GemmV2))
        .add("hpu::fp8_gemm_v2.scalar", KERNEL_FN_GLOBAL(habana::Fp8GemmV2))
        .add(
            "hpu::fp8_gemm_v2.scalar_list",
            KERNEL_FN_GLOBAL(habana::Fp8GemmV2))
        .add("hpu::fp8_transpose", KERNEL_FN_GLOBAL(habana::Fp8Transpose))
        .add("hpu::fp8_permute", KERNEL_FN_GLOBAL(habana::Fp8Permute))
        .add("hpu::fp8_reshape", KERNEL_FN_GLOBAL(habana::Fp8Reshape))
        .add("hpu::fp8_copy_", KERNEL_FN_GLOBAL(habana::Fp8Copy_))
        .add("hpu::fp8_kv_reorder_", KERNEL_FN_GLOBAL(habana::Fp8KvReorder))
        .add("hpu::fp8_index_copy_", KERNEL_FN_GLOBAL(habana::Fp8IndexCopy_))
        .add("hpu::fp8_repeat_v2", KERNEL_FN_GLOBAL(habana::Fp8RepeatV2))
        .add(
            "hpu::fp8_index_select_v2",
            KERNEL_FN_GLOBAL(habana::Fp8IndexSelectV2))
        .add(
            "hpu::in_place_interleave_",
            KERNEL_FN_GLOBAL(habana::InPlaceInterleave_))
        .add(
            "hpu::in_place_interleave",
            KERNEL_FN_GLOBAL(habana::InPlaceInterleave))
        .add("hpu::conv2d_fp8", KERNEL_FN_GLOBAL(habana::Conv2dFp8))
        .add("hpu::conv2d_fp8.scalar", KERNEL_FN_GLOBAL(habana::Conv2dFp8))
        .add("hpu::sum_fp8", KERNEL_FN_GLOBAL(habana::SumFp8));
