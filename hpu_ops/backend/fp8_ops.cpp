/**
 * Copyright (c) 2023-2024 Intel Corporation
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

#include "hpu_ops/fp8_ops.h"
#include "generated/backend/cast_from_fp8.h"
#include "generated/backend/cast_to_fp8.h"
#include "generated/backend/cast_to_fp8_v2.h"
#include "generated/backend/conv2d_fp8.h"
#include "generated/backend/fp8_gemm.h"
#include "generated/backend/fp8_gemm_v2.h"
#include "habana_kernels/random_gen_kernels.h"
#include "hpu_ops/backend/reduction_template.h"
#include "hpu_ops/common/batched_matmul_output_shape.h"
#include "hpu_ops/common/convolution_gen.h"
#include "hpu_ops/custom_op_outshape.h"
#include "hpu_ops/fp8_utils.h"

namespace sh = synapse_helpers;

namespace habana {
namespace fp8 {
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
    const c10::IValue& scale_shape_ival) {
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
    const c10::IValue& scale_shape_ival) {
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

void HandleScale(
    habana::OpBackend* op,
    sh::graph& graph,
    const habana::VariantWrapper<TensorsPair, c10::IValue>& scaleOpt,
    const at::Tensor& input,
    bool isTranspose,
    std::vector<sh::tensor>& adjusted_scale,
    std::vector<synTensor>& syn_inputs,
    int deviceId,
    const c10::IValue& scale_shape) {
  if (scaleOpt.isTensorsPair()) {
    auto scale = scaleOpt.toTensorsPair();
    if (scale.pt_t.dim() == 2) {
      TORCH_CHECK(
          scale.pt_t.size(0) == 1 || scale.pt_t.size(1) == 1,
          "Scale tensor must be 1D or 2D with one of the dimensions being 1.");
    }

    auto sizes = input.sizes().vec();
    if (isTranspose) {
      TORCH_CHECK(
          sizes.size() >= 2,
          "Input tensor must have at least 2 dimensions to perform transpose operation.");
      std::swap(sizes[sizes.size() - 1], sizes[sizes.size() - 2]);
    }
    TORCH_CHECK(
        at::are_expandable(at::IntArrayRef(sizes), scale.pt_t.sizes()),
        "Input and its scale must be broadcastable. Got: ",
        at::IntArrayRef(sizes),
        " and ",
        scale.pt_t.sizes());

    ValidateScaleShape(scale.pt_t, scale_shape);
    HandleScaleTensor(
        op,
        graph,
        scale.pt_t,
        scale.syn_t,
        adjusted_scale,
        syn_inputs,
        scale_shape);
  } else {
    auto scale = scaleOpt.toIValue();
    ValidateScaleShape(scale, scale_shape);
    HandleScaleScalar(
        op, graph, scale, deviceId, adjusted_scale, syn_inputs, scale_shape);
  }
}

} // namespace fp8

using namespace habana::fp8;

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

void CastToFp8::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto scale = stack[1].toOptional<torch::Tensor>().value_or(torch::Tensor());
  bool stochastic_rounding = stack[2].toBool();
  auto src_type = self.scalar_type();
  auto sizes = self.sizes();
  auto out = stack_tensor(stack, 3);
  auto amax = stack_tensor(stack, 4);
  auto dst_type = out.scalar_type();

  bool is_amax = amax.numel() != 0;

  TORCH_CHECK(
      src_type == at::ScalarType::Float or src_type == at::ScalarType::BFloat16,
      "CastToFp8 input must be of float or bfloat16 dtype.");
  TORCH_CHECK(
      sizes == out.sizes(), "Input and output must have the same shape");

  auto guid = get_guid_with_precision("convert_to_fp8", src_type);

  auto params = GetCastParams(stochastic_rounding, src_type, dst_type);

  std::vector<synTensor> syn_inputs{syn_in(0)};
  if (scale.defined()) {
    syn_inputs.push_back(syn_in(1));
  }
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{{sizes, dst_type, 0}};
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

OutputMetaDataVector CastToFp8V2Meta(const at::Stack& stack) {
  bool is_amax = stack[3].toBool();
  std::vector<int64_t> amax_shape{};
  if (not is_amax) {
    amax_shape.push_back(0);
  }

  OutputMetaDataVector meta(2);
  meta[0].shape = stack[0].toTensor().sizes().vec();
  meta[0].dtype = stack[4].toScalarType();
  meta[1].shape = amax_shape;
  meta[1].dtype = c10::ScalarType::Float;
  return meta;
}

void CastToFp8V2::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto scale = stack[1];
  bool stochastic_rounding = stack[2].toBool();
  bool is_amax = stack[3].toBool();
  auto src_type = self.scalar_type();
  auto dst_type = stack[4].toScalarType();
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

  auto meta = CastToFp8V2Meta(stack);
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
      {meta[0].shape, meta[0].dtype, 0}};
  if (is_amax) {
    output_attrs.push_back({meta[1].shape, meta[1].dtype, 1});
  }

  auto params = GetCastParams(stochastic_rounding, src_type, dst_type);

  auto casted = OpBackend::BuildNode(
      this, graph, {guid, syn_inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(casted[0]);
  if (is_amax) {
    syn_out(1) = std::move(casted[1]);
  }
}

/********** CastFromFp8 **********/

OutputMetaDataVector CastFromFp8Meta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = stack[0].toTensor().sizes().vec();
  meta.dtype = stack[2].toScalarType();
  return {meta};
}

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

  auto guid = get_guid_with_precision("convert_from_fp8", dst_type);

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

/********** Fp8Gemm **********/

void Fp8Gemm::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "Fp8Gemm::AddNode");
  auto A = stackGetter.getNextInput<TensorsPair>();
  bool trans_A = stackGetter.getNextInput<bool>();
  auto B = stackGetter.getNextInput<TensorsPair>();
  bool trans_B = stackGetter.getNextInput<bool>();
  auto D = stackGetter.getNextInput<TensorsPair>();
  auto out_type = stackGetter.getNextInput<c10::ScalarType>();
  auto scaleAOpt = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  auto scaleBOpt = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  auto biasOpt = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  bool accumulate = stackGetter.getNextInput<bool>();

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

// Left for now, used by torch.compile meta
sym_sizes_vec fp8_gemm_out_shape(
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

REGISTER_CUSTOM_OP_OUTSHAPE_FUN(fp8_gemm, fp8_gemm_out_shape);

OutputMetaDataVector Fp8GemmV2Meta(const at::Stack& stack) {
  auto A = stack_tensor(stack, 0);
  bool trans_A = stack[1].toBool();
  auto B = stack_tensor(stack, 2);
  bool trans_B = stack[3].toBool();
  OutputMetaData meta;
  try {
    meta.shape = getBatchMatmulOutShape(A.sizes(), B.sizes(), trans_A, trans_B);
  } catch (const std::invalid_argument& e) {
    TORCH_CHECK(false, e.what());
    return {};
  }
  meta.dtype = stack[5].toScalarType();
  return {meta};
}

void Fp8GemmV2::AddNode(sh::graph& graph, const at::Stack& stack) {
  TORCH_CHECK(stack.size() == 11, "Fp8GemmV2 must have 11 input arguments");

  StackGetter stackGetter(this, stack, "Fp8Gemm::AddNode");
  auto A = stackGetter.getNextInput<TensorsPair>();
  bool transA = stackGetter.getNextInput<bool>();
  auto B = stackGetter.getNextInput<TensorsPair>();
  bool transB = stackGetter.getNextInput<bool>();
  auto DOpt = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  auto out_type = stackGetter.getNextInput<c10::ScalarType>();
  auto scaleAOpt =
      stackGetter.getNextInput<std::variant<TensorsPair, c10::IValue>>();
  auto scaleBOpt =
      stackGetter.getNextInput<std::variant<TensorsPair, c10::IValue>>();
  auto biasOpt = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  bool accumulate = stackGetter.getNextInput<bool>();
  auto scale_shape = stackGetter.getNextInput<c10::IValue>();

  std::string guid = get_guid_with_precision("fp8_gemm", out_type);

  std::vector<synTensor> syn_inputs = {A.syn_t, B.syn_t};
  std::vector<sh::tensor> adjusted_scale;

  HandleScale(
      this,
      graph,
      scaleAOpt,
      A.pt_t,
      transA,
      adjusted_scale,
      syn_inputs,
      p_context_->device_id_);
  HandleScale(
      this,
      graph,
      scaleBOpt,
      B.pt_t,
      transB,
      adjusted_scale,
      syn_inputs,
      p_context_->device_id_,
      scale_shape);

  if (scaleAOpt.isTensorsPair() && scaleBOpt.isTensorsPair()) {
    TORCH_CHECK(
        at::are_expandable(
            scaleAOpt.toTensorsPair().pt_t.sizes(),
            scaleBOpt.toTensorsPair().pt_t.sizes()),
        "Scale tensors must be broadcastable.");
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

  synGEMMParams params{transA, transB};

  auto meta = Fp8GemmV2Meta(stack)[0];

  auto gemm = OpBackend::BuildNode(
      this,
      graph,
      {guid,
       syn_inputs,
       {{meta.shape, meta.dtype, 0}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(gemm[0]);
}

/********** InPlaceInterleave **********/

OutputMetaDataVector InPlaceInterleaveMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto shape = self.sizes().vec();

  TORCH_CHECK(
      stack.size() == 1, "InPlaceInterleave must have 1 input argument");
  TORCH_CHECK(shape.size() == 4, "Input has to be a 4D tensor.");
  TORCH_CHECK(shape[0] % 4 == 0, "Batch size has to be a multiple of 4.");

  OutputMetaData meta;
  meta.shape = shape;
  meta.dtype = self.scalar_type();
  return {meta};
}

struct InPlaceInterleave : OpBackend {
  InPlaceInterleave(int device_id, c10::ScalarType scalar_type)
      : OpBackend(
            device_id,
            "in_place_interleave_fwd",
            scalar_type,
            {0},
            {},
            {},
            false) {
    SetOutputMetaFn(InPlaceInterleaveMeta);
  }
};

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

OutputMetaDataVector Conv2dFp8Meta(const at::Stack& stack) {
  const auto input_shape = stack_tensor(stack, 0).sizes();
  const auto weight_shape = stack_tensor(stack, 1).sizes();
  const auto stride =
      expand_param_if_needed(stack[3].toIntList().vec(), "stride", 2);
  const auto padding =
      expand_param_if_needed(stack[4].toIntList().vec(), "padding", 2);
  const auto dilation =
      expand_param_if_needed(stack[5].toIntList().vec(), "dilation", 2);
  const auto out_dtype =
      stack[7].toOptional<c10::ScalarType>().value_or(at::ScalarType::BFloat16);

  OutputMetaData meta;
  meta.dtype = out_dtype;
  meta.shape = ComputeConv2dOutputSize(
      input_shape, weight_shape, stride, padding, dilation);
  return {meta};
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

void Conv2dFp8::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "Conv2dFp8::AddNode");
  auto input = stackGetter.getNextInput<TensorsPair>();
  auto weight = stackGetter.getNextInput<TensorsPair>();
  auto bias_opt = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  auto stride = expand_param_if_needed(
      stackGetter.getNextInput<std::vector<int64_t>>(), "stride", 2);
  auto padding = expand_param_if_needed(
      stackGetter.getNextInput<std::vector<int64_t>>(), "padding", 2);
  auto dilation = expand_param_if_needed(
      stackGetter.getNextInput<std::vector<int64_t>>(), "dilation", 2);
  auto groups = stackGetter.getNextInput<int>();
  auto out_dtype =
      stackGetter.getNextInput<c10::optional<c10::ScalarType>>().value_or(
          at::ScalarType::BFloat16);
  auto scale_input_opt =
      stackGetter.getNextInput<std::variant<TensorsPair, c10::IValue>>();
  auto scale_weight_opt =
      stackGetter.getNextInput<std::variant<TensorsPair, c10::IValue>>();

  TORCH_CHECK(
      input.pt_t.dim() == 4 and weight.pt_t.dim() == 4,
      "Input and weight must be 4D tensors");
  if (bias_opt) {
    TORCH_CHECK(
        bias_opt->pt_t.dim() == 1 and
            bias_opt->pt_t.sizes()[0] == weight.pt_t.sizes()[0],
        "Bias must be 1D tensor with size equal to weight dim0");
  }

  update_guid_dtype(guid_, out_dtype);

  auto out_shape = ComputeConv2dOutputSize(
      input.pt_t.sizes(), weight.pt_t.sizes(), stride, padding, dilation);
  auto params = FillConv2dFp8Params(
      weight.pt_t.sizes(), stride, padding, dilation, groups);

  std::vector<synTensor> syn_inputs{input.syn_t, weight.syn_t};
  syn_inputs.push_back(bias_opt ? bias_opt->syn_t : nullptr);
  std::vector<sh::tensor> adjusted_scale;

  for (int i = 0; i < 2; ++i) {
    const auto& scale = (i == 0) ? scale_input_opt : scale_weight_opt;
    if (scale.isTensorsPair()) {
      auto scale_tp = scale.toTensorsPair();
      TORCH_CHECK(
          scale_tp.pt_t.numel() == 1,
          "Multi-element scale tensors are not supported yet.");
      syn_inputs.push_back(scale_tp.syn_t);
    } else {
      HandleScaleScalar(
          this,
          graph,
          scale.toIValue(),
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
      {guid_,
       std::move(syn_inputs),
       {{out_shape, out_dtype, 0}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(conv[0]);
}

} // namespace habana

static const auto& CastKernelRegistry =
    habana::KernelRegistry()
        .add(
            "hpu::in_place_interleave",
            KERNEL_FN_GLOBAL(habana::InPlaceInterleave));
