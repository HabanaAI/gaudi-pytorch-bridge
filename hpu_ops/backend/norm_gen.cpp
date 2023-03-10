/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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
#include "generated/backend/_weight_norm_interface.h"
#include "generated/backend/_weight_norm_interface_backward.h"
#include "generated/backend/linalg_vector_norm.h"
#include "generated/backend/native_layer_norm.h"
#include "generated/backend/native_layer_norm_backward.h"
#include "generated/backend/norm.h"
#include "habana_kernels/norm_kernels.h"
#include "hpu_ops/backend/reduction_template.h"
#include "hpu_ops/hpu_op_helper.h"

#define INF std::numeric_limits<float>::infinity()
namespace habana {

namespace sh = synapse_helpers;

sizes_vec NormOutputShape(const at::Stack&) {
  return {{}};
}

// Second param is unused so neglecting it
sizes_vec NormOpOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto dim =
      stack.at(2).isNone() ? std::vector<int64_t>() : stack.at(2).toIntVector();

  const bool keepdim = stack.at(3).toBool();

  return ReductionOutputShape(self, dim, keepdim);
}

std::shared_ptr<void> FillPFormNormOpParams(
    const int64_t ndim,
    size_t& size,
    int64_t index,
    c10::optional<at::Scalar> opt_ord) {
  const at::Scalar ord = opt_ord.value_or(2);
  PARAMS_STUB(ns_ReduceLpV2::Params);
  auto reduction_dim = ndim - 1 - index;
  params->reductionDimension = reduction_dim;
  if (ord.isFloatingPoint()) {
    get<float>(params->p) = ord.to<float>();
    params->typeOfP = TYPE_P_IS_FLOAT;
  } else {
    get<int>(params->p) = ord.to<int>();
    params->typeOfP = TYPE_P_IS_INT;
  }
  return params;
}

void NormHabanaOperator::AddNode(sh::graph& graph, const at::Stack& stack) {
  const auto self = stack.at(0).toTensor();
  const auto p = stack.at(1).toScalar();
  const auto dtype = stack.at(2).toScalarType();
  auto outshape = self.sizes();
  auto n_dims = self.dim();

  auto input = syn_in(0);
  auto input_in_dtype = HandleReductionDtype(this, graph, self, input, dtype);
  if (input_in_dtype.has_value()) {
    input = input_in_dtype.value().get();
  }

  if (p.toFloat() == 2.0) {
    if (n_dims <= 1 || self.sizes()[0] == 1) {
      auto mul = BuildOp(
          graph,
          MULT_GUID + habana_helpers::name_suffix_from_type(dtype),
          {input, input},
          {{outshape, dtype}});

      std::vector<synTensor> reduction_inputs = {mul[0].get()};
      std::vector<sh::tensor> reshape;

      if (n_dims > 1) {
        auto reshape_outshape = self.numel();
        reshape.emplace_back(
            ReshapeHelper(graph, reduction_inputs[0], reshape_outshape, dtype));
        reduction_inputs = {reshape[0].get()};
      }

      ns_Reduction::Params reduce_params{};
      reduce_params.reductionDimension = 0;
      auto sum = BuildOp(
          graph,
          "reduce_sum_fwd_" + habana_helpers::name_suffix_from_type(dtype),
          reduction_inputs,
          {{1, dtype}},
          &reduce_params,
          sizeof(reduce_params));

      auto sqrt = BuildOp(
          graph,
          "sqrt_fwd_" + habana_helpers::name_suffix_from_type(dtype),
          {sum[0].get()},
          {{1, dtype, 0}});

      syn_out(0) = std::move(sqrt[0]);

    } else {
      auto norm = BuildOp(
          graph,
          "frobenius_norm_fwd_" + habana_helpers::name_suffix_from_type(dtype),
          {input},
          {{1, dtype, 0}});

      syn_out(0) = std::move(norm[0]);
    }

  } else {
    auto reshape_outshape = self.numel();
    std::vector<sh::tensor> reshape;
    if (n_dims > 1) {
      reshape.emplace_back(
          ReshapeHelper(graph, input, reshape_outshape, dtype));
      input = reshape[0].get();
    }

    ns_LpNormKernel::Params lpnorm_params{};
    lpnorm_params.p = p.toFloat();
    lpnorm_params.dim = 0;
    lpnorm_params.eps = 0;
    auto norm = BuildOp(
        graph,
        "lpnorm_fwd_" + habana_helpers::name_suffix_from_type(dtype),
        {input},
        {{reshape_outshape, dtype}, {reshape_outshape, dtype}},
        &lpnorm_params,
        sizeof(lpnorm_params));

    auto reciprocal = BuildOp(
        graph,
        "reciprocal_fwd_" + habana_helpers::name_suffix_from_type(dtype),
        {norm[1].get()},
        {{reshape_outshape, dtype}});

    synSliceParamsNDims slice_params{};
    slice_params.axes[0] = 0;
    slice_params.starts[0] = 0;
    slice_params.ends[0] = 1;
    slice_params.steps[0] = 1;
    auto slice = BuildOp(
        graph,
        "slice_" + habana_helpers::name_suffix_from_type(dtype),
        {reciprocal[0].get()},
        {{1, dtype, 0}},
        &slice_params,
        sizeof(slice_params));

    syn_out(0) = std::move(slice[0]);
  }
}

static sh::tensor L0NormPreprocess(
    OpBackend* op,
    sh::graph& graph,
    std::vector<synTensor> input,
    const at::IntArrayRef inputshape,
    const at::ScalarType& dtype) {
  auto zero = OpBackend::BuildConstant(op, graph, 0, dtype, inputshape);
  auto compare = OpBackend::BuildNode(
      op,
      graph,
      {"equal_fwd_" + habana_helpers::name_suffix_from_type(dtype),
       {input[0], zero.get()},
       {{inputshape, torch::kBool}}});

  auto not_equal = OpBackend::BuildNode(
      op,
      graph,
      {"not_fwd_i8", {compare[0].get()}, {{inputshape, torch::kBool}}});

  return OpBackend::BuildCast(
      op, graph, not_equal[0].get(), inputshape, torch::kBool, dtype);
}

static sh::tensor NegPosInfNormPreprocess(
    OpBackend* op,
    sh::graph& graph,
    std::vector<synTensor> input,
    const at::IntArrayRef inputshape,
    const at::ScalarType& dtype) {
  auto abs = OpBackend::BuildNode(
      op,
      graph,
      {"abs_fwd_" + habana_helpers::name_suffix_from_type(dtype),
       std::move(input),
       {{inputshape, dtype}}});

  return std::move(abs.at(0));
}

void VecNormCheck(
    const torch::Tensor& self,
    const at::ScalarType& dtype,
    at::Scalar ord,
    const std::vector<int64_t>& dim) {
  auto p = (ord.isFloatingPoint()) ? ord.toFloat() : ord.toInt();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      dtype == torch::kBFloat16 || dtype == torch::kFloat,
      "linalg.vector_norm: Expected input dtype to be Float or kBFloat16, but got ",
      dtype);

  if (self.numel() == 0) {
    TORCH_CHECK(
        p >= 0,
        "linalg.vector_norm of negative order cannot be performed on an empty tensor");
    if (p == INF) {
      bool has_identity = true;
      if (dim.size() == 0) {
        has_identity = false;
      } else {
        for (unsigned i = 0; i < dim.size(); ++i) {
          if (self.size(dim[i]) == 0) {
            has_identity = false;
            break;
          }
        }
      }
      TORCH_CHECK(
          has_identity,
          "linalg.vector_norm cannot compute the infinity norm on an empty ",
          "dimension because the operation does not have an identity");
    }
  }
}

void NormCheck(const at::ScalarType& dtype) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      dtype == torch::kBFloat16 || dtype == torch::kFloat,
      "norm: Expected input dtype to be Float or kBFloat16, but got ",
      dtype);
}

static sh::tensor NormCommon(
    OpBackend* op,
    sh::graph& graph,
    synTensor input_tensor,
    at::ScalarType dtype,
    const torch::Tensor& self,
    const std::vector<int64_t>& dim,
    const bool keepdim,
    const at::Scalar& ord,
    std::vector<NodeAttr::NodeOutputAttr> output_attr,
    const bool is_vec_norm) {
  auto p = (ord.isFloatingPoint()) ? ord.toFloat() : ord.toInt();
  auto norm_ord = ord.toFloat();
  auto self_shape = self.sizes().vec();
  struct vec_norm_inputs {
    std::string guid;
    std::function<sh::tensor(
        OpBackend*,
        sh::graph&,
        std::vector<synTensor>,
        const at::IntArrayRef,
        const at::ScalarType&)>
        pre_fn{};
  };
  if (is_vec_norm)
    VecNormCheck(self, dtype, ord, dim);
  else
    NormCheck(dtype);

  if (self.numel() == 0) {
    at::Scalar s = (p < 0) && is_vec_norm ? INF : 0;
    return OpBackend::BuildConstant(op, graph, s, dtype, 1, 0);
  }

  std::map<float, vec_norm_inputs> mod_inputs = {
      {0.0, {"reduce_sum_fwd_", L0NormPreprocess}},
      {1.0, {"reduce_L1_fwd_"}},
      {2.0, {"reduce_L2_fwd_"}},
      {INF, {"reduce_max_fwd_", NegPosInfNormPreprocess}},
      {-INF, {"reduce_min_fwd_", NegPosInfNormPreprocess}}};

  // Using Lp_fwd if the ord value is not present in the map
  if (mod_inputs.find(norm_ord) == mod_inputs.end()) {
    auto norm_itr = HandleReductionDimAndKeepdim(
        op,
        graph,
        self,
        {input_tensor},
        dim,
        keepdim,
        "reduce_Lp_fwd_" + habana_helpers::name_suffix_from_type(dtype),
        output_attr,
        FillPFormNormOpParams,
        ord);
    return std::move(norm_itr.at(0));
  } else {
    auto inputs = mod_inputs[norm_ord];
    if (norm_ord != INF && norm_ord != -INF) {
      auto norm_itr = HandleReductionDimAndKeepdim(
          op,
          graph,
          self,
          {inputs.pre_fn
               ? inputs.pre_fn(op, graph, {input_tensor}, self_shape, dtype)
                     .get()
               : input_tensor},
          dim,
          keepdim,
          inputs.guid + habana_helpers::name_suffix_from_type(dtype),
          output_attr);
      return std::move(norm_itr.at(0));
    }
    auto norm_itr = HandleReductionDimAndKeepdim(
        op,
        graph,
        self,
        {inputs.pre_fn
             ? inputs.pre_fn(op, graph, {input_tensor}, self_shape, dtype).get()
             : input_tensor},
        dim,
        keepdim,
        inputs.guid + habana_helpers::name_suffix_from_type(dtype),
        {{output_attr[0].sizes, output_attr[0].dtype}});

    // Handle Inf values
    ns_IsInfKernel::Params params{1, 1};
    auto isinf_output = OpBackend::BuildNode(
        op,
        graph,
        {"isinf_fwd_" + habana_helpers::name_suffix_from_type(dtype),
         {input_tensor},
         {{self_shape, torch::kInt8}},
         &params,
         sizeof(params)});

    auto isinf_casted = OpBackend::BuildCast(
        op,
        graph,
        isinf_output[0].get(),
        self_shape,
        torch::kInt8,
        torch::kInt32);

    op->SetScalarType(at::kInt);
    auto isinf_reduced = HandleReductionDimAndKeepdim(
        op,
        graph,
        self,
        {isinf_casted.get()},
        dim,
        keepdim,
        inputs.guid + "i32",
        {{output_attr[0].sizes, torch::kInt32}});

    auto isinf_condition = OpBackend::BuildCast(
        op,
        graph,
        isinf_reduced.at(0).get(),
        output_attr[0].sizes,
        torch::kInt32,
        torch::kInt8);

    auto const_inf =
        OpBackend::BuildConstant(op, graph, INF, dtype, output_attr[0].sizes);

    auto intermediate = OpBackend::BuildNode(
        op,
        graph,
        {"where_fwd_" + habana_helpers::name_suffix_from_type(dtype),
         {isinf_condition.get(), const_inf.get(), norm_itr.at(0).get()},
         {{output_attr[0].sizes, dtype}}});

    // Handle NaN values
    auto isnan_output = OpBackend::BuildNode(
        op,
        graph,
        {"isnan_fwd_" + habana_helpers::name_suffix_from_type(dtype),
         {input_tensor},
         {{self_shape, torch::kInt8}}});

    auto isnan_casted = OpBackend::BuildCast(
        op,
        graph,
        isnan_output[0].get(),
        self_shape,
        torch::kInt8,
        torch::kInt32);

    auto isnan_reduced = HandleReductionDimAndKeepdim(
        op,
        graph,
        self,
        {isnan_casted.get()},
        dim,
        keepdim,
        "reduce_max_fwd_i32",
        {{output_attr[0].sizes, torch::kInt32}});

    auto isnan_condition = OpBackend::BuildCast(
        op,
        graph,
        isnan_reduced.at(0).get(),
        output_attr[0].sizes,
        torch::kInt32,
        torch::kInt8);

    auto const_nan =
        OpBackend::BuildConstant(op, graph, NAN, dtype, output_attr[0].sizes);

    auto out = OpBackend::BuildNode(
        op,
        graph,
        {"where_fwd_" + habana_helpers::name_suffix_from_type(dtype),
         {isnan_condition.get(), const_nan.get(), intermediate.at(0).get()},
         {{output_attr[0].sizes, dtype, 0}}});

    return std::move(out.at(0));
  }
}

void VecNormOp::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto optional_dtype = stack.at(4).toOptional<at::ScalarType>();
  auto optional_ord = stack.at(1).toOptional<at::Scalar>();
  std::vector<int64_t> dim =
      stack.at(2).isNone() ? std::vector<int64_t>() : stack.at(2).toIntVector();
  const at::Scalar ord = optional_ord.value_or(2);
  const at::ScalarType& dtype = optional_dtype.value_or(ScalarType());
  const bool keepdim = stack.at(3).toBool();

  auto output_shape = NormOpOutputShape(stack)[0];
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      dtype == torch::kBFloat16 || dtype == torch::kFloat,
      "linalg.vector_norm: Expected input dtype to be Float or kBFloat16, but got ",
      dtype);

  auto result = NormCommon(
      this,
      graph,
      syn_in(0),
      dtype,
      self,
      dim,
      keepdim,
      ord,
      {{output_shape, dtype, 0}},
      true /* vec_norm */);
  syn_out(0) = std::move(result);
}

void NormOpWithDtype::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto optional_ord = stack.at(1).toOptional<at::Scalar>();
  std::vector<int64_t> dim = stack.at(2).toIntVector();
  const at::Scalar ord = optional_ord.value_or(2);
  const at::ScalarType& dtype = stack.at(4).toScalarType();
  const bool keepdim = stack.at(3).toBool();
  auto output_shape = NormOpOutputShape(stack)[0];

  auto result = NormCommon(
      this,
      graph,
      syn_in(0),
      dtype,
      self,
      dim,
      keepdim,
      ord,
      {{output_shape, dtype, 0}},
      false /* norm */);
  syn_out(0) = std::move(result);
}

void NormOpWithOutDtype::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto optional_ord = stack.at(1).toOptional<at::Scalar>();
  std::vector<int64_t> dim = stack.at(2).toIntList().vec();
  const at::Scalar ord = optional_ord.value_or(2);
  const bool keepdim = stack.at(3).toBool();
  auto output_shape = NormOpOutputShape(stack)[0];

  auto result = NormCommon(
      this,
      graph,
      syn_in(0),
      ScalarType(),
      self,
      dim,
      keepdim,
      ord,
      {{output_shape, ScalarType(), 0}},
      false /* norm */);
  syn_out(0) = std::move(result);
}

void NormOpScalar::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto ord = stack.at(1).toScalar();

  auto result = NormCommon(
      this,
      graph,
      syn_in(0),
      ScalarType(),
      self,
      {},
      false,
      ord,
      {{1, ScalarType(), 0}},
      false /* norm */);
  syn_out(0) = std::move(result);
}

sizes_vec LayerNormOutputShape(const at::Stack& stack) {
  auto input = stack[0].toTensor();
  auto normalized_shape = stack[1].toIntList();

  const auto input_shape = input.sizes();
  auto output_sizes = input_shape.vec();
  const int axis = input.dim() - normalized_shape.size();
  std::vector<int64_t> shape_mean_rstd = output_sizes;
  for (size_t i = axis; i < shape_mean_rstd.size(); ++i) {
    shape_mean_rstd[i] = 1;
  }
  return {output_sizes, shape_mean_rstd, shape_mean_rstd};
}

static sh::tensor CreateLayerNormBiasWeightTensor(
    OpBackend* op,
    sh::graph& graph,
    std::vector<sh::tensor>& storage,
    const OpBackend::TensorsPair& input,
    const c10::optional<OpBackend::TensorsPair>& weightOrBiasOpt,
    int64_t constant_numel,
    float constant_value,
    std::array<int64_t, 1>& weightOrBias_shape) {
  if (weightOrBiasOpt) {
    sh::tensor* pWeightOrBias = &weightOrBiasOpt->sh_t;
    if (weightOrBiasOpt->pt_t.scalar_type() != c10::kFloat) {
      storage.push_back(OpBackend::BuildCast(
          op,
          graph,
          pWeightOrBias->get(),
          weightOrBiasOpt->pt_t.sizes(),
          weightOrBiasOpt->pt_t.scalar_type(),
          c10::kFloat));
      pWeightOrBias = &storage.back();
    }

    weightOrBias_shape = {weightOrBiasOpt->pt_t.numel()};
    auto weightOrBias = OpBackend::BuildReshape(
        op, graph, pWeightOrBias->get(), weightOrBias_shape, c10::kFloat);
    return weightOrBias;
  } else {
    weightOrBias_shape = {constant_numel};
    auto weightOrBias = OpBackend::BuildConstant(
        op, graph, constant_value, c10::kFloat, weightOrBias_shape);
    return weightOrBias;
  }
}

void LayerNormHabanaOperator::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "LayerNormHabanaOperator::AddNode");
  auto input = getNextInput<TensorsPair>(stackGetter);
  auto normalized_shape = getNextInput<std::vector<int64_t>>(stackGetter);
  auto weightOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto biasOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto eps = getNextInput<double>(stackGetter);

  const auto input_shape = input.pt_t.sizes();
  const auto input_ndim = input.pt_t.dim();
  const int normalized_ndim = normalized_shape.size();

  auto use_tpc_affine_path =
      !weightOpt && !biasOpt && (input_ndim == 4) && (normalized_ndim == 3);

  int64_t normalized_shape_numel = c10::multiply_integers(
      normalized_shape.cbegin(), normalized_shape.cend());

  int64_t weightOrBias_constant_numel = use_tpc_affine_path
      ? input_shape[input_ndim - 1]
      : normalized_shape_numel;

  std::vector<sh::tensor> storage;
  // Manual handling of reserved size - maximum number of calls to
  // storage.push_back
  storage.reserve(3);

  std::array<int64_t, 1> weightOrBias_shape = {};
  sh::tensor weight = CreateLayerNormBiasWeightTensor(
      this,
      graph,
      storage,
      input,
      weightOpt,
      weightOrBias_constant_numel,
      1.0f,
      weightOrBias_shape);

  sh::tensor bias = CreateLayerNormBiasWeightTensor(
      this,
      graph,
      storage,
      input,
      biasOpt,
      weightOrBias_constant_numel,
      0.0f,
      weightOrBias_shape);

  sh::tensor* localInput = &input.sh_t;

  if (input_ndim < normalized_ndim ||
      !input_shape.slice(input_ndim - normalized_ndim)
           .equals(normalized_shape)) {
    std::stringstream ss;
    ss << "Given normalized_shape=" << normalized_shape
       << ", expected input with shape [*";
    for (auto size : normalized_shape) {
      ss << ", " << size;
    }
    ss << "], but got input of size" << input_shape;
    AT_ERROR(ss.str());
  }

  const int64_t axis = input_ndim - normalized_ndim;
  int64_t m =
      c10::multiply_integers(input_shape.cbegin(), input_shape.cbegin() + axis);
  int64_t n =
      c10::multiply_integers(input_shape.cbegin() + axis, input_shape.cend());

  int64_t input_reshaped_shape[] = {1, 1, m, n};
  if (!use_tpc_affine_path) {
    storage.push_back(ReshapeHelper(
        graph,
        localInput->get(),
        input_reshaped_shape,
        input.pt_t.scalar_type()));
    localInput = &storage.back();
  }

  ns_LayerNormKernel::ParamsNorm params_norm{};
  ns_LayerNormKernel::Params params{};
  void* paramsPtr = nullptr;
  size_t paramsSize = 0;

  if (use_tpc_affine_path) {
    params_norm.eps = static_cast<float>(eps);
    params_norm.epsValid = true;
    params_norm.NormAxisBmp =
        (1 << normalized_ndim) - 1; // normalize across CWH
    params_norm.ParamAxisBmp = 1;
    paramsPtr = &params_norm;
    paramsSize = sizeof(params_norm);
  } else {
    params.eps = static_cast<float>(eps);
    params.epsValid = true;
    paramsPtr = &params;
    paramsSize = sizeof(params);
  }

  auto outputShapes = LayerNormOutputShape(stack);
  int64_t mean_rstd_shape[] = {1, 1, m, 1};

  auto getOutputType = [this](int i) {
    return (i == 0) ? this->ScalarType() : c10::kFloat;
  };

  std::vector<NodeAttr::NodeOutputAttr> node_output_attr;
  for (int i = 0; i < outputShapes.size(); ++i) {
    c10::ScalarType outputType = getOutputType(i);
    if (use_tpc_affine_path) {
      node_output_attr.push_back({outputShapes[i], outputType, i});
    } else {
      node_output_attr.push_back(
          {i == 0 ? input_reshaped_shape : mean_rstd_shape, outputType});
    }
  }

  auto ln = BuildOp(
      graph,
      guid_,
      {localInput->get(), bias.get(), weight.get()},
      node_output_attr,
      paramsPtr,
      paramsSize);

  for (size_t i = 0; i < ln.size(); ++i) {
    if (use_tpc_affine_path) {
      syn_out(i) = std::move(ln[i]);
    } else {
      auto reshaped = ReshapeHelper(
          graph, ln[i].get(), outputShapes[i], getOutputType(i), i);
      syn_out(i) = std::move(reshaped);
    }
  }
}

sizes_vec LayerNormBwdOutputShape(const at::Stack& stack) {
  auto input = stack[1].toTensor();
  auto input_size = input.sizes().vec();

  std::vector<int64_t> weight_size;
  if (stack[5].isTensor()) {
    auto weight = stack[5].toTensor();
    weight_size = weight.sizes().vec();
  } else {
    weight_size = stack[2].toIntList().vec();
  }

  return {input_size, weight_size, weight_size};
}

static void CheckMeanRstdSizes(
    const char* label,
    const OpBackend::TensorsPair& meanOrRstd) {
  TORCH_CHECK(
      meanOrRstd.pt_t.sizes().size() <= 4,
      "Input ",
      label,
      " for LayerNormBackward is over 4 dims - unsupported!");
}

void LayerNormBwdHabanaOperator::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "LayerNormBwdHabanaOperator::AddNode");
  auto grad_out = getNextInput<TensorsPair>(stackGetter);
  auto input = getNextInput<TensorsPair>(stackGetter);
  auto normalized_shape = getNextInput<std::vector<int64_t>>(stackGetter);
  auto mean = getNextInput<TensorsPair>(stackGetter);
  auto rstd = getNextInput<TensorsPair>(stackGetter);
  auto weightOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto biasOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto output_mask = getNextInput<c10::List<bool>>(stackGetter);

  CheckMeanRstdSizes("mean", mean);
  CheckMeanRstdSizes("rstd", rstd);

  const auto input_shape = input.pt_t.sizes();
  const auto input_ndim = input.pt_t.dim();
  const int normalized_ndim = normalized_shape.size();
  const int axis = input_ndim - normalized_ndim;
  int64_t m =
      c10::multiply_integers(input_shape.cbegin(), input_shape.cbegin() + axis);
  int64_t n =
      c10::multiply_integers(input_shape.cbegin() + axis, input_shape.cend());
  std::array<int64_t, 4> sizes_as_4D = {1, 1, m, n};

  std::vector<sh::tensor> storage;
  // Manual handling of reserved size - maximum number of calls to
  // storage.push_back
  storage.reserve(7);

  for (int i = 0; i < 2; ++i) {
    const auto& src = (i == 0) ? grad_out : input;
    storage.push_back(ReshapeHelper(
        graph, src.sh_t.get(), sizes_as_4D, src.pt_t.scalar_type()));
  }

  const auto& grad_out_as_4D = storage[0];
  const auto& input_as_4D = storage[1];

  std::array<int64_t, 1> weightShape = {};
  sh::tensor weight = CreateLayerNormBiasWeightTensor(
      this,
      graph,
      storage,
      input,
      weightOpt,
      c10::multiply_integers(
          normalized_shape.cbegin(), normalized_shape.cend()),
      1.0f,
      weightShape);

  std::array<int64_t, 4> mean_rstd_as_4D = {1, 1, m, 1};
  std::array<unsigned, 2> storage_indices = {};
  for (int i = 0; i < storage_indices.size(); ++i) {
    const auto& src = (i == 0) ? mean : rstd;
    storage.push_back(ReshapeHelper(
        graph, src.sh_t.get(), mean_rstd_as_4D, src.pt_t.scalar_type()));

    if (src.pt_t.scalar_type() != c10::kFloat) {
      storage.push_back(CastHelper(
          graph,
          storage.back().get(),
          mean_rstd_as_4D,
          src.pt_t.scalar_type(),
          c10::kFloat));
    }
    storage_indices[i] = storage.size() - 1;
  }

  const auto& mean_as_4D = storage[storage_indices[0]];
  const auto& rstd_as_4D = storage[storage_indices[1]];

  auto outputShapes = LayerNormBwdOutputShape(stack);

  ns_LayerNormKernel::Params params;
  params.epsValid = false;

  auto lnbwd = BuildOp(
      graph,
      guid_,
      {input_as_4D.get(),
       grad_out_as_4D.get(),
       mean_as_4D.get(),
       rstd_as_4D.get(),
       weight.get()},
      {{sizes_as_4D, ScalarType()},
       {weightShape, c10::kFloat},
       {weightShape, c10::kFloat}},
      &params,
      sizeof(params));

  if (ScalarType() != c10::kFloat) {
    for (int i = 1; i < lnbwd.size(); ++i) {
      lnbwd[i] = CastHelper(
          graph, lnbwd[i].get(), weightShape, c10::kFloat, ScalarType());
    }
  }

  static std::array<int, 3> outIds = {0, 2, 1};
  for (size_t i = 0; i < outIds.size(); ++i) {
    auto reshaped = ReshapeHelper(
        graph, lnbwd[i].get(), outputShapes[i], ScalarType(), outIds[i]);
    syn_out(outIds[i]) = std::move(reshaped);
  }
}

sizes_vec WeightNormOutputShape(const at::Stack& stack) {
  const torch::Tensor& v_in = stack_tensor(stack, 0);
  const torch::Tensor& g_in = stack_tensor(stack, 1);
  auto dim = stack.at(2).toInt();
  auto shapes = at::infer_size(v_in.sizes(), g_in.sizes());
  auto norm_shapes = v_in.sizes()[dim];
  return {shapes, {norm_shapes}};
}

void WeightNormOp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto v_in = stack_tensor(stack, 0);
  auto g_in = stack_tensor(stack, 1);
  auto dim = stack.at(2).toInt();

  /*
  NOTE:
  We use the CPU implementation that follows the "non-fused" (ie., assumes
  can_use_fused=0) path.
  */
  TORCH_CHECK(
      v_in.device() == g_in.device(),
      "weight_norm: expected v_in and g_in to be on the same device, but v_in is "
      "on ",
      v_in.device(),
      " and g_in is on ",
      g_in.device());

  auto outsize_norm_op = v_in.sizes()[dim];

  std::vector<int64_t> dim_to_norm;
  for (int64_t i = 0; i < v_in.ndimension(); ++i) {
    if (i != dim) // skip given dimension
      dim_to_norm.push_back(i);
  }

  at::Scalar ord = 2.0;

  // align with cuda behavior, keep norm in 'Float' when g is 'BFloat16'
  const auto dtype = (g_in.scalar_type() == at::ScalarType::BFloat16)
      ? at::ScalarType::Float
      : g_in.scalar_type();

  std::vector<synapse_helpers::tensor> normOp;
  if (dtype != v_in.scalar_type()) {
    auto cast_bf16_to_float =
        CastHelper(graph, syn_in(0), v_in.sizes(), v_in.scalar_type(), dtype);

    normOp.emplace_back(NormCommon(
        this,
        graph,
        cast_bf16_to_float.get(),
        dtype,
        v_in,
        dim_to_norm,
        false,
        ord,
        {{outsize_norm_op, dtype, 1}},
        false));
  } else {
    normOp.emplace_back(NormCommon(
        this,
        graph,
        syn_in(0),
        ScalarType(),
        v_in,
        dim_to_norm,
        false,
        ord,
        {{outsize_norm_op, ScalarType(), 1}},
        false));
  }
  auto outsize_div_op = at::infer_size(g_in.sizes(), outsize_norm_op);
  const std::string opStringSuffix =
      "_fwd_" + habana_helpers::name_suffix_from_type(ScalarType());
  auto divOp = BuildOp(
      graph,
      "div" + opStringSuffix,
      {syn_in(1), normOp[0].get()},
      {{outsize_div_op, ScalarType()}});

  auto outsize_mul_op = at::infer_size(v_in.sizes(), outsize_div_op);
  auto mulOp = BuildOp(
      graph,
      "mult" + opStringSuffix,
      {syn_in(0), divOp.at(0).get()},
      {{outsize_mul_op, ScalarType(), 0}});

  syn_out(0) = std::move(mulOp[0]);
  syn_out(1) = std::move(normOp[0]);
}

sizes_vec WeightNormBwdOutputShape(const at::Stack& stack) {
  const torch::Tensor& grad_w = stack_tensor(stack, 0);
  const torch::Tensor& saved_v = stack_tensor(stack, 1);
  const torch::Tensor& saved_g = stack_tensor(stack, 2);
  const torch::Tensor& saved_norms = stack_tensor(stack, 3);
  auto dim = stack.at(4).toInt();
  int64_t last_dim = saved_v.dim() - 1;
  int64_t last_size = saved_v.size(last_dim);
  std::vector<int64_t> bcast_size(saved_v.dim(), 1);
  if (dim == 0) {
    bcast_size[0] = saved_v.size(0);
  } else {
    bcast_size[last_dim] = last_size;
  }
  auto shapes1 = at::infer_size(grad_w.sizes(), saved_v.sizes());
  auto shapes2 = at::infer_size(saved_norms.sizes(), saved_g.sizes());
  auto shapes3 = at::infer_size(shapes1, shapes2);
  auto shapes = at::infer_size(shapes3, bcast_size);
  return {shapes, bcast_size};
}

void WeightNormBwdOp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const torch::Tensor& grad_w = stack_tensor(stack, 0);
  const torch::Tensor& saved_v = stack_tensor(stack, 1);
  const torch::Tensor& saved_g = stack_tensor(stack, 2);
  const torch::Tensor& saved_norms = stack_tensor(stack, 3);
  auto dim = stack.at(4).toInt();

  // In Functions.cpp, the HardshrinkBackward object supplies
  // "grad.contiguous()" as the first argument, so grad_w should be contiguous
  // here. All these checks should succeed:
  TORCH_CHECK(grad_w.is_contiguous(), "grad_w must be contiguous");
  TORCH_CHECK(saved_v.is_contiguous(), "saved_v must be contiguous");
  TORCH_CHECK(saved_g.is_contiguous(), "saved_g must be contiguous");
  TORCH_CHECK(saved_norms.is_contiguous(), "saved_norms must be contiguous");

  int64_t last_dim = saved_v.dim() - 1;
  int64_t last_size = saved_v.size(last_dim);

  // Like weight_norm_fused_backward, weight_norm_differentiable_backward should
  // only ever be called through a WeightNormFusedBackward object, so we expect
  // that dim == 0 || dim == saved_v.size(-1)
  TORCH_CHECK(
      dim == 0 || dim == last_dim,
      "Expected dim to be the first or last dimension");

  // saved_g and saved_norms are already shaped to broadcast over the correct
  // dimensions

  // ...but saved_norms might be Float when saved_g and saved_v are half.
  // To consider:  saved_norms.to(..., True );

  /////auto norms = saved_norms.to(saved_g.scalar_type());
  std::vector<synapse_helpers::tensor> norms_cast;
  if (saved_norms.scalar_type() != saved_g.scalar_type()) {
    norms_cast.emplace_back(CastHelper(
        graph,
        syn_in(3),
        saved_norms.sizes(),
        saved_norms.scalar_type(),
        saved_g.scalar_type()));
  }
  std::vector<synapse_helpers::tensor> per_dim_sums;
  std::vector<synapse_helpers::tensor> divOp21;
  std::vector<synapse_helpers::tensor> mulOp22;
  std::vector<synapse_helpers::tensor> divOp23;
  std::vector<synapse_helpers::tensor> mulOp24;
  std::vector<synapse_helpers::tensor> subOp25;
  std::vector<synapse_helpers::tensor> grad_v;
  std::vector<synapse_helpers::tensor> grad_g;
  std::vector<int64_t> bcast_size(saved_v.dim(), 1);

  // Analytic backward path using differentiable primitive ops
  if (dim == 0) {
    bcast_size[0] = saved_v.size(0);

    auto outsize_mulOp11 = at::infer_size(grad_w.sizes(), saved_v.sizes());
    auto mulOp11 = BuildOp(
        graph,
        "mult_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0), syn_in(1)},
        {{outsize_mulOp11, ScalarType()}});

    std::vector<int64_t> reshape_outshape;
    reshape_outshape.push_back(saved_v.size(0));

    if (grad_w.numel() > saved_v.numel()) {
      reshape_outshape.push_back(grad_w.numel() / saved_v.size(0));
    } else {
      reshape_outshape.push_back(saved_v.numel() / saved_v.size(0));
    }

    auto reshapeOp12 =
        ReshapeHelper(graph, mulOp11[0].get(), reshape_outshape, ScalarType());

    ns_Reduction::Params reduce_params{};
    int axis = 1;
    reduce_params.reductionDimension = reshape_outshape.size() - axis - 1;

    int reductionDimension = reshape_outshape.size() - axis - 1;
    reduce_params.reductionDimension = reductionDimension;
    auto sumOp13 = BuildOp(
        graph,
        "reduce_sum_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {reshapeOp12.get()},
        {{{reshape_outshape[reductionDimension]}, ScalarType()}},
        &reduce_params,
        sizeof(reduce_params));

    per_dim_sums.emplace_back(
        ReshapeHelper(graph, sumOp13[0].get(), bcast_size, ScalarType()));
  } else {
    bcast_size[last_dim] = last_size;
    auto outsize_mulOp11 = at::infer_size(grad_w.sizes(), saved_v.sizes());
    auto mulOp11 = BuildOp(
        graph,
        "mult_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0), syn_in(1)},
        {{outsize_mulOp11, ScalarType()}});

    std::vector<int64_t> reshape_outshape;
    if (grad_w.numel() > saved_v.numel()) {
      reshape_outshape.push_back(grad_w.numel() / last_size);
    } else {
      reshape_outshape.push_back(saved_v.numel() / last_size);
    }
    reshape_outshape.push_back(last_size);

    auto reshapeOp12 =
        ReshapeHelper(graph, mulOp11[0].get(), reshape_outshape, ScalarType());

    ns_Reduction::Params reduce_params{};
    int axis = 0;
    int reductionDimension = reshape_outshape.size() - axis - 1;
    reduce_params.reductionDimension = reductionDimension;

    auto sumOp13 = BuildOp(
        graph,
        "reduce_sum_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {reshapeOp12.get()},
        {{{reshape_outshape[reductionDimension]}, ScalarType()}},
        &reduce_params,
        sizeof(reduce_params));

    per_dim_sums.emplace_back(
        ReshapeHelper(graph, sumOp13[0].get(), bcast_size, ScalarType()));
  }

  auto outsize_divOp21 = at::infer_size(saved_g.sizes(), saved_norms.sizes());
  if (saved_norms.scalar_type() != saved_g.scalar_type()) {
    divOp21 = BuildOp(
        graph,
        "div_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(2), norms_cast[0].get()},
        {{outsize_divOp21, ScalarType()}});

    mulOp22 = BuildOp(
        graph,
        "mult_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {norms_cast[0].get(), norms_cast[0].get()},
        {{saved_norms.sizes(), ScalarType()}});
  } else {
    divOp21 = BuildOp(
        graph,
        "div_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(2), syn_in(3)},
        {{outsize_divOp21, ScalarType()}});

    mulOp22 = BuildOp(
        graph,
        "mult_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(3), syn_in(3)},
        {{saved_norms.sizes(), ScalarType()}});
  }
  divOp23 = BuildOp(
      graph,
      "div_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {per_dim_sums[0].get(), mulOp22[0].get()},
      {{bcast_size, ScalarType()}});

  auto outsize_mulOp24 = at::infer_size(saved_v.sizes(), bcast_size);
  mulOp24 = BuildOp(
      graph,
      "mult_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(1), divOp23[0].get()},
      {{outsize_mulOp24, ScalarType()}});

  auto outsize_subOp25 = at::infer_size(grad_w.sizes(), outsize_mulOp24);
  subOp25 = BuildOp(
      graph,
      "sub_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), mulOp24[0].get()},
      {{outsize_subOp25, ScalarType()}});

  auto outsize_grad_v = at::infer_size(outsize_divOp21, outsize_subOp25);
  grad_v = BuildOp(
      graph,
      "mult_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {divOp21[0].get(), subOp25[0].get()},
      {{outsize_grad_v, ScalarType(), 0}});
  if (saved_norms.scalar_type() != saved_g.scalar_type()) {
    grad_g = BuildOp(
        graph,
        "div_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {per_dim_sums[0].get(), norms_cast[0].get()},
        {{bcast_size, ScalarType(), 1}});
  } else {
    grad_g = BuildOp(
        graph,
        "div_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {per_dim_sums[0].get(), syn_in(3)},
        {{bcast_size, ScalarType(), 1}});
  }

  syn_out(0) = std::move(grad_v.at(0));
  syn_out(1) = std::move(grad_g.at(0));
}
} // namespace habana
