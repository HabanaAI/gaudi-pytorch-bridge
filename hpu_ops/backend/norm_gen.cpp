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
#include <perf_lib_layer_params.h>
#include "generated/backend/_weight_norm_interface.h"
#include "generated/backend/_weight_norm_interface_backward.h"
#include "generated/backend/linalg_vector_norm.h"
#include "generated/backend/native_group_norm_backward.h"
#include "generated/backend/native_layer_norm.h"
#include "generated/backend/native_layer_norm_backward.h"
#include "generated/backend/norm.h"
#include "hpu_ops/backend/reduction_template.h"

#define INF std::numeric_limits<float>::infinity()
namespace habana {

namespace sh = synapse_helpers;

OutputMetaDataVector NormMeta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = {};
  meta.dtype = stack.size() == 3 ? stack.at(2).toScalarType()
                                 : stack_tensor(stack, 0).scalar_type();

  return {meta};
}

// Second param is unused so neglecting it
sizes_vec NormOpOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto dim =
      stack.at(2).isNone() ? std::vector<int64_t>() : stack.at(2).toIntVector();

  const bool keepdim = stack.at(3).toBool();

  return ReductionOutputShape(self, dim, keepdim);
}

OutputMetaDataVector NormOpMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);

  OutputMetaData meta;
  meta.dtype = (stack.size() >= 5 && !stack.at(4).isTensor())
      ? stack.at(4).toScalarType()
      : self.scalar_type();
  meta.shape = NormOpOutputShape(stack)[0];

  return {meta};
}

OutputMetaDataVector VecNormMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);

  OutputMetaData meta;
  meta.dtype =
      stack.at(4).toOptional<at::ScalarType>().value_or(self.scalar_type());
  meta.shape = NormOpOutputShape(stack)[0];

  return {meta};
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
          get_guid_with_precision("mult", dtype),
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
          get_guid_with_precision("reduce_sum_fwd", dtype),
          std::move(reduction_inputs),
          {{1, dtype}},
          &reduce_params,
          sizeof(reduce_params));

      auto sqrt = BuildOp(
          graph,
          get_guid_with_precision("sqrt_fwd", dtype),
          {sum[0].get()},
          {{1, dtype, 0}});

      syn_out(0) = std::move(sqrt[0]);

    } else {
      auto norm = BuildOp(
          graph,
          get_guid_with_precision("frobenius_norm_fwd", dtype),
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
        get_guid_with_precision("lpnorm_fwd", dtype),
        {input},
        {{reshape_outshape, dtype}, {reshape_outshape, dtype}},
        &lpnorm_params,
        sizeof(lpnorm_params));

    auto reciprocal = BuildOp(
        graph,
        get_guid_with_precision("reciprocal_fwd", dtype),
        {norm[1].get()},
        {{reshape_outshape, dtype}});

    synSliceParamsV2 slice_params{};
    slice_params.axes[0] = 0;
    slice_params.starts[0] = 0;
    slice_params.ends[0] = 1;
    slice_params.steps[0] = 1;
    auto slice = BuildOp(
        graph,
        get_guid_with_precision("slice", dtype),
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
      {get_guid_with_precision("equal_fwd", dtype),
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
      {get_guid_with_precision("abs_fwd", dtype),
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
      {0.0, {"reduce_sum_fwd", L0NormPreprocess}},
      {1.0, {"reduce_L1_fwd"}},
      {2.0, {"reduce_L2_fwd"}},
      {INF, {"reduce_max_fwd", NegPosInfNormPreprocess}},
      {-INF, {"reduce_min_fwd", NegPosInfNormPreprocess}}};

  // Using Lp_fwd if the ord value is not present in the map
  if (mod_inputs.find(norm_ord) == mod_inputs.end()) {
    auto norm_itr = HandleReductionDimAndKeepdim(
        op,
        graph,
        self,
        {input_tensor},
        dim,
        keepdim,
        get_guid_with_precision("reduce_Lp_fwd", dtype),
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
          get_guid_with_precision(inputs.guid, dtype),
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
        get_guid_with_precision(inputs.guid, dtype),
        {{output_attr[0].sizes, output_attr[0].dtype}});

    // Handle Inf values
    ns_IsInfKernel::Params params{1, 1};
    auto isinf_output = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("isinf_fwd", dtype),
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
        inputs.guid + "_i32",
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
        {get_guid_with_precision("where_fwd", dtype),
         {isinf_condition.get(), const_inf.get(), norm_itr.at(0).get()},
         {{output_attr[0].sizes, dtype}}});

    // Handle NaN values
    auto isnan_output = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("isnan_fwd", dtype),
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
        {get_guid_with_precision("where_fwd", dtype),
         {isnan_condition.get(), const_nan.get(), intermediate.at(0).get()},
         {{output_attr[0].sizes, dtype, 0}}});

    return std::move(out.at(0));
  }
}

void VecNormOp::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto optional_ord = stack.at(1).toOptional<at::Scalar>();
  std::vector<int64_t> dim =
      stack.at(2).isNone() ? std::vector<int64_t>() : stack.at(2).toIntVector();
  const at::Scalar ord = optional_ord.value_or(2);
  const bool keepdim = stack.at(3).toBool();

  auto meta = VecNormMeta(stack)[0];
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      meta.dtype == torch::kBFloat16 || meta.dtype == torch::kFloat,
      "linalg.vector_norm: Expected input dtype to be Float or kBFloat16, but got ",
      meta.dtype);

  auto result = NormCommon(
      this,
      graph,
      syn_in(0),
      meta.dtype,
      self,
      dim,
      keepdim,
      ord,
      {{meta.shape, meta.dtype, 0}},
      true /* vec_norm */);
  syn_out(0) = std::move(result);
}

void NormOpWithDtype::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto optional_ord = stack.at(1).toOptional<at::Scalar>();
  std::vector<int64_t> dim = stack.at(2).toIntVector();
  const at::Scalar ord = optional_ord.value_or(2);
  const bool keepdim = stack.at(3).toBool();
  auto meta = NormOpMeta(stack)[0];

  auto result = NormCommon(
      this,
      graph,
      syn_in(0),
      meta.dtype,
      self,
      dim,
      keepdim,
      ord,
      {{meta.shape, meta.dtype, 0}},
      false /* norm */);
  syn_out(0) = std::move(result);
}

void NormOpScalar::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto ord = stack.at(1).toScalar();
  auto meta = NormMeta(stack)[0];

  auto result = NormCommon(
      this,
      graph,
      syn_in(0),
      meta.dtype,
      self,
      {},
      false,
      ord,
      {{meta.shape, meta.dtype, 0}},
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

OutputMetaDataVector LayerNormHabanaMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto shapes = LayerNormOutputShape(stack);
  OutputMetaDataVector metaVec(3);
  metaVec[0].shape = shapes[0];
  metaVec[0].dtype = self.scalar_type();
  metaVec[1].shape = shapes[1];
  metaVec[1].dtype = at::ScalarType::Float;
  metaVec[2].shape = shapes[2];
  metaVec[2].dtype = at::ScalarType::Float;
  return metaVec;
}

static synTensor CreateLayerNormBiasWeightTensor(
    OpBackend* op,
    sh::graph& graph,
    std::vector<sh::tensor>& storage,
    const OpBackend::TensorsPair& input,
    const c10::optional<OpBackend::TensorsPair>& weightOrBiasOpt,
    int64_t constant_numel,
    float constant_value,
    std::vector<int64_t>& weightOrBias_shape,
    bool isReshapeRequired) {
  if (weightOrBiasOpt) {
    synTensor synWeightOrBias = weightOrBiasOpt->syn_t;
    if (weightOrBiasOpt->pt_t.scalar_type() != c10::kFloat) {
      storage.push_back(OpBackend::BuildCast(
          op,
          graph,
          synWeightOrBias,
          weightOrBiasOpt->pt_t.sizes(),
          weightOrBiasOpt->pt_t.scalar_type(),
          c10::kFloat));
      synWeightOrBias = storage.back().get();
    }

    weightOrBias_shape = {weightOrBiasOpt->pt_t.numel()};
    storage.push_back(OpBackend::BuildReshape(
        op, graph, synWeightOrBias, weightOrBias_shape, c10::kFloat));
  } else {
    if (isReshapeRequired) {
      weightOrBias_shape = {1, 1, constant_numel};
    } else {
      weightOrBias_shape = {constant_numel};
    }
    storage.push_back(OpBackend::BuildConstant(
        op, graph, constant_value, c10::kFloat, weightOrBias_shape));
  }
  return storage.back().get();
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

  // Since G1 uses TPC kernels, W/B tensors have to be 3D in affine mode.
  // This also applies to F16 on G2. For BF16 and F32 CGUIDs are used on G2
  // so there is no need to change shape of W/B tensors. For G3 affine tests
  // are skipped.
  auto is_reshape_for_tpc_kernels_required = use_tpc_affine_path &&
      (habana::HPURegistrar::get_device().type() !=
           synDeviceType::synDeviceGaudi2 ||
       input.pt_t.scalar_type() == torch::kFloat16);

  int64_t normalized_shape_numel = c10::multiply_integers(
      normalized_shape.cbegin(), normalized_shape.cend());

  int64_t weightOrBias_constant_numel = use_tpc_affine_path
      ? input_shape[input_ndim - 1]
      : normalized_shape_numel;

  std::vector<sh::tensor> storage;
  // Manual handling of reserved size - maximum number of calls to
  // storage.push_back
  storage.reserve(5);

  std::vector<int64_t> weightOrBias_shape = {};
  synTensor synWeight = CreateLayerNormBiasWeightTensor(
      this,
      graph,
      storage,
      input,
      weightOpt,
      weightOrBias_constant_numel,
      1.0f,
      weightOrBias_shape,
      is_reshape_for_tpc_kernels_required);

  synTensor synBias = CreateLayerNormBiasWeightTensor(
      this,
      graph,
      storage,
      input,
      biasOpt,
      weightOrBias_constant_numel,
      0.0f,
      weightOrBias_shape,
      is_reshape_for_tpc_kernels_required);

  synTensor synInput = input.syn_t;

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
        graph, synInput, input_reshaped_shape, input.pt_t.scalar_type()));
    synInput = storage.back().get();
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

  auto metas = LayerNormHabanaMeta(stack);
  int64_t mean_rstd_shape[] = {1, 1, m, 1};

  std::vector<NodeAttr::NodeOutputAttr> node_output_attr;
  for (size_t i = 0; i < metas.size(); ++i) {
    c10::ScalarType outputType = metas[i].dtype;
    if (use_tpc_affine_path) {
      node_output_attr.push_back({metas[i].shape, outputType, i});
    } else {
      node_output_attr.push_back(
          {i == 0 ? input_reshaped_shape : mean_rstd_shape, outputType});
    }
  }

  auto ln = BuildOp(
      graph,
      guid_,
      {synInput, synBias, synWeight},
      std::move(node_output_attr),
      paramsPtr,
      paramsSize);

  for (size_t i = 0; i < ln.size(); ++i) {
    if (use_tpc_affine_path) {
      syn_out(i) = std::move(ln[i]);
    } else {
      auto reshaped =
          ReshapeHelper(graph, ln[i].get(), metas[i].shape, metas[i].dtype, i);
      syn_out(i) = std::move(reshaped);
    }
  }
}

sizes_vec LayerNormBwdOutputShape(const at::Stack& stack) {
  auto input = stack[1].toTensor();
  auto input_size = input.sizes().vec();

  auto weight_size = stack[2].toIntList().vec();

  return {input_size, weight_size, weight_size};
}

OutputMetaDataVector LayerNormBwdMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto dtype = self.scalar_type();
  auto shapes = LayerNormBwdOutputShape(stack);
  OutputMetaDataVector metaVec(3);
  metaVec[0].shape = shapes[0];
  metaVec[0].dtype = self.scalar_type();
  metaVec[1].shape = shapes[1];
  metaVec[1].dtype = dtype;
  metaVec[2].shape = shapes[2];
  metaVec[2].dtype = dtype;
  return metaVec;
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
  storage.reserve(8);

  for (int i = 0; i < 2; ++i) {
    const auto& src = (i == 0) ? grad_out : input;
    storage.push_back(
        ReshapeHelper(graph, src.syn_t, sizes_as_4D, src.pt_t.scalar_type()));
  }

  synTensor grad_out_as_4D = storage[0].get();
  synTensor input_as_4D = storage[1].get();

  std::vector<int64_t> weightShape = {};
  synTensor synWeight = CreateLayerNormBiasWeightTensor(
      this,
      graph,
      storage,
      input,
      weightOpt,
      c10::multiply_integers(
          normalized_shape.cbegin(), normalized_shape.cend()),
      1.0f,
      weightShape,
      false);

  std::array<int64_t, 4> mean_rstd_as_4D = {1, 1, m, 1};
  std::array<unsigned, 2> storage_indices = {};
  for (size_t i = 0; i < storage_indices.size(); ++i) {
    const auto& src = (i == 0) ? mean : rstd;
    storage.push_back(ReshapeHelper(
        graph, src.syn_t, mean_rstd_as_4D, src.pt_t.scalar_type()));

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

  synTensor mean_as_4D = storage[storage_indices[0]].get();
  synTensor rstd_as_4D = storage[storage_indices[1]].get();

  auto metas = LayerNormBwdMeta(stack);

  ns_LayerNormKernel::Params params;
  params.epsValid = false;

  auto lnbwd = BuildOp(
      graph,
      guid_,
      {input_as_4D, grad_out_as_4D, mean_as_4D, rstd_as_4D, synWeight},
      {{sizes_as_4D, metas[0].dtype},
       {weightShape, c10::kFloat},
       {weightShape, c10::kFloat}},
      &params,
      sizeof(params));

  for (size_t i = 1; i < lnbwd.size(); ++i) {
    if (metas[i].dtype != c10::kFloat) {
      lnbwd[i] = CastHelper(
          graph, lnbwd[i].get(), weightShape, c10::kFloat, metas[i].dtype);
    }
  }

  static std::array<int, 3> outIds = {0, 2, 1};
  for (size_t i = 0; i < outIds.size(); ++i) {
    auto reshaped = ReshapeHelper(
        graph, lnbwd[i].get(), metas[i].shape, metas[i].dtype, outIds[i]);
    syn_out(outIds[i]) = std::move(reshaped);
  }
}

OutputMetaDataVector WeightNormMeta(const at::Stack& stack) {
  const torch::Tensor& v_in = stack_tensor(stack, 0);
  const torch::Tensor& g_in = stack_tensor(stack, 1);
  auto dim = stack.at(2).toInt();
  auto shapes = at::infer_size(v_in.sizes(), g_in.sizes());
  auto norm_shapes = v_in.sizes()[dim];

  OutputMetaDataVector metaVec(2);
  metaVec[0].shape = v_in.sizes().vec();
  metaVec[1].shape = g_in.sizes().vec();

  metaVec[0].dtype = v_in.scalar_type();
  metaVec[1].dtype = g_in.scalar_type();
  return metaVec;
}

void WeightNormOp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto metas = WeightNormMeta(stack);
  auto v_in = stack_tensor(stack, 0);
  auto g_in = stack_tensor(stack, 1);
  auto dim = stack.at(2).toInt();

  /*
  NOTE:
  We use the CPU implementation that follows the "non-fused" (ie., assumes
  can_use_fused=0) path.
  */
  TORCH_CHECK(
      v_in.device().type() == g_in.device().type(),
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
        {{metas[1].shape, dtype, 1}},
        false));
  } else {
    normOp.emplace_back(NormCommon(
        this,
        graph,
        syn_in(0),
        metas[1].dtype,
        v_in,
        dim_to_norm,
        false,
        ord,
        {{metas[1].shape, metas[1].dtype, 1}},
        false));
  }

  auto divOp = BuildOp(
      graph,
      get_guid_with_precision("div_fwd", ScalarType()),
      {syn_in(1), normOp[0].get()},
      {{metas[1].shape, metas[1].dtype}});
  auto mulOp = BuildOp(
      graph,
      get_guid_with_precision("mult_fwd", ScalarType()),
      {syn_in(0), divOp.at(0).get()},
      {{metas[0].shape, metas[0].dtype, 0}});

  syn_out(0) = std::move(mulOp[0]);
  syn_out(1) = std::move(normOp[0]);
}

OutputMetaDataVector WeightNormBwdMeta(const at::Stack& stack) {
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

  OutputMetaDataVector metaVec(2);
  metaVec[0].shape = at::infer_size(grad_w.sizes(), saved_v.sizes());
  metaVec[1].shape = bcast_size;
  metaVec[0].dtype = saved_v.scalar_type();
  metaVec[1].dtype = saved_g.scalar_type();
  return metaVec;
}

void WeightNormBwdOp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const torch::Tensor& grad_w = stack_tensor(stack, 0);
  const torch::Tensor& saved_v = stack_tensor(stack, 1);
  const torch::Tensor& saved_g = stack_tensor(stack, 2);
  const torch::Tensor& saved_norms = stack_tensor(stack, 3);
  auto dim = stack.at(4).toInt();

  const auto metas = WeightNormBwdMeta(stack);

  // It is expected that both outputs have the same dtype
  const auto commonOutDtype = metas[0].dtype;

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

  std::vector<synapse_helpers::tensor> norms_cast;
  synTensor saved_norms_syn_tensor = syn_in(3);
  if (saved_norms.scalar_type() != commonOutDtype) {
    norms_cast.emplace_back(CastHelper(
        graph,
        saved_norms_syn_tensor,
        saved_norms.sizes(),
        saved_norms.scalar_type(),
        commonOutDtype));
    saved_norms_syn_tensor = norms_cast[0].get();
  }
  std::vector<synapse_helpers::tensor> per_dim_sums;
  std::vector<synapse_helpers::tensor> divOp21;
  std::vector<synapse_helpers::tensor> mulOp22;
  std::vector<synapse_helpers::tensor> divOp23;
  std::vector<synapse_helpers::tensor> mulOp24;
  std::vector<synapse_helpers::tensor> subOp25;
  std::vector<synapse_helpers::tensor> grad_v;
  std::vector<synapse_helpers::tensor> grad_g;
  std::vector<int64_t> bcast_size = metas[1].shape;

  auto outsize_mulOp11 = at::infer_size(grad_w.sizes(), saved_v.sizes());
  auto mulOp11 = BuildOp(
      graph,
      get_guid_with_precision("mult_fwd", commonOutDtype),
      {syn_in(0), syn_in(1)},
      {{outsize_mulOp11, commonOutDtype}});

  std::vector<int64_t> reshape_outshape;
  // Analytic backward path using differentiable primitive ops
  if (dim == 0) {
    reshape_outshape.push_back(saved_v.size(0));
    if (grad_w.numel() > saved_v.numel()) {
      reshape_outshape.push_back(grad_w.numel() / saved_v.size(0));
    } else {
      reshape_outshape.push_back(saved_v.numel() / saved_v.size(0));
    }

  } else {
    if (grad_w.numel() > saved_v.numel()) {
      reshape_outshape.push_back(grad_w.numel() / last_size);
    } else {
      reshape_outshape.push_back(saved_v.numel() / last_size);
    }
    reshape_outshape.push_back(last_size);
  }

  auto reshapeOp12 =
      ReshapeHelper(graph, mulOp11[0].get(), reshape_outshape, commonOutDtype);

  ns_Reduction::Params reduce_params{};
  int axis = dim == 0 ? 1 : 0;

  int reductionDimension = reshape_outshape.size() - axis - 1;
  reduce_params.reductionDimension = reductionDimension;

  per_dim_sums = BuildOp(
      graph,
      get_guid_with_precision("reduce_sum_fwd", commonOutDtype),
      {reshapeOp12.get()},
      {{bcast_size, commonOutDtype}},
      &reduce_params,
      sizeof(reduce_params));

  auto outsize_divOp21 = at::infer_size(saved_g.sizes(), saved_norms.sizes());
  divOp21 = BuildOp(
      graph,
      get_guid_with_precision("div_fwd", commonOutDtype),
      {syn_in(2), saved_norms_syn_tensor},
      {{outsize_divOp21, commonOutDtype}});

  mulOp22 = BuildOp(
      graph,
      get_guid_with_precision("mult_fwd", commonOutDtype),
      {saved_norms_syn_tensor, saved_norms_syn_tensor},
      {{saved_norms.sizes(), commonOutDtype}});
  divOp23 = BuildOp(
      graph,
      get_guid_with_precision("div_fwd", commonOutDtype),
      {per_dim_sums[0].get(), mulOp22[0].get()},
      {{bcast_size, commonOutDtype}});

  auto outsize_mulOp24 = at::infer_size(saved_v.sizes(), bcast_size);
  mulOp24 = BuildOp(
      graph,
      get_guid_with_precision("mult_fwd", commonOutDtype),
      {syn_in(1), divOp23[0].get()},
      {{saved_v.sizes(), commonOutDtype}});

  auto outsize_subOp25 = at::infer_size(grad_w.sizes(), saved_v.sizes());
  subOp25 = BuildOp(
      graph,
      get_guid_with_precision("sub_fwd", commonOutDtype),
      {syn_in(0), mulOp24[0].get()},
      {{outsize_subOp25, commonOutDtype}});

  auto outsize_grad_v = at::infer_size(saved_g.sizes(), outsize_subOp25);
  grad_v = BuildOp(
      graph,
      get_guid_with_precision("mult_fwd", commonOutDtype),
      {divOp21[0].get(), subOp25[0].get()},
      {{outsize_grad_v, commonOutDtype, 0}});
  grad_g = BuildOp(
      graph,
      get_guid_with_precision("div_fwd", commonOutDtype),
      {per_dim_sums[0].get(), saved_norms_syn_tensor},
      {{bcast_size, commonOutDtype, 1}});

  syn_out(0) = std::move(grad_v.at(0));
  syn_out(1) = std::move(grad_g.at(0));
}

static std::shared_ptr<void> FillReduceSumMultiDimParamsGroupNorm(
    int ndims,
    size_t& size) {
  PARAMS_STUB(ns_Reduction::ParamsV2);
  unsigned maskval = 0;
  for (int i = 0; i < ndims; ++i) {
    if (i != 1)
      maskval |= (1 << (ndims - i - 1)); // (ndims-i-1) is TPC order
  }

  params->reductionDimensionMask = maskval;
  params->keepDim = false;
  return params;
}

sizes_vec NativeGroupNormBwdOutputShape(const at::Stack& stack) {
  auto input = stack[1].toTensor();
  auto input_size = input.sizes().vec();

  int weight_size = stack[6].toInt();

  return {input_size, {weight_size}, {weight_size}};
}

OutputMetaDataVector GroupNormMeta(const at::Stack& stack) {
  constexpr unsigned OUTPUTS_NUMBER = 3;
  auto input = stack_tensor(stack, 1);
  auto shapes = NativeGroupNormBwdOutputShape(stack);
  OutputMetaDataVector metaVec(OUTPUTS_NUMBER);

  for (unsigned i = 0; i < OUTPUTS_NUMBER; ++i) {
    metaVec[i].shape = shapes[i];
    metaVec[i].dtype = i == 0 ? input.scalar_type() : at::ScalarType::Float;
  }

  return metaVec;
}

std::vector<sh::tensor> GetBnFwdOutGroupNorm(
    OpBackend* op,
    sh::graph& graph,
    synTensor input,
    const std::vector<int64_t>& outputShape,
    const at::ScalarType outputDType,
    const std::vector<int64_t>& constantShape) {
  auto bn_weight_running_var = OpBackend::BuildConstant(
      op, graph, 1.0f, at::ScalarType::Float, constantShape);
  auto bn_bias_running_mean = OpBackend::BuildConstant(
      op, graph, 0.0f, at::ScalarType::Float, constantShape);

  PARAMS_STUB_VARS(ns_BatchNormKernel::ParamsV2, params, size);
  params->momentum = 0.0;
  params->epsilon = 1e-5;
  params->isTraining = true;

  return OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("batch_norm_fwd", outputDType),
       {input,
        bn_bias_running_mean.get(),
        bn_weight_running_var.get(),
        bn_bias_running_mean.get(),
        bn_weight_running_var.get()},
       {NodeAttr::NodeOutputAttr{{outputShape}, outputDType},
        NodeAttr::NodeOutputAttr{constantShape, at::ScalarType::Float},
        NodeAttr::NodeOutputAttr{constantShape, at::ScalarType::Float},
        NodeAttr::NodeOutputAttr{constantShape, at::ScalarType::Float},
        NodeAttr::NodeOutputAttr{constantShape, at::ScalarType::Float}},
       params.get(),
       size});
}

sh::tensor PreprocessMeanRstdInputGroupNorm(
    OpBackend* op,
    sh::graph& graph,
    const at::Tensor& ptTensor,
    synTensor synTensor) {
  const int64_t numel = ptTensor.numel();

  auto tensor_reshaped = OpBackend::BuildReshape(
      op, graph, synTensor, numel, ptTensor.scalar_type());

  return ptTensor.scalar_type() != at::ScalarType::Float
      ? OpBackend::BuildCast(
            op,
            graph,
            tensor_reshaped.get(),
            numel,
            ptTensor.scalar_type(),
            at::ScalarType::Float)
      : std::move(tensor_reshaped);
}

std::vector<int64_t> GetBnInputShapeGroupNorm(
    const at::Tensor& tensor,
    const int64_t Nmod) {
  const int64_t M = tensor.numel() / Nmod;
  const auto dims = tensor.dim();

  std::vector<int64_t> bnInputShape(dims, 1);
  bnInputShape[1] = Nmod;
  bnInputShape[dims - 1] = M;
  return bnInputShape;
}

std::vector<int64_t> GetBnMeanWeightShapeGroupNorm(
    const int64_t dims,
    const int64_t numel) {
  std::vector<int64_t> bn_fwd_mean_shape(dims, 1);
  bn_fwd_mean_shape[1] = numel;
  return bn_fwd_mean_shape;
}

std::vector<int64_t> GetBnMeanShapeGroupNorm(
    const int64_t dims,
    const int64_t Nmod) {
  return GetBnMeanWeightShapeGroupNorm(dims, Nmod);
}

std::vector<int64_t> GetBnWeightShapeGroupNorm(
    const int64_t dims,
    const int64_t C) {
  return GetBnMeanWeightShapeGroupNorm(dims, C);
}

void NativeGroupNormBwdHabanaOperator::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "NativeGroupNormBwdHabanaOperator::AddNode");
  auto grad_out = getNextInput<TensorsPair>(stackGetter);
  auto input = getNextInput<TensorsPair>(stackGetter);
  auto mean = getNextInput<TensorsPair>(stackGetter);
  auto rstd = getNextInput<TensorsPair>(stackGetter);
  auto weightOpt = getNextInput<at::optional<TensorsPair>>(stackGetter);
  auto N = getNextInput<int>(stackGetter);
  auto C = getNextInput<int>(stackGetter);
  auto HxW = getNextInput<int>(stackGetter);
  auto num_groups = getNextInput<int>(stackGetter);
  const auto Nmod = N * num_groups;

  TORCH_CHECK(grad_out.pt_t.numel() == N * C * HxW);
  TORCH_CHECK(input.pt_t.numel() == N * C * HxW);
  TORCH_CHECK(mean.pt_t.numel() == Nmod);
  TORCH_CHECK(rstd.pt_t.numel() == Nmod);

  const bool use_bn_fwd_in_gn_bwd =
      GET_ENV_FLAG_NEW(PT_HPU_USE_BN_FWD_IN_GN_BWD);

  const auto metas = OutputMeta(stack);

  std::vector<int64_t> bn_input_shape =
      GetBnInputShapeGroupNorm(input.pt_t, Nmod);
  std::vector<int64_t> weight_shape =
      GetBnWeightShapeGroupNorm(input.pt_t.dim(), C);
  std::vector<int64_t> mean_shape =
      GetBnMeanShapeGroupNorm(input.pt_t.dim(), Nmod);
  std::vector<int64_t> mean_shape_squeezed{Nmod};

  auto mean_casted =
      PreprocessMeanRstdInputGroupNorm(this, graph, mean.pt_t, mean.syn_t);
  auto rstd_casted =
      PreprocessMeanRstdInputGroupNorm(this, graph, rstd.pt_t, rstd.syn_t);

  auto bn_fwd_input =
      ReshapeHelper(graph, input.syn_t, bn_input_shape, metas[0].dtype);

  std::vector<sh::tensor> bn_fwd_out;
  if (use_bn_fwd_in_gn_bwd) {
    bn_fwd_out = GetBnFwdOutGroupNorm(
        this,
        graph,
        bn_fwd_input.get(),
        bn_input_shape,
        metas[0].dtype,
        mean_shape_squeezed);

  } else {
    auto mean_for_bn_fwd = ReshapeHelper(
        graph, mean_casted.get(), mean_shape, at::ScalarType::Float);
    auto rstd_for_bn_fwd = ReshapeHelper(
        graph, rstd_casted.get(), mean_shape, at::ScalarType::Float);

    auto subOp = BuildOp(
        graph,
        get_guid_with_precision("sub_fwd", metas[0].dtype),
        {bn_fwd_input.get(), mean_for_bn_fwd.get()},
        {{bn_input_shape, metas[0].dtype}});
    bn_fwd_out = BuildOp(
        graph,
        get_guid_with_precision("mult_fwd", metas[0].dtype),
        {subOp[0].get(), rstd_for_bn_fwd.get()},
        {{bn_input_shape, metas[0].dtype}});
  }

  const auto bn_fwd_out_reshaped = ReshapeHelper(
      graph, bn_fwd_out[0].get(), input.pt_t.sizes().vec(), metas[0].dtype);

  const auto weightReshaped = weightOpt
      ? ReshapeHelper(
            graph, weightOpt->syn_t, weight_shape, at::ScalarType::Float)
      : ConstantHelper(graph, 1.0f, at::ScalarType::Float, weight_shape);

  auto grad_weight_mult = BuildOp(
      graph,
      get_guid_with_precision("mult_fwd", metas[0].dtype),
      {grad_out.syn_t, weightReshaped.get()},
      {{metas[0].shape, metas[0].dtype}});

  auto grad_weight_mult_reshape = ReshapeHelper(
      graph, grad_weight_mult[0].get(), bn_input_shape, metas[0].dtype);

  const auto bn_bwd_grad =
      grad_out.pt_t.scalar_type() != input.pt_t.scalar_type()
      ? CastHelper(
            graph,
            grad_weight_mult_reshape.get(),
            bn_input_shape,
            grad_out.pt_t.scalar_type(),
            input.pt_t.scalar_type())
      : std::move(grad_weight_mult_reshape);

  auto bn_bwd_in =
      ReshapeHelper(graph, input.syn_t, bn_input_shape, metas[0].dtype);
  auto bn_gamma =
      ConstantHelper(graph, 1.0f, at::ScalarType::Float, mean_shape_squeezed);

  PARAMS_STUB_VARS(ns_BatchNormKernel::ParamsV2, params, size);
  params->epsilon = 1e-05;
  params->isTraining = true;
  params->momentum = 0.0;
  auto bn_bwd = BuildOp(
      graph,
      get_guid_with_precision("batch_norm_bwd", metas[0].dtype),
      {bn_bwd_in.get(),
       bn_bwd_grad.get(),
       mean_casted.get(),
       rstd_casted.get(),
       bn_gamma.get()},
      {{bn_input_shape, metas[0].dtype},
       {mean_shape_squeezed, at::ScalarType::Float},
       {mean_shape_squeezed, at::ScalarType::Float}},
      params.get(),
      size);

  auto output =
      ReshapeHelper(graph, bn_bwd[0].get(), metas[0].shape, metas[0].dtype, 0);

  size_t reduce_size = 0;
  auto reduce_params =
      FillReduceSumMultiDimParamsGroupNorm(input.pt_t.dim(), reduce_size);

  // WEIGHT
  auto grad_bn_fwd_mult = BuildOp(
      graph,
      get_guid_with_precision("mult_fwd", metas[0].dtype),
      {grad_out.syn_t, bn_fwd_out_reshaped.get()},
      {{metas[0].shape, metas[0].dtype}});

  auto grad_gamma = BuildOp(
      graph,
      get_guid_with_precision(
          "reduce_sum_multi_dim_fwd", at::ScalarType::Float),
      {grad_bn_fwd_mult[0].get()},
      {{metas[1].shape, metas[1].dtype, 1}},
      reduce_params.get(),
      reduce_size);

  // BIAS
  auto grad_beta = BuildOp(
      graph,
      get_guid_with_precision(
          "reduce_sum_multi_dim_fwd", at::ScalarType::Float),
      {grad_out.syn_t},
      {{metas[2].shape, metas[2].dtype, 2}},
      reduce_params.get(),
      reduce_size);

  syn_out(0) = std::move(output);
  syn_out(1) = std::move(grad_gamma[0]);
  syn_out(2) = std::move(grad_beta[0]);
}

} // namespace habana
