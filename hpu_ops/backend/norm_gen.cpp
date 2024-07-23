/*******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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
#include "backend/helpers/cast_sequence.h"
#include "generated/backend/_weight_norm_interface.h"
#include "generated/backend/_weight_norm_interface_backward.h"
#include "generated/backend/linalg_vector_norm.h"
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

ns_ReduceLpV2::ParamsV2 FillPFormNormOpParams(
    const int64_t ndims,
    at::IntArrayRef dims,
    bool keepDim,
    const at::Scalar& ord) {
  ns_ReduceLpV2::ParamsV2 params;
  (ns_Reduction::ParamsV2&)params = FillReductionParams(ndims, dims, keepDim);

  if (ord.isFloatingPoint()) {
    get<float>(params.p) = ord.to<float>();
    params.typeOfP = TYPE_P_IS_FLOAT;
  } else {
    get<int>(params.p) = ord.to<int>();
    params.typeOfP = TYPE_P_IS_INT;
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

void VecNormCheck(
    const torch::Tensor& self,
    const at::ScalarType& dtype,
    at::Scalar ord,
    c10::IntArrayRef dim) {
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
    at::IntArrayRef dim,
    const bool keepdim,
    const at::Scalar& ord,
    std::vector<NodeAttr::NodeOutputAttr> output_attr,
    const bool is_vec_norm) {
  auto p = (ord.isFloatingPoint()) ? ord.toFloat() : ord.toInt();
  auto self_shape = self.sizes().vec();

  if (is_vec_norm)
    VecNormCheck(self, dtype, ord, dim);
  else
    NormCheck(dtype);

  if (self.numel() == 0) {
    at::Scalar s = (p < 0) && is_vec_norm ? INF : 0;
    return OpBackend::BuildConstant(op, graph, s, dtype, 1, 0);
  }

  auto params = FillPFormNormOpParams(self.dim(), dim, keepdim, ord);
  auto reduce_lp_output = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("reduce_Lp_multi_dim_fwd", dtype),
       {input_tensor},
       output_attr,
       &params,
       sizeof(params)});

  return std::move(reduce_lp_output.at(0));
}

void VecNormOp::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto optional_ord = stack.at(1).toOptional<at::Scalar>();
  auto dim =
      stack.at(2).isNone() ? c10::DimVector{} : stack.at(2).toDimVector();
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
  auto dim = stack.at(2).toDimVector();
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

std::shared_ptr<void> FillNativeLayerNormParams(
    const at::Stack& stack,
    size_t& size) {
  const auto eps = stack.at(4).toDouble();
  const auto normalized_ndim = stack.at(1).toIntList().size();
  PARAMS_STUB(ns_LayerNormKernel::ParamsPt);
  params->eps = static_cast<float>(eps);
  params->epsValid = true;
  params->normalizedShapeDims = normalized_ndim;

  return params;
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
  for (size_t i = 0; i < metaVec.size(); ++i) {
    metaVec[i].shape = shapes[i];
    metaVec[i].dtype = i == 0 ? self.scalar_type() : at::ScalarType::Float;
  }
  return metaVec;
}

static synTensor CreateLayerNormBiasWeightTensor(
    OpBackend* op,
    sh::graph& graph,
    std::vector<sh::tensor>& storage,
    const c10::optional<OpBackend::TensorsPair>& weightOrBiasOpt,
    const std::vector<int64_t>& constant_shape,
    float constant_value,
    std::vector<int64_t>& weightOrBias_shape) {
  if (weightOrBiasOpt) {
    synTensor synWeightOrBias = weightOrBiasOpt->syn_t;
    if (habana_helpers::DataTypeToCastType(
            weightOrBiasOpt->pt_t.scalar_type()) !=
        habana_helpers::DataTypeToCastType(c10::kFloat)) {
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
    weightOrBias_shape = constant_shape;
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

  auto metas = LayerNormHabanaMeta(stack);

  if (GetExecutionMode() == habana_helpers::HabanaFrontendTypes::EAGER) {
    const auto input_dim = stack_tensor(stack, 0).dim();

    if (input_dim == 5) {
      SetSynapseLayouts(
          {sh::layouts::SynapseLayoutFormat::WHDCN,
           sh::layouts::SynapseLayoutFormat::DONT_CARE,
           sh::layouts::SynapseLayoutFormat::DONT_CARE},
          {sh::layouts::SynapseLayoutFormat::WHDCN,
           sh::layouts::SynapseLayoutFormat::WHDCN,
           sh::layouts::SynapseLayoutFormat::WHDCN});
    } else if (input_dim == 4) {
      SetSynapseLayouts(
          {sh::layouts::SynapseLayoutFormat::WHCN,
           sh::layouts::SynapseLayoutFormat::DONT_CARE,
           sh::layouts::SynapseLayoutFormat::DONT_CARE},
          {sh::layouts::SynapseLayoutFormat::WHCN,
           sh::layouts::SynapseLayoutFormat::WHCN,
           sh::layouts::SynapseLayoutFormat::WHCN});
    } else if (input_dim == 3) {
      SetSynapseLayouts(
          {sh::layouts::SynapseLayoutFormat::WHN,
           sh::layouts::SynapseLayoutFormat::DONT_CARE,
           sh::layouts::SynapseLayoutFormat::DONT_CARE},
          {sh::layouts::SynapseLayoutFormat::WHN,
           sh::layouts::SynapseLayoutFormat::WHN,
           sh::layouts::SynapseLayoutFormat::WHN});
    }

    size_t size = 0;
    const auto params = FillParams(stack, size);

    std::vector<NodeAttr::NodeOutputAttr> node_output_attr;

    for (size_t i = 0; i < metas.size(); ++i) {
      node_output_attr.push_back({metas[i].shape, metas[i].dtype, i});
    }

    auto ln = BuildOp(
        graph,
        guid_,
        {input.syn_t,
         weightOpt ? weightOpt.value().syn_t : nullptr,
         biasOpt ? biasOpt.value().syn_t : nullptr},
        node_output_attr,
        params.get(),
        size);

    for (size_t i = 0; i < ln.size(); ++i) {
      syn_out(i) = std::move(ln[i]);
    }

  } else {
    auto eps = getNextInput<double>(stackGetter);

    const auto input_shape = input.pt_t.sizes();
    const auto input_ndim = input.pt_t.dim();
    const int normalized_ndim = normalized_shape.size();

    auto use_tpc_affine_path =
        !weightOpt && !biasOpt && (input_ndim == 4) && (normalized_ndim == 3);

    // Since G1 uses TPC kernels, W/B tensors have to be 3D in affine mode.
    // This also applies to F16 on G2. For BF16 and F32 CGUIDs are used on G2
    // so there is no need to change shape of W/B tensors. For G3 situation is
    // the same as for G2.
    auto is_reshape_for_tpc_kernels_required = use_tpc_affine_path &&
        (habana::HPURegistrar::get_device().type() ==
         synDeviceType::synDeviceGaudi);

    int64_t normalized_shape_numel = c10::multiply_integers(
        normalized_shape.cbegin(), normalized_shape.cend());

    int64_t weightOrBias_constant_numel = use_tpc_affine_path
        ? input_shape[input_ndim - 1]
        : normalized_shape_numel;

    std::vector<int64_t> weightOrBias_constant_shape =
        is_reshape_for_tpc_kernels_required
        ? std::vector<int64_t>{1, 1, weightOrBias_constant_numel}
        : (use_tpc_affine_path && input.pt_t.scalar_type() == torch::kFloat16)
            ? normalized_shape
            : std::vector<int64_t>{weightOrBias_constant_numel};

    std::vector<sh::tensor> storage;
    // Manual handling of reserved size - maximum number of calls to
    // storage.push_back
    storage.reserve(5);

    std::vector<int64_t> weightOrBias_shape = {};
    synTensor synWeight = CreateLayerNormBiasWeightTensor(
        this,
        graph,
        storage,
        weightOpt,
        weightOrBias_constant_shape,
        1.0f,
        weightOrBias_shape);

    synTensor synBias = CreateLayerNormBiasWeightTensor(
        this,
        graph,
        storage,
        biasOpt,
        weightOrBias_constant_shape,
        0.0f,
        weightOrBias_shape);

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
    int64_t m = c10::multiply_integers(
        input_shape.cbegin(), input_shape.cbegin() + axis);
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
        get_guid_with_precision("layer_norm_fwd", metas[0].dtype),
        {synInput, synBias, synWeight},
        std::move(node_output_attr),
        paramsPtr,
        paramsSize);

    for (size_t i = 0; i < ln.size(); ++i) {
      if (use_tpc_affine_path) {
        syn_out(i) = std::move(ln[i]);
      } else {
        auto reshaped = ReshapeHelper(
            graph, ln[i].get(), metas[i].shape, metas[i].dtype, i);
        syn_out(i) = std::move(reshaped);
      }
    }
  }
}

std::shared_ptr<void> FillNativeLayerNormBwdParams(
    const at::Stack& stack,
    size_t& size) {
  const auto normalized_ndim = stack.at(2).toIntList().size();
  PARAMS_STUB(ns_LayerNormKernel::ParamsPt);
  params->epsValid = false;
  params->normalizedShapeDims = normalized_ndim;
  return params;
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
  for (size_t i = 0; i < metaVec.size(); ++i) {
    metaVec[i].shape = shapes[i];
    metaVec[i].dtype = dtype;
  }
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

  auto metas = LayerNormBwdMeta(stack);

  if (GetExecutionMode() == habana_helpers::HabanaFrontendTypes::EAGER) {
    const auto input_dim = stack_tensor(stack, 0).dim();

    if (input_dim == 5) {
      SetSynapseLayouts(
          {sh::layouts::SynapseLayoutFormat::WHDCN,
           sh::layouts::SynapseLayoutFormat::WHDCN,
           sh::layouts::SynapseLayoutFormat::WHDCN,
           sh::layouts::SynapseLayoutFormat::WHDCN,
           sh::layouts::SynapseLayoutFormat::DONT_CARE},
          {sh::layouts::SynapseLayoutFormat::WHDCN,
           sh::layouts::SynapseLayoutFormat::DONT_CARE,
           sh::layouts::SynapseLayoutFormat::DONT_CARE});
    } else if (input_dim == 4) {
      SetSynapseLayouts(
          {sh::layouts::SynapseLayoutFormat::WHCN,
           sh::layouts::SynapseLayoutFormat::WHCN,
           sh::layouts::SynapseLayoutFormat::WHCN,
           sh::layouts::SynapseLayoutFormat::WHCN,
           sh::layouts::SynapseLayoutFormat::DONT_CARE},
          {sh::layouts::SynapseLayoutFormat::WHCN,
           sh::layouts::SynapseLayoutFormat::DONT_CARE,
           sh::layouts::SynapseLayoutFormat::DONT_CARE});
    } else if (input_dim == 3) {
      SetSynapseLayouts(
          {sh::layouts::SynapseLayoutFormat::WHN,
           sh::layouts::SynapseLayoutFormat::WHN,
           sh::layouts::SynapseLayoutFormat::WHN,
           sh::layouts::SynapseLayoutFormat::WHN,
           sh::layouts::SynapseLayoutFormat::DONT_CARE},
          {sh::layouts::SynapseLayoutFormat::WHN,
           sh::layouts::SynapseLayoutFormat::DONT_CARE,
           sh::layouts::SynapseLayoutFormat::DONT_CARE});
    }

    size_t size = 0;
    const auto params = FillParams(stack, size);

    std::vector<NodeAttr::NodeOutputAttr> node_output_attr;
    for (size_t i = 0; i < metas.size(); ++i) {
      node_output_attr.push_back({metas[i].shape, metas[i].dtype, i});
    }

    auto lnbwd = BuildOp(
        graph,
        guid_,
        {grad_out.syn_t,
         input.syn_t,
         mean.syn_t,
         rstd.syn_t,
         weightOpt ? weightOpt.value().syn_t : nullptr},
        {node_output_attr},
        params.get(),
        size);

    for (size_t i = 0; i < lnbwd.size(); ++i) {
      syn_out(i) = std::move(lnbwd[i]);
    }

  } else {
    const auto input_shape = input.pt_t.sizes();
    const auto input_ndim = input.pt_t.dim();
    const int normalized_ndim = normalized_shape.size();
    const int axis = input_ndim - normalized_ndim;
    int64_t m = c10::multiply_integers(
        input_shape.cbegin(), input_shape.cbegin() + axis);
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
        weightOpt,
        {c10::multiply_integers(
            normalized_shape.cbegin(), normalized_shape.cend())},
        1.0f,
        weightShape);

    std::array<int64_t, 4> mean_rstd_as_4D = {1, 1, m, 1};
    std::array<unsigned, 2> storage_indices = {};
    for (size_t i = 0; i < storage_indices.size(); ++i) {
      const auto& src = (i == 0) ? mean : rstd;
      storage.push_back(ReshapeHelper(
          graph, src.syn_t, mean_rstd_as_4D, src.pt_t.scalar_type()));

      if (src.pt_t.scalar_type() != c10::kFloat) {
        storage.push_back(BuildCast(
            this,
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

    ns_LayerNormKernel::Params params;
    params.epsValid = false;

    auto lnbwd = BuildOp(
        graph,
        get_guid_with_precision("layer_norm_bwd", metas[0].dtype),
        {input_as_4D, grad_out_as_4D, mean_as_4D, rstd_as_4D, synWeight},
        {{sizes_as_4D, metas[0].dtype},
         {weightShape, c10::kFloat},
         {weightShape, c10::kFloat}},
        &params,
        sizeof(params));

    for (size_t i = 1; i < lnbwd.size(); ++i) {
      if (habana_helpers::DataTypeToCastType(metas[i].dtype) !=
          habana_helpers::DataTypeToCastType(c10::kFloat)) {
        lnbwd[i] = BuildCast(
            this,
            graph,
            lnbwd[i].get(),
            weightShape,
            c10::kFloat,
            metas[i].dtype);
      }
    }

    static std::array<int, 3> outIds = {0, 2, 1};
    for (size_t i = 0; i < outIds.size(); ++i) {
      auto reshaped = ReshapeHelper(
          graph, lnbwd[i].get(), metas[i].shape, metas[i].dtype, outIds[i]);
      syn_out(outIds[i]) = std::move(reshaped);
    }
  }
}

OutputMetaDataVector WeightNormMeta(const at::Stack& stack) {
  const torch::Tensor& v_in = stack_tensor(stack, 0);
  const torch::Tensor& g_in = stack_tensor(stack, 1);

  OutputMetaDataVector metaVec(2);
  metaVec[0].shape = v_in.sizes().vec();
  metaVec[1].shape = g_in.sizes().vec();

  metaVec[0].dtype = v_in.scalar_type();
  metaVec[1].dtype = g_in.scalar_type();
  return metaVec;
}

void WeightNormOp::AddNode(sh::graph& graph, const at::Stack& stack) {
  const auto metas = WeightNormMeta(stack);
  auto v_in = stack_tensor(stack, 0);
  auto g_in = stack_tensor(stack, 1);
  auto dim = stack.at(2).toInt();

  const auto& v_in_dtype = metas[0].dtype;
  const auto& g_in_dtype = metas[1].dtype;
  const auto& v_in_shape = metas[0].shape;
  const auto& g_in_shape = metas[1].shape;

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

  c10::DimVector dims_to_norm;
  dims_to_norm.reserve(v_in.ndimension());
  for (int64_t i = 0; i < v_in.ndimension(); ++i) {
    if (i != dim) // skip given dimension
      dims_to_norm.push_back(i);
  }

  at::Scalar ord = 2.0;

  auto normOp = NormCommon(
      this,
      graph,
      g_in_dtype != v_in_dtype
          ? BuildCast(
                this, graph, syn_in(0), v_in.sizes(), v_in_dtype, g_in_dtype)
                .get()
          : syn_in(0),
      g_in_dtype,
      v_in,
      dims_to_norm,
      g_in_shape.size() == v_in_shape.size(),
      ord,
      {{g_in_shape, g_in_dtype, 1}},
      false);

  auto divOp = BuildOp(
      graph,
      get_guid_with_precision("div_fwd", v_in_dtype),
      {syn_in(1), normOp.get()},
      {{g_in_shape, g_in_dtype}});
  auto mulOp = BuildOp(
      graph,
      get_guid_with_precision("mult_fwd", v_in_dtype),
      {syn_in(0), divOp.at(0).get()},
      {{v_in_shape, v_in_dtype, 0}});

  syn_out(0) = std::move(mulOp[0]);
  syn_out(1) = std::move(normOp);
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

void WeightNormBwdOp::AddNode(sh::graph& graph, const at::Stack& stack) {
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

  std::vector<sh::tensor> norms_cast;
  synTensor saved_norms_syn_tensor = syn_in(3);
  if (saved_norms.scalar_type() != commonOutDtype) {
    norms_cast.emplace_back(BuildCast(
        this,
        graph,
        saved_norms_syn_tensor,
        saved_norms.sizes(),
        saved_norms.scalar_type(),
        commonOutDtype));
    saved_norms_syn_tensor = norms_cast[0].get();
  }
  std::vector<sh::tensor> per_dim_sums;
  std::vector<sh::tensor> divOp21;
  std::vector<sh::tensor> mulOp22;
  std::vector<sh::tensor> divOp23;
  std::vector<sh::tensor> mulOp24;
  std::vector<sh::tensor> subOp25;
  std::vector<sh::tensor> grad_v;
  std::vector<sh::tensor> grad_g;
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
} // namespace habana
