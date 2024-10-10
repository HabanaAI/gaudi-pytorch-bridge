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
#include "generated/backend/native_layer_norm.h"
#include "generated/backend/native_layer_norm_backward.h"

#include "backend/helpers/cast_sequence.h"

namespace habana {

namespace sh = synapse_helpers;

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

} // namespace habana
