/**
 * Copyright (c) 2021-2025 Intel Corporation
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
#include "backend/helpers/habana_types.h"
#include "generated/backend/native_layer_norm.h"
#include "generated/backend/native_layer_norm_backward.h"

#include "backend/helpers/cast_sequence.h"

namespace habana {

namespace sh = synapse_helpers;

FillParamsT FillNativeLayerNormPtParams(const at::Stack& stack) {
  const auto eps = stack.at(4).toDouble();
  const auto normalized_ndim = stack.at(1).toIntList().size();
  PARAMS_STUB(ns_LayerNormKernel::ParamsPt);
  params->eps = static_cast<float>(eps);
  params->epsValid = true;
  params->normalizedShapeDims = normalized_ndim;

  return paramsT;
}

FillParamsT FillNativeLayerNormParams(const at::Stack& stack) {
  const auto eps = stack.at(4).toDouble();
  PARAMS_STUB(ns_LayerNormKernel::Params);
  params->eps = static_cast<float>(eps);
  params->epsValid = true;

  return paramsT;
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
    const std::optional<TensorsPair>& weightOrBiasOpt,
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

using namespace std::literals;

void LayerNormHabanaOperator::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "LayerNormHabanaOperator::AddNode");
  auto input = stackGetter.getNextInput<TensorsPair>();
  auto normalized_shape = stackGetter.getNextInput<std::vector<int64_t>>();
  auto weightOpt = stackGetter.getNextInput<std::optional<TensorsPair>>();
  auto biasOpt = stackGetter.getNextInput<std::optional<TensorsPair>>();

  auto metas = LayerNormHabanaMeta(stack);
  // CGUID layer_norm_fwd_pt should be used when we call underneath the TPC
  // Kernel layer_norm, instead of the HabanaNorm flow

  const bool shouldCallTPCLayerNorm =
      habana::HPUDeviceContext::get_device().type() ==
          synDeviceType::synDeviceGaudi ||
      input.pt_t.scalar_type() == torch::kFloat16 ||
      GetExecutionMode() == habana_helpers::HabanaFrontendTypes::EAGER;

  if (shouldCallTPCLayerNorm) {
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

    const auto params = FillParams(stack);

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
        params.ptr(),
        params.size());

    for (size_t i = 0; i < ln.size(); ++i) {
      syn_out(i) = std::move(ln[i]);
    }

  } else {
    const auto input_shape = input.pt_t.sizes();
    const auto input_ndim = input.pt_t.dim();
    const int normalized_ndim = normalized_shape.size();
    const int64_t normalized_shape_numel = c10::multiply_integers(
        normalized_shape.cbegin(), normalized_shape.cend());

    std::vector<int64_t> weightOrBias_constant_shape =
        std::vector<int64_t>{normalized_shape_numel};

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
    const int64_t m = c10::multiply_integers(
        input_shape.cbegin(), input_shape.cbegin() + axis);
    const int64_t n =
        c10::multiply_integers(input_shape.cbegin() + axis, input_shape.cend());

    const int64_t input_reshaped_shape[] = {1, 1, m, n};
    auto reshapedInput = ReshapeHelper(
        graph, input.syn_t, input_reshaped_shape, input.pt_t.scalar_type());

    int64_t mean_rstd_shape[] = {1, 1, m, 1};
    std::vector<NodeAttr::NodeOutputAttr> node_output_attr;
    for (size_t i = 0; i < metas.size(); ++i) {
      c10::ScalarType outputType = metas[i].dtype;
      node_output_attr.push_back(
          {i == 0 ? input_reshaped_shape : mean_rstd_shape, outputType});
    }

    const auto params = FillNativeLayerNormParams(stack);

    auto ln = BuildOp(
        graph,
        get_guid_with_precision("layer_norm_fwd"sv, metas[0].dtype),
        {reshapedInput.get(), synBias, synWeight},
        std::move(node_output_attr),
        params.ptr(),
        params.size());

    for (size_t i = 0; i < ln.size(); ++i) {
      auto reshaped =
          ReshapeHelper(graph, ln[i].get(), metas[i].shape, metas[i].dtype, i);
      syn_out(i) = std::move(reshaped);
    }
  }
}

FillParamsT FillNativeLayerNormBwdParams(const at::Stack& stack) {
  const auto normalized_ndim = stack.at(2).toIntList().size();
  PARAMS_STUB(ns_LayerNormKernel::ParamsPt);
  params->epsValid = false;
  params->normalizedShapeDims = normalized_ndim;
  return paramsT;
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
  StackGetter stackGetter(this, stack, "LayerNormBwdHabanaOperator::AddNode");
  auto grad_out = stackGetter.getNextInput<TensorsPair>();
  auto input = stackGetter.getNextInput<TensorsPair>();
  auto normalized_shape = stackGetter.getNextInput<std::vector<int64_t>>();
  auto mean = stackGetter.getNextInput<TensorsPair>();
  auto rstd = stackGetter.getNextInput<TensorsPair>();
  auto weightOpt = stackGetter.getNextInput<std::optional<TensorsPair>>();

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

    const auto params = FillParams(stack);

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
        params.ptr(),
        params.size());

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
        get_guid_with_precision("layer_norm_bwd"sv, metas[0].dtype),
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
