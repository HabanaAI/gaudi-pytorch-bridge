/**
* Copyright (c) 2021-2024 Intel Corporation
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

#include <perf_lib_layer_params.h>
#include <torch/script.h>

#include "backend/create_pt_tensor.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/kernel/hpu_shape_inference.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/roi_align_kernels.h"

using namespace habana;

InferOutputMetaRetType RoiAlignFwdOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  InferOutputMetaRetType out;
  auto input = inputs[0].toTensor();
  auto num_rois = inputs[2].toTensor();
  auto output_h = inputs[3].toInt();
  auto output_w = inputs[4].toInt();
  std::vector<int64_t> out_shape;
  auto channel_dim = synapse_helpers::layouts::INPUT_C_IDX;
  out_shape.assign(
      {num_rois.sizes()[0], input.sizes()[channel_dim], output_h, output_w});
  out.AddOutputTensor(TensorMetaData(
      out_shape,
      HabanaOperator::CalculateStrides(
          out_shape, input.suggest_memory_format()),
      input.scalar_type(),
      input.suggest_memory_format()));

  out.AddShapeTensor(TensorMetaData(
      out_shape,
      HabanaOperator::CalculateStrides(
          out_shape, input.suggest_memory_format()),
      input.scalar_type(),
      input.suggest_memory_format()));
  return out;
}

void RoiAlignFwdOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 9,
      "Incorrect size of inputs expected for RoiAlignFwd operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for RoiAlign operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg1 expected to be tensor for RoiAlign operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input arg1 expected to be tensor for RoiAlign operator");

  auto input = inputs[0].toTensor();
  auto rois = inputs[1].toTensor();
  auto num_rois = inputs[2].toTensor();
  auto output_h = inputs[3].toInt();
  auto output_w = inputs[4].toInt();
  auto mode = inputs[5].toInt();
  auto sampling_ratio = inputs[6].toInt();
  auto spatial_scale = inputs[7].toScalar().toFloat();
  auto aligned = inputs[8].toBool();

  ns_RoiAlignKernel::ParamsAlignment roi_params{};
  roi_params.mode =
      mode ? RoiAlignMode_t::ROI_ALIGN_MAX : RoiAlignMode_t::ROI_ALIGN_AVG;
  roi_params.sampling_ratio = sampling_ratio;
  roi_params.spatial_scale = spatial_scale;
  roi_params.aligned = aligned;
  std::vector<int64_t> out_shape;
  auto channel_dim = synapse_helpers::layouts::INPUT_C_IDX;
  out_shape.assign(
      {num_rois.sizes()[0], input.sizes()[channel_dim], output_h, output_w});

  auto output = habana::createPTTensor(
      input, out_shape, input.options(), output_metadata.at(0).persistent);

  // Allocate Shape Tensor
  if (graph.is_dynamic_graph()) {
    AllocateSynapseShapeTensor(graph, output);
  }

  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &roi_params, sizeof(roi_params));
}

InferOutputMetaRetType RoiAlignBwdOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  InferOutputMetaRetType out;
  auto rois = inputs[1].toTensor();
  torch::jit::Stack temp(inputs);

  if (rois.scalar_type() == c10::ScalarType::BFloat16) {
    auto cast_op =
        make_operator<CastOperator>(rois.device().index(), "cast_bf16_to_f32");
    torch::jit::Stack castOp_stack = {
        inputs[1].toTensor(), c10::ScalarType::Float};
    out.call_InferOutputMeta(cast_op, castOp_stack);
  }

  auto quad_tree_op = make_operator<habana::QuadTreeFwdImplOperator>(
      this->p_context_->device_id_, c10::ScalarType::Float);
  out.call_InferOutputMeta(quad_tree_op, inputs);

  auto roi_bwd_impl_op = make_operator<habana::RoiAlignBwdImplOperator>(
      this->p_context_->device_id_, inputs[0].toTensor().scalar_type());
  auto& roi_bwd_impl_op_out = out.call_InferOutputMeta(roi_bwd_impl_op, inputs);
  auto roi_bwd_impl_op_tensor = roi_bwd_impl_op_out.GetOutputTensor()[0];
  out.MoveToOutput(std::move(roi_bwd_impl_op_tensor));

  return out;
}

void RoiAlignBwdOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  auto rois = inputs[1].toTensor();
  // quad_tree supports f32 only, therefore rois need to be casted to f32 before
  // feeding into quad_tree
  std::shared_ptr<HabanaOperator> cast_op;
  if (rois.scalar_type() == c10::ScalarType::BFloat16) {
    cast_op =
        make_operator<CastOperator>(rois.device().index(), "cast_bf16_to_f32");
    cast_op->SetSynapseInput(p_context_->syn_inputs_[1]);
    std::vector<c10::IValue> stack = {rois, c10::ScalarType::Float};
    auto md = OutputMetaDataVector(1);
    md[0].dtype = stack[1].toScalarType();
    cast_op->AllocateAndAddSynapseNode(graph, stack, md);
  }
  auto quad_tree_op = make_operator<habana::QuadTreeFwdImplOperator>(
      this->p_context_->device_id_, c10::ScalarType::Float);
  quad_tree_op->SetSynapseInput(
      (rois.scalar_type() == c10::ScalarType::BFloat16)
          ? cast_op->GetSynOutputs()[0]
          : p_context_->syn_inputs_[1]);
  quad_tree_op->SetSynapseInput(p_context_->syn_inputs_[2]);
  quad_tree_op->SetSynapseInput(p_context_->syn_inputs_[3]);
  quad_tree_op->AllocateAndAddSynapseNode(
      graph, inputs, OutputMetaDataVector(1));

  auto roi_bwd_op = make_operator<habana::RoiAlignBwdImplOperator>(
      this->p_context_->device_id_, inputs[0].toTensor().scalar_type());
  roi_bwd_op->SetSynapseInput(p_context_->syn_inputs_[0]);
  roi_bwd_op->SetSynapseInput(
      (rois.scalar_type() == c10::ScalarType::BFloat16)
          ? cast_op->GetSynOutputs()[0]
          : p_context_->syn_inputs_[1]);
  roi_bwd_op->SetSynapseInput(p_context_->syn_inputs_[2]);
  roi_bwd_op->SetSynapseInput(quad_tree_op->GetSynOutputs()[0]);
  roi_bwd_op->AllocateAndAddSynapseNode(graph, inputs, output_metadata);

  p_context_->syn_outputs_.emplace_back(
      std::move(roi_bwd_op->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(roi_bwd_op->GetOutputs()[0]));
}

InferOutputMetaRetType RoiAlignBwdImplOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  auto input_shape = inputs[3].toTensor();
  auto grad_out = inputs[0].toTensor();
  InferOutputMetaRetType out;
  out.AddOutputTensor(TensorMetaData(
      input_shape.sizes().vec(),
      HabanaOperator::CalculateStrides(
          input_shape.sizes().vec(), grad_out.suggest_memory_format()),
      grad_out.scalar_type(),
      grad_out.suggest_memory_format()));

  out.AddShapeTensor(TensorMetaData(
      input_shape.sizes().vec(),
      HabanaOperator::CalculateStrides(
          input_shape.sizes().vec(), grad_out.suggest_memory_format()),
      grad_out.scalar_type(),
      grad_out.suggest_memory_format()));
  return out;
}

void RoiAlignBwdImplOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 7,
      "Incorrect size of inputs expected for RoiAlignBwd operator");

  auto grad_out = inputs[0].toTensor();
  auto rois = inputs[1].toTensor();
  auto num_rois = inputs[2].toTensor();
  auto input_shape = inputs[3].toTensor();
  auto sampling_ratio = inputs[4].toInt();
  auto spatial_scale = inputs[5].toScalar().toFloat();
  auto aligned = inputs[6].toBool();

  ns_RoiAlignBwdKernel::ParamsIsValidCount roi_params{};
  roi_params.mode = RoiAlignMode_t::ROI_ALIGN_AVG;
  roi_params.sampling_ratio = sampling_ratio;
  roi_params.spatial_scale = spatial_scale;
  roi_params.aligned = aligned;
  roi_params.isValidCount = false;

  auto output = habana::createPTTensor(
      grad_out,
      input_shape.sizes(),
      grad_out.options(),
      output_metadata.at(0).persistent);

  // Restriction coming from TPC kernel. Adding this restriction serves 2
  // purpose, (1) if output_size (input_size for roi_align_fwd) for this kernel
  // exceeds the limit set by TPC, throw an assert in bridge instead of assert
  // in glue-code, (2) with Dynamic Shapes enabled, this ensures that we
  // fallback from max_policy = Caclulated to max_policy = Historic if required
  constexpr float segPerAxis = 16;
  constexpr float maxVlmCount = 320;
  TORCH_CHECK(
      (std::ceil(input_shape.sizes()[2] / segPerAxis) *
       std::ceil(input_shape.sizes()[3] / segPerAxis)) <= maxVlmCount,
      "VLM count exceeded in Roi_align_bwd, input image size too large to handle")

  // Allocate Shape Tensor
  if (graph.is_dynamic_graph()) {
    AllocateSynapseShapeTensor(graph, output);
  }

  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &roi_params, sizeof(roi_params));
}

InferOutputMetaRetType QuadTreeFwdImplOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  auto rois = inputs[1].toTensor();
  auto num_rois = inputs[2].toTensor();
  auto input_shape = inputs[3].toTensor();
  std::vector<int64_t> output_size = {
      input_shape.sizes()[0], 256, num_rois.sizes()[0] + 1};
  InferOutputMetaRetType out;
  out.AddOutputTensor(TensorMetaData(
      output_size,
      HabanaOperator::CalculateStrides(
          output_size, rois.suggest_memory_format()),
      c10::ScalarType::Short,
      rois.suggest_memory_format()));

  out.AddShapeTensor(TensorMetaData(
      output_size,
      HabanaOperator::CalculateStrides(
          output_size, rois.suggest_memory_format()),
      c10::ScalarType::Short,
      rois.suggest_memory_format()));
  return out;
}

void QuadTreeFwdImplOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  // auto grad_out = inputs[0].toTensor();
  auto rois = inputs[1].toTensor();
  auto num_rois = inputs[2].toTensor();
  auto input_shape = inputs[3].toTensor();
  // auto sampling_ratio = inputs[4].toInt();
  auto spatial_scale = inputs[5].toScalar().toFloat();
  // auto aligned = inputs[6].toBool();

  std::vector<int64_t> output_size = {
      input_shape.sizes()[0], 256, num_rois.sizes()[0] + 1};

  auto quadTree_output = habana::createPTTensor(
      rois,
      output_size,
      rois.options(),
      rois.suggest_memory_format(),
      c10::ScalarType::Short,
      output_metadata.at(0).persistent);

  ns_QuadTree::ParamsTorchVersion quad_tree_params;
  memset(&quad_tree_params, 0, sizeof(quad_tree_params));
  // all parameter settings as per recommendation in TPC docs
  quad_tree_params.segments = 256; // should be a power of 4
  quad_tree_params.isValidCount = false;
  quad_tree_params.enableAbsoluteCoords = true;
  quad_tree_params.levelScalarFactor = spatial_scale;
  quad_tree_params.enableTorchVersion = true;

  // Allocate Shape Tensor
  if (graph.is_dynamic_graph()) {
    AllocateSynapseShapeTensor(graph, quadTree_output);
  }

  AllocateSynapseOutput(graph, quadTree_output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &quad_tree_params, sizeof(quad_tree_params));
}

static auto& RoiAlignKernelsKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::roi_align_fwd", KERNEL_FN(RoiAlignFwdOperator))
        .add("hpu::roi_align_bwd", KERNEL_FN(RoiAlignBwdOperator));
