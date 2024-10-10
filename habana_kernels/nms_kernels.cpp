/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include <torch/script.h>

#include "backend/create_pt_tensor.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/kernel/hpu_shape_inference.h"
#include "backend/synapse_helpers/tensor_builder_base.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/nms_kernels.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_kernels/topk_kernels.h"

using namespace torch;
using namespace habana;
using tensor_name_generator = synapse_helpers::detail::tensor_name_generator;

void FilterAndSqueezeOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for filter&squeeze operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for filter&squeeze operator");
  TORCH_CHECK(
      inputs[1].isScalar(),
      "Input arg2 expected to be scalar for filter&squeeze operator");

  auto self = inputs[0].toTensor();
  auto threshold = inputs[1].toScalar();

  ns_FilterAndSqueeze::Params params{};
  params.threshold.f = threshold.toFloat();
  auto scores = habana::createPTTensor(
      self, self.sizes(), self.options(), output_metadata.at(0).persistent);
  auto box_ids = habana::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      output_metadata.at(1).persistent);
  auto valid_box_ids = habana::createPTTensor(
      self,
      {self.sizes()[0], self.sizes()[1]},
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      output_metadata.at(2).persistent);
  AllocateSynapseOutputs(
      graph, {scores, box_ids, valid_box_ids}, output_metadata);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void NMSOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 4, "Incorrect size of inputs expected for NMS operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for NMS operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for NMS operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input arg3 expected to be tensor for NMS operator");

  auto boxes = inputs[0].toTensor();
  auto box_ids = inputs[1].toTensor();
  auto valid_box_ids = inputs[2].toTensor();
  auto iou = inputs[3].toScalar();

  ns_Nms::Params params{iou.toFloat()};
  auto box_id_out = habana::createPTTensor(
      box_ids,
      box_ids.sizes(),
      box_ids.options(),
      output_metadata.at(0).persistent);
  AllocateSynapseOutput(graph, box_id_out, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void PostNmsOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for PostNms operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for PostNms operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for PostNms operator");

  auto box_ids = inputs[0].toTensor();
  auto valid_box_ids = inputs[1].toTensor();

  auto box_id_out = habana::createPTTensor(
      box_ids,
      {static_cast<int>(box_ids.sizes()[2])},
      box_ids.options(),
      output_metadata.at(0).persistent);
  auto valid_box_id_out = habana::createPTTensor(
      valid_box_ids,
      {1},
      valid_box_ids.options(),
      output_metadata.at(1).persistent);
  AllocateSynapseOutput(
      graph,
      box_id_out,
      output_metadata.at(0),
      false); // is_shape_tensor

  // For dynamic case the max_output_size in params is equal to
  // max value of output size
  ns_PostNms::Params params;
  if (graph.is_dynamic_graph() && (!graph.is_dry_run())) {
    synapse_helpers::tensor& syn_tensor = p_context_->syn_outputs_.back();
    auto tensor_id = syn_tensor.id();
    std::vector<int64_t> min, max;
    std::tie(min, max) = habana::ShapeInference::GetMinMaxShape(tensor_id);
    params.max_output_size = static_cast<int>(max[0]);
  } else {
    params.max_output_size = static_cast<int>(box_ids.sizes()[2]);
  }

  AllocateSynapseOutput(
      graph,
      valid_box_id_out,
      output_metadata.at(1),
      false); // is_shape_tensor

  auto shape_tensor = habana::createPTTensor(
      valid_box_ids,
      {5},
      valid_box_ids.options(),
      output_metadata.at(2).persistent);
  synDataType synType = syn_type_uint32;
  AllocateSynapseOutput(
      graph,
      shape_tensor,
      synType,
      output_metadata.at(2),
      graph.is_dynamic_graph() ? true : false);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void HabanaNMSOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto scores = inputs[1].toTensor();
  auto box_id_out = habana::createPTTensor(
      scores,
      {scores.sizes()[0]},
      scores.options().dtype(c10::ScalarType::Int),
      true);
  auto valid_box_id_out = habana::createPTTensor(
      scores, {1}, scores.options().dtype(c10::ScalarType::Int), true);
  auto shape_tensor = habana::createPTTensor(
      scores, {5}, scores.options().dtype(c10::ScalarType::Int), true);

  std::vector<at::Tensor> outputs{box_id_out, valid_box_id_out, shape_tensor};
  HabanaOperator::SetPTOutputs(outputs);
}

void HabanaNMSOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of inputs expected for HabanaNms operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for HabanaNms operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg1 expected to be tensor for HabanaNms operator");
  TORCH_CHECK(
      inputs[2].isScalar(),
      "Input arg2 expected to be scalar for HabanaNms operator");
  TORCH_CHECK(
      inputs[3].isScalar(),
      "Input arg2 expected to be scalar for HabanaNms operator");

  auto boxes = inputs[0].toTensor();
  auto scores = inputs[1].toTensor();
  auto iou = inputs[2].toScalar();
  auto threshold = inputs[3].toScalar();

  // Reshape from [Scores] -> [N, Classes, Scores] where N = Classes = 1.
  // Needed because Filter & Squeeze works only on this 3D input tensor.
  auto reshape_op1 = make_operator<ReshapeOperator>(
      scores.device().index(), scores.scalar_type());
  reshape_op1->SetSynapseInput(p_context_->syn_inputs_[1]);
  auto shape1 = scores.sizes().vec();
  shape1.insert(shape1.cbegin(), 1);
  shape1.insert(shape1.cbegin(), 1);
  torch::jit::Stack stack = {IValue(scores), IValue(shape1)};
  reshape_op1->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();

  // Input (Scores): [N, Classes, kBox]
  // Output (Filtered Scores) : [N, Classes, kBox]
  // Output (Filtered BoxIds) : [N, Classes, kBox]
  // Output (Valid Box Count) : [N, Classes]
  auto filter_op = make_operator<FilterAndSqueezeOperator>(
      scores.device().index(), "filter_and_squeeze_fwd_f32");
  filter_op->SetSynapseInput(reshape_op1->GetSynOutputs()[0]);
  stack = {IValue(reshape_op1->GetOutputs()[0]), IValue(threshold)};
  filter_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(3));
  stack.clear();

  // Input (Scores): [kBox]
  // Output (Sorted Scores): [kBox]
  // Output (Sorted BoxIds): [kBox]
  auto sort_op = make_operator<TopkOperator>(scores.device().index(), "topk");
  sort_op->SetSynapseInput(p_context_->syn_inputs_[1]);
  stack = {
      IValue(scores),
      IValue(scores.sizes()[0]),
      IValue(scores.dim() - 1),
      IValue(true),
      IValue(true)};
  sort_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(2));
  stack.clear();

  // Input (Boxes): [kBox, 4]
  // Input (Sorted Scores): [kBox]
  // Output (Gathered Boxes): [kBox, 4]
  auto gather_op = make_operator<GatherOperator>(
      scores.device().index(), scores.scalar_type());
  gather_op->SetSynapseInput(p_context_->syn_inputs_[0]);
  auto& syn11 = gather_op->SetSynapseInput(sort_op->GetSynOutputs()[1]);
  stack = {
      IValue(boxes),
      IValue(0),
      IValue(sort_op->GetOutputs()[1]),
      IValue(false)};
  gather_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();

  // Reshape sorted scores from [kBox] -> [N, Classes, kBox], where N = Classes
  // = 1
  auto reshape_op3 = make_operator<ReshapeOperator>(
      scores.device().index(), scores.scalar_type());
  reshape_op3->SetSynapseInput(syn11);
  auto shape3 = sort_op->GetOutputs()[1].sizes().vec();
  shape3.insert(shape3.cbegin(), 1);
  shape3.insert(shape3.cbegin(), 1);
  stack = {IValue(sort_op->GetOutputs()[1]), IValue(shape3)};
  reshape_op3->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();

  // Reshape gathered boxes from [kBox, 4] -> [4, kBox]
  auto t2_op = make_operator<TransposeOperator>(
      scores.device().index(), scores.scalar_type());
  stack = {IValue(gather_op->GetOutputs()[0]), IValue(0), IValue(1)};
  t2_op->SetSynapseInput(gather_op->GetSynOutputs()[0]);
  t2_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();

  // Reshape gathered boxes from [4, kBox] -> [N, 4, Classes, kBox], where N =
  // Classes = 1
  auto reshape_op4 = make_operator<ReshapeOperator>(
      scores.device().index(), scores.scalar_type());
  reshape_op4->SetSynapseInput(t2_op->GetSynOutputs()[0]);
  auto shape4 = t2_op->GetOutputs()[0].sizes().vec();
  shape4.insert(shape4.cbegin() + 1, 1);
  shape4.insert(shape4.cbegin(), 1);
  stack = {IValue(t2_op->GetOutputs()[0]), IValue(shape4)};
  reshape_op4->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();

  // Input (Gathered boxes): [N, 4, Classes, KBox]
  // Input (Sorted box-id): [N, Classes, kBox]
  // Input (valid box count): [N, Classes]
  // Output (Box-id out): [N, Classes, kBox]
  auto nms_op =
      make_operator<NMSOperator>(scores.device().index(), "nms_fwd_f32");
  nms_op->SetSynapseInput(reshape_op4->GetSynOutputs()[0]);
  nms_op->SetSynapseInput(reshape_op3->GetSynOutputs()[0]);
  auto& syn_nms2 = nms_op->SetSynapseInput(filter_op->GetSynOutputs()[2]);
  stack = {
      IValue(reshape_op4->GetOutputs()[0]),
      IValue(reshape_op3->GetOutputs()[0]),
      IValue(filter_op->GetOutputs()[2]),
      IValue(iou)};
  nms_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();

  auto postnms_op = make_operator<PostNmsOperator>(
      scores.device().index(), "post_nms_fwd_i32");
  postnms_op->SetSynapseInput(nms_op->GetSynOutputs()[0]);
  postnms_op->SetSynapseInput(syn_nms2);
  stack = {IValue(nms_op->GetOutputs()[0]), IValue(filter_op->GetOutputs()[2])};
  postnms_op->AllocateAndAddSynapseNode(graph, stack, output_metadata);
  stack.clear();

  p_context_->syn_outputs_.emplace_back(
      std::move(postnms_op->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(postnms_op->GetOutputs()[0]));
  p_context_->syn_outputs_.emplace_back(
      std::move(postnms_op->GetSynOutputs()[1]));
  p_context_->pt_outputs_.emplace_back(std::move(postnms_op->GetOutputs()[1]));
  p_context_->syn_outputs_.emplace_back(
      std::move(postnms_op->GetSynOutputs()[2]));
  p_context_->pt_outputs_.emplace_back(std::move(postnms_op->GetOutputs()[2]));
}

InferOutputMetaRetType BatchedNMSOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  InferOutputMetaRetType out;

  auto boxes = inputs[0].toTensor();
  auto scores = inputs[1].toTensor();
  auto indexes = inputs[2].toTensor();
  auto max_classes = inputs[6].toScalar().toInt();
  // Add a check for validating shape tensor 2 sizes which can come wrong
  // in case if LOCAL_HISTORIC min is used. Explained below:
  // ITR1: Scores 1; shape_tensor 2 = 1*81 ; MIN {Scores = 1; shape tensor 2 =
  // 81
  // ITR2: Scores 3; shape_tensor 3 = 3*81= 243 ; MIN {Scores = 3; shape
  // tensor 2 = 81.
  // Here above in itr 2 score is 3 since 1 is considered
  // broadcast and rejected but since shape tensor 2 is 81 its not rejected,
  // although calculation is wrong.
  auto shape_tensor_2_size = inputs[5].toTensor().sizes()[0];
  TORCH_CHECK(
      (scores.sizes()[0] * max_classes) == shape_tensor_2_size,
      "Shape tensor 2 calculation mismatch for batched_nms");

  out.AddOutputTensor(TensorMetaData(
      {static_cast<int>(indexes.sizes()[0]) * max_classes},
      HabanaOperator::CalculateStrides(
          {static_cast<int>(indexes.sizes()[0]) * max_classes},
          indexes.suggest_memory_format()),
      indexes.scalar_type(),
      indexes.suggest_memory_format()));

  out.AddOutputTensor(TensorMetaData(
      {5},
      HabanaOperator::CalculateStrides({5}, indexes.suggest_memory_format()),
      indexes.scalar_type(),
      indexes.suggest_memory_format()));

  return out;
}

void BatchedNMSOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 7,
      "Incorrect size of inputs expected for HabanaBatchedNms operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for HabanaBatchedNms operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for HabanaBatchedNms operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input arg3 expected to be tensor for HabanaBatchedNms operator");
  TORCH_CHECK(
      inputs[3].isScalar(),
      "Input arg4 expected to be scalar for HabanaBatchedNms operator");

  auto boxes = inputs[0].toTensor();
  auto scores = inputs[1].toTensor();
  auto indexes = inputs[2].toTensor();
  auto iou = inputs[3].toScalar();
  auto max_classes = inputs[6].toScalar().toInt();
  // Add a check for validating shape tensor 2 sizes which can come wrong
  // in case if LOCAL_HISTORIC min is used. Explained below:
  // ITR1: Scores 1; shape_tensor 2 = 1*81 ; MIN {Scores = 1; shape tensor 2 =
  // 81
  // ITR2: Scores 3; shape_tensor 3 = 3*81= 243 ; MIN {Scores = 3; shape
  // tensor 2 = 81.
  // Here above in itr 2 score is 3 since 1 is considered
  // broadcast and rejected but since shape tensor 2 is 81 its not rejected,
  // although calculation is wrong.
  auto shape_tensor_2_size = inputs[5].toTensor().sizes()[0];
  TORCH_CHECK(
      (scores.sizes()[0] * max_classes) == shape_tensor_2_size,
      "Shape tensor 2 calculation mismatch for batched_nms");
  TORCH_CHECK(
      scores.sizes()[0] == indexes.sizes()[0],
      "Number of categories and scores missmatch for batched_nms");

  auto box_id_out = habana::createPTTensor(
      indexes,
      {static_cast<int>(indexes.sizes()[0]) * max_classes},
      indexes.options(),
      output_metadata.at(0).persistent);
  AllocateSynapseOutput(
      graph,
      box_id_out,
      output_metadata.at(0),
      false); // is_shape_tensor

  auto shape_tensor = habana::createPTTensor(
      indexes, {5}, indexes.options(), output_metadata.at(1).persistent);
  AllocateSynapseOutput(graph, shape_tensor, output_metadata.at(1), false);

  ns_BatchedNmsKernel::Params params{};
  params.nms_threshold = iou.toFloat();
  params.max_num_classes = max_classes;
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

static auto& NMSKernelsKernelRegistry =
    habana::KernelRegistry()
        .add(
            "hpu::habana_nms",
            [](int device_id, c10::ScalarType scalar_type) {
              std::string node_type =
                  get_guid_with_precision("habana_nms", scalar_type);
              return std::make_shared<HabanaNMSOperator>(device_id, node_type);
            })
        .add(
            "hpu::batched_nms",
            [](int device_id, c10::ScalarType scalar_type) {
              std::string node_type =
                  get_guid_with_precision("batched_nms_fwd", scalar_type);
              return std::make_shared<BatchedNMSOperator>(device_id, node_type);
            });
