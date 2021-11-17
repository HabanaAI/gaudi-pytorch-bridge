/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <perf_lib_layer_params.h>
#include <torch/script.h>

#include "habana_bridge/kernel/hpu_shape_inference.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/nms_kernels.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_kernels/topk_kernels.h"
#include "synapse_helpers/tensor_builder_base.h"

using namespace torch;
using namespace habana;
using tensor_name_generator = synapse_helpers::detail::tensor_name_generator;

void FilterAndSqueezeOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
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
  auto scores = habana_helpers::createPTTensor(
      self, self.sizes(), self.options(), is_output_persistent);
  auto box_ids = habana_helpers::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      is_output_persistent);
  auto valid_box_ids = habana_helpers::createPTTensor(
      self,
      {self.sizes()[0], self.sizes()[1]},
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      is_output_persistent);
  AllocateSynapseOutputs(
      graph,
      {scores, box_ids, valid_box_ids},
      {is_output_persistent, is_output_persistent, is_output_persistent},
      {true, true, true});
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void NMSOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
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
  auto box_id_out = habana_helpers::createPTTensor(
      box_ids, box_ids.sizes(), box_ids.options(), is_output_persistent);
  AllocateSynapseOutput(graph, box_id_out, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void PostNmsOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
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

  auto box_id_out = habana_helpers::createPTTensor(
      box_ids,
      {static_cast<int>(box_ids.sizes()[2])},
      box_ids.options(),
      is_output_persistent[0]);
  auto valid_box_id_out = habana_helpers::createPTTensor(
      valid_box_ids, {1}, valid_box_ids.options(), is_output_persistent[1]);
  AllocateSynapseOutput(
      graph,
      box_id_out,
      is_output_persistent[0],
      false, // is_shape_tensor
      true); // use_metadata

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
      is_output_persistent[1],
      false, // is_shape_tensor
      true); // use_metadata

  auto shape_tensor = habana_helpers::createPTTensor(
      valid_box_ids, {5}, valid_box_ids.options(), is_output_persistent[2]);
  synDataType synType = syn_type_uint32;
  AllocateSynapseOutput(
      graph,
      shape_tensor,
      synType,
      is_output_persistent[2],
      graph.is_dynamic_graph() ? true : false);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void HabanaNMSOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto scores = inputs[1].toTensor();
  auto box_id_out = habana_helpers::createPTTensor(
      scores,
      {scores.sizes()[0]},
      scores.options().dtype(c10::ScalarType::Int),
      true);
  auto valid_box_id_out = habana_helpers::createPTTensor(
      scores, {1}, scores.options().dtype(c10::ScalarType::Int), true);
  auto shape_tensor = habana_helpers::createPTTensor(
      scores, {5}, scores.options().dtype(c10::ScalarType::Int), true);

  std::vector<at::Tensor> outputs{box_id_out, valid_box_id_out, shape_tensor};
  HabanaOperator::SetPTOutputs(outputs);
}

void HabanaNMSOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
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
  reshape_op1->AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  // Input (Scores): [N, Classes, kBox]
  // Output (Filtered Scores) : [N, Classes, kBox]
  // Output (Filtered BoxIds) : [N, Classes, kBox]
  // Output (Valid Box Count) : [N, Classes]
  auto filter_op = make_operator<FilterAndSqueezeOperator>(
      scores.device().index(), "filter_and_squeeze_fwd_f32");
  filter_op->SetSynapseInput(reshape_op1->GetSynOutputs()[0]);
  stack = {IValue(reshape_op1->GetOutputs()[0]), IValue(threshold)};
  filter_op->AllocateAndAddSynapseNode(graph, stack, false);
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
  sort_op->AllocateAndAddSynapseNode(graph, stack, {false, false});
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
  gather_op->AllocateAndAddSynapseNode(graph, stack, false);
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
  reshape_op3->AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  // Reshape gathered boxes from [kBox, 4] -> [4, kBox]
  auto t2_op = make_operator<TransposeOperator>(
      scores.device().index(), scores.scalar_type());
  stack = {IValue(gather_op->GetOutputs()[0]), IValue(0), IValue(1)};
  t2_op->SetSynapseInput(gather_op->GetSynOutputs()[0]);
  t2_op->AllocateAndAddSynapseNode(graph, stack, false);
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
  reshape_op4->AllocateAndAddSynapseNode(graph, stack, false);
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
  nms_op->AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  auto postnms_op = make_operator<PostNmsOperator>(
      scores.device().index(), "post_nms_fwd_i32");
  postnms_op->SetSynapseInput(nms_op->GetSynOutputs()[0]);
  postnms_op->SetSynapseInput(syn_nms2);
  postnms_op->SetOutputMetadata(output_metadata_);
  stack = {IValue(nms_op->GetOutputs()[0]), IValue(filter_op->GetOutputs()[2])};
  postnms_op->AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
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

at::Tensor habana_nms_hpu(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    float iou_threshold,
    float score_threshold) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = scores.scalar_type();
  std::string node_type =
      "habana_nms_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = scores.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::vector<at::Tensor> pt_inputs{boxes, scores};
  std::vector<c10::IValue> stack = {
      IValue(boxes),
      IValue(scores),
      IValue(Scalar(iou_threshold)),
      IValue(Scalar(score_threshold))};
  // Create the operator
  HabanaNMSOperator Op(device_id, node_type);
  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, {true, true, true});
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 3, "Incorrect size of outputs");

  // Extract correct output using shape information.
  auto output = out.at(0).slice(0l, 0l, out.at(1).item().toLong(), 1l);
  PT_KERNEL_END;
  return output;
}

static auto& KernelRegistry = habana::KernelRegistry().add(
    "hpu::habana_nms",
    [](int device_id, c10::ScalarType scalar_type) {
      std::string node_type =
          "habana_nms_" + habana_helpers::name_suffix_from_type(scalar_type);
      return std::make_shared<HabanaNMSOperator>(device_id, node_type);
    });
