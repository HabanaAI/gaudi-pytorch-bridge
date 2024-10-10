/*******************************************************************************
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
#include "repeat.h"
#include <perf_lib_layer_params.h>
#include <torch/script.h>
#include "backend/create_pt_tensor.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/dynamic_shape_info.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/kernel/hpu_shape_inference.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/tensor_impl.h"

using namespace torch;
using namespace habana;

std::vector<int64_t> RepeatOperator::compute_output_shape(
    const at::Tensor& self,
    at::IntArrayRef repeats) {
  int64_t num_new_dimensions = repeats.size() - self.dim();
  std::vector<int64_t> padded_size(num_new_dimensions, 1);
  padded_size.insert(
      padded_size.end(), self.sizes().begin(), self.sizes().end());
  std::vector<int64_t> outshape(repeats.size());
  for (size_t i = 0; i < repeats.size(); ++i) {
    outshape[i] = padded_size[i] * repeats[i];
  }
  return outshape;
}

std::vector<int64_t> RepeatOperator::compute_reshape_output(
    const at::Tensor& self,
    at::IntArrayRef repeats) {
  int64_t num_new_dimensions = repeats.size() - self.dim();
  std::vector<int64_t> padded_size(num_new_dimensions, 1);
  padded_size.insert(
      padded_size.end(), self.sizes().begin(), self.sizes().end());
  return padded_size;
}

std::vector<int64_t> RepeatOperatorHT::ComputeRepeatShapefromH2DTensor(
    const at::Tensor& host_tensor) {
  auto tmeta{get_tensor_extra_meta(host_tensor)};

  bool is_dry_run = false;
  if (habana::ShapeInference::GetCurrentPass() ==
          habana::ShapeInfo::InferencePass::MIN_SHAPE ||
      habana::ShapeInference::GetCurrentPass() ==
          habana::ShapeInfo::InferencePass::MAX_SHAPE) {
    is_dry_run = true;
  }

  void* host_ptr = nullptr;
  if (is_dry_run) {
    host_ptr = tmeta->get_compile_host_ptr();
  } else {
    host_ptr = tmeta->get_host_ptr();
  }

  size_t h2d_data_size = tmeta->get_host_size();
  if (habana::ShapeInference::GetCurrentPass() ==
      habana::ShapeInfo::InferencePass::MIN_SHAPE) {
    size_t data_size = h2d_data_size * tmeta->get_host_el_size();
    host_ptr = static_cast<char*>(host_ptr) + data_size;
  }

  std::vector<int64_t> repeat;
  uint32_t* h2d_data = static_cast<uint32_t*>(host_ptr);
  for (size_t i = 0; i < h2d_data_size; i++) {
    repeat.push_back(*h2d_data++);
  }

  return repeat;
}

InferOutputMetaRetType RepeatOperatorHT::InferOutputMeta(
    torch::jit::Stack& inputs) {
  InferOutputMetaRetType out;
  auto input = inputs[0].toTensor();
  auto param_tensor = inputs[1].toTensor();

  auto repeat_shape = ComputeRepeatShapefromH2DTensor(param_tensor);
  int64_t size = static_cast<int64_t>(repeat_shape.size());

  std::vector<int64_t> rpt_cast;
  for_each(repeat_shape.rbegin(), repeat_shape.rend(), [&](const int32_t& n) {
    rpt_cast.push_back(static_cast<int64_t>(n));
  });
  auto out_size = RepeatOperator::compute_output_shape(input, rpt_cast);

  if (size > input.ndimension()) {
    auto reshapeSize = RepeatOperator::compute_reshape_output(
        input, IntArrayRef(rpt_cast.data(), rpt_cast.size()));
    auto reshapeOp = make_operator<ReshapeOperator>(
        this->p_context_->device_id_, input.scalar_type());
    torch::jit::Stack temp_stack = {IValue(input), IValue(reshapeSize)};
    out.call_InferOutputMeta(reshapeOp, temp_stack);
  }

  auto out_metadata = TensorMetaData(
      out_size,
      HabanaOperator::CalculateStrides(out_size, input.suggest_memory_format()),
      input.scalar_type(),
      input.suggest_memory_format());
  out.AddOutputTensor(out_metadata);
  return out;
}

void RepeatOperatorHT::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for RepeatHTOperator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 & arg3 expected to be shape tensor for RepeatHTOperator");
  auto input = inputs[0].toTensor();
  auto param_tensor = inputs[1].toTensor();
  auto repeat_shape = ComputeRepeatShapefromH2DTensor(param_tensor);
  int64_t size = static_cast<int64_t>(repeat_shape.size());

  std::vector<int64_t> rpt_cast;
  for_each(repeat_shape.rbegin(), repeat_shape.rend(), [&](const int32_t& n) {
    rpt_cast.push_back(static_cast<int64_t>(n));
  });

  std::vector<int32_t> repeats;
  for (auto t : repeat_shape) {
    repeats.push_back(static_cast<int32_t>(t));
  }

  if (size > input.ndimension()) {
    auto reshapeSize = RepeatOperator::compute_reshape_output(
        input, IntArrayRef(rpt_cast.data(), rpt_cast.size()));
    auto reshapeOp = make_operator<ReshapeOperator>(
        this->p_context_->device_id_, input.scalar_type());
    torch::jit::Stack temp_stack = {IValue(input), IValue(reshapeSize)};
    reshapeOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    reshapeOp->AllocateAndAddSynapseNode(
        graph, temp_stack, OutputMetaDataVector(1));
    synapse_helpers::tensor& syn_tensor = reshapeOp->GetSynOutputs()[0];
    p_context_->syn_inputs_[0] = std::move(syn_tensor);
  }

  if (!graph.is_dry_run() &&
      output_metadata.at(0).allocated_tensor.has_value()) {
    AllocateSynapseOutput(
        graph,
        output_metadata.at(0).allocated_tensor.value(),
        output_metadata.at(0));
  } else {
    auto output = habana::createPTTensor(
        input,
        RepeatOperator::compute_output_shape(input, rpt_cast),
        input.options(),
        output_metadata.at(0).persistent);

    AllocateSynapseOutput(graph, output, output_metadata.at(0));
  }
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

void RepeatOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for repeat operator");
  TORCH_CHECK(
      inputs[1].isIntList(),
      "Input arg2 expected to be intlist for repeat operator");
  auto input = inputs[0].toTensor();
  auto repeats = inputs[1].toIntVector();
  int64_t size = repeats.size();

  if (size > input.ndimension()) {
    torch::jit::Stack temp_stack;
    auto reshapeSize = RepeatOperator::compute_reshape_output(input, repeats);
    auto reshapeOp = make_operator<ReshapeOperator>(
        this->p_context_->device_id_, input.scalar_type());
    temp_stack = {IValue(input), IValue(reshapeSize)};
    reshapeOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    reshapeOp->AllocateAndAddSynapseNode(
        graph, temp_stack, OutputMetaDataVector(1));
    synapse_helpers::tensor& syn_tensor = reshapeOp->GetSynOutputs()[0];
    p_context_->syn_input_orig_.emplace_back(
        std::move(p_context_->syn_inputs_[0]));
    p_context_->syn_inputs_[0] = std::move(syn_tensor);
  }
  ns_TileKernel::ParamsV2 params{};

  if (!graph.is_dry_run() &&
      output_metadata.at(0).allocated_tensor.has_value()) {
    AllocateSynapseOutput(
        graph,
        output_metadata.at(0).allocated_tensor.value(),
        output_metadata.at(0));
  } else {
    auto output = habana::createPTTensor(
        input,
        RepeatOperator::compute_output_shape(input, repeats),
        input.options(),
        output_metadata.at(0).persistent);
    AllocateSynapseOutput(graph, output, output_metadata.at(0));
  }

  for (int64_t i = 0; i < size; ++i) {
    params.repeat[size - i - 1] = repeats[i];
  }

  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

std::vector<int64_t> RepeatInlvOperator::compute_output_shape(
    const at::Tensor& input,
    int64_t dim,
    int64_t out_size) {
  auto outshape = input.sizes().vec();
  outshape[dim] = out_size;
  return outshape;
}

InferOutputMetaRetType RepeatInlvOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  InferOutputMetaRetType out;
  auto input = inputs[0].toTensor();
  auto out_shape = inputs[3].toTensor();
  auto out_metadata = TensorMetaData(
      out_shape.sizes().vec(),
      HabanaOperator::CalculateStrides(
          out_shape.sizes().vec(), input.suggest_memory_format()),
      input.scalar_type(),
      input.suggest_memory_format());
  out.AddOutputTensor(out_metadata);
  return out;
}

void RepeatInlvOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for repeat-interleave operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for repeat-interleave operator");
  TORCH_CHECK(
      inputs[2].isInt(),
      "Input arg3 expected to be Int for repeat-interleave operator");
  TORCH_CHECK(
      inputs[3].isTensor(),
      "Input arg4 expected to be tensor for repeat-interleave operator");

  auto input = inputs[0].toTensor();
  auto dim = inputs[2].toInt();
  auto out_shape = inputs[3].toTensor();

  ns_RepeatKernelGaudiTF::Params params;
  params.axis = input.dim() - 1 - dim;
  auto output = habana::createPTTensor(
      input,
      out_shape.sizes(),
      input.options(),
      output_metadata.at(0).persistent);
  // throw away shape tensor before adding synapse node
  p_context_->syn_inputs_.pop_back();
  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

std::vector<int64_t> RepeatInlvOperatorHT::ComputeRepeatShapefromH2DTensor(
    const at::Tensor& host_tensor) {
  auto tmeta{get_tensor_extra_meta(host_tensor)};

  bool is_dry_run = false;
  if (habana::ShapeInference::GetCurrentPass() ==
          habana::ShapeInfo::InferencePass::MIN_SHAPE ||
      habana::ShapeInference::GetCurrentPass() ==
          habana::ShapeInfo::InferencePass::MAX_SHAPE) {
    is_dry_run = true;
  }

  void* host_ptr = nullptr;
  if (is_dry_run) {
    host_ptr = tmeta->get_compile_host_ptr();
  } else {
    host_ptr = tmeta->get_host_ptr();
  }

  size_t h2d_data_size = tmeta->get_host_size();
  if (habana::ShapeInference::GetCurrentPass() ==
      habana::ShapeInfo::InferencePass::MIN_SHAPE) {
    size_t data_size = h2d_data_size * tmeta->get_host_el_size();
    host_ptr = static_cast<char*>(host_ptr) + data_size;
  }

  std::vector<int64_t> repeat;
  uint32_t* h2d_data = static_cast<uint32_t*>(host_ptr);
  for (size_t i = 0; i < h2d_data_size; i++) {
    repeat.push_back(*h2d_data++);
  }

  return repeat;
}

InferOutputMetaRetType RepeatInlvOperatorHT::InferOutputMeta(
    torch::jit::Stack& inputs) {
  InferOutputMetaRetType out;
  auto input = inputs[0].toTensor();
  auto repeats_ht = inputs[1].toTensor();

  auto repeat_vec = ComputeRepeatShapefromH2DTensor(repeats_ht);
  auto out_size = std::accumulate(repeat_vec.begin(), repeat_vec.end(), 0ll);
  auto out_shape = RepeatInlvOperator::compute_output_shape(input, 0, out_size);

  auto out_metadata = TensorMetaData(
      out_shape,
      HabanaOperator::CalculateStrides(
          out_shape, input.suggest_memory_format()),
      input.scalar_type(),
      input.suggest_memory_format());
  out.AddOutputTensor(out_metadata);
  return out;
}

void RepeatInlvOperatorHT::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for repeat-interleave operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for repeat-interleave operator");
  TORCH_CHECK(
      inputs[2].isInt(),
      "Input arg3 expected to be Int for repeat-interleave operator");

  auto input = inputs[0].toTensor();
  auto dim = inputs[2].toInt();

  auto repeats_ht = inputs[1].toTensor();
  TORCH_CHECK(p_context_->syn_inputs_[1].ref().is_host_to_device_tensor());
  auto tmeta{get_tensor_extra_meta(repeats_ht)};

  TORCH_CHECK(
      tmeta->get_host_dt_type() == habana::HostDataType::INT32_T,
      "Incorrect datatype of HOST ",
      tmeta->get_host_dt_type(),
      ", expecting ",
      habana::HostDataType::INT32_T);

  auto repeat_vec = ComputeRepeatShapefromH2DTensor(repeats_ht);
  auto out_size = std::accumulate(repeat_vec.begin(), repeat_vec.end(), 0ll);

  auto out_shape = RepeatInlvOperator::compute_output_shape(input, 0, out_size);

  ns_RepeatKernelGaudiTF::Params params;
  params.axis = input.dim() - 1 - dim;
  auto output = habana::createPTTensor(
      input, out_shape, input.options(), output_metadata.at(0).persistent);
  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

static auto& RepeatKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::repeat_inlv", KERNEL_FN(RepeatInlvOperator))
        .add("hpu::repeat_inlv_ht", KERNEL_FN(RepeatInlvOperatorHT))
        .add("hpu::repeat_ht", KERNEL_FN(RepeatOperatorHT));
