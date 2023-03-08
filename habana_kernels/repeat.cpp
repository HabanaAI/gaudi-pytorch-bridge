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
#include "backend/helpers/tensor_utils.h"
#include "backend/kernel/hpu_shape_inference.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/dynamic_shape_info.h"
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
  auto impl = habana_lazy::GetHbInternalTensorImpl(host_tensor);

  bool is_dry_run = false;
  if (habana::ShapeInference::GetCurrentPass() ==
          habana::ShapeInfo::InferencePass::MIN_SHAPE ||
      habana::ShapeInference::GetCurrentPass() ==
          habana::ShapeInfo::InferencePass::MAX_SHAPE) {
    is_dry_run = true;
  }

  void* host_ptr = nullptr;
  if (is_dry_run) {
    host_ptr = impl->get_compile_host_ptr();
  } else {
    host_ptr = impl->get_host_ptr();
  }

  size_t h2d_data_size = impl->get_host_size();
  if (habana::ShapeInference::GetCurrentPass() ==
      habana::ShapeInfo::InferencePass::MIN_SHAPE) {
    size_t data_size = h2d_data_size * impl->get_host_el_size();
    host_ptr = static_cast<char*>(host_ptr) + data_size;
  }

  std::vector<int64_t> repeat;
  uint32_t* h2d_data = static_cast<uint32_t*>(host_ptr);
  for (size_t i = 0; i < h2d_data_size; i++) {
    repeat.push_back(*h2d_data++);
  }

  return repeat;
}

OutputShapeInfRetType RepeatOperatorHT::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  OutputShapeInfRetType out;
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
    out.call_ComputeOutputShape(reshapeOp, temp_stack);
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

  auto impl = habana_lazy::GetHbInternalTensorImpl(param_tensor);
  HABANA_ASSERT(impl);

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

  auto output = habana::createPTTensor(
      input,
      RepeatOperator::compute_output_shape(input, rpt_cast),
      input.options(),
      output_metadata.at(0).persistent);

  AllocateSynapseOutput(graph, output, output_metadata.at(0));
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
      inputs[1].isIntList() || inputs[1].isTensor(),
      "Input arg2 expected to be intlist or tenspr shape for repeat operator");
  auto input = inputs[0].toTensor();
  auto repeats = inputs[1].isIntList() ? inputs[1].toIntVector()
                                       : inputs[1].toTensor().sizes().vec();
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

  auto output = habana::createPTTensor(
      input,
      RepeatOperator::compute_output_shape(input, repeats),
      input.options(),
      output_metadata.at(0).persistent);

  if (inputs[1].isIntList()) {
    for (int64_t i = 0; i < size; ++i) {
      params.repeat[size - i - 1] = repeats[i];
    }

    // Allocate Shape Tensor
    if (graph.is_dynamic_graph()) {
      auto repeatsShape =
          habana::createPTTensor(input, repeats, input.options(), false);
      AllocateSynapseShapeTensor(
          graph, repeatsShape, INPUT_DESCRIBING_SHAPE_TENSOR);
    }
  } else {
    TORCH_CHECK(p_context_->syn_inputs_.back().ref().is_input_shape_tensor());
  }

  AllocateSynapseOutput(graph, output, output_metadata.at(0));
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

OutputShapeInfRetType RepeatInlvOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  OutputShapeInfRetType out;
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

  if (habana_helpers::GetRefineDynamicShapeStatus()) {
    auto repeats_ht = inputs[1].toTensor();
    TORCH_CHECK(p_context_->syn_inputs_[1].ref().is_host_to_device_tensor());
    auto impl = habana_lazy::GetHbInternalTensorImpl(repeats_ht);
    HABANA_ASSERT(impl);

    TORCH_CHECK(
        impl->get_host_dt_type() == habana_lazy::HostDataType::INT32_T,
        "Incorrect datatype of HOST ",
        impl->get_host_dt_type(),
        ", expecting ",
        habana_lazy::HostDataType::INT32_T);

    // set min/max for repeats_ht, this min/max is used only for memory
    // allocations by synapse (not for actual compilation), therefore we can set
    // only last element of ht to out_shape (rest of elements set to 0). Recall
    // that out_shape is computed by summing elements in ht.
    if (habana::ShapeInference::GetCurrentPass() ==
        habana::ShapeInfo::InferencePass::MIN_SHAPE) {
      auto size_tensor = input.sizes().vec()[dim];
      std::vector<int32_t> d(size_tensor, 0);
      d[size_tensor - 1] = out_shape.sizes()[dim];
      impl->set_min<int32_t>(d);
    } else if (
        habana::ShapeInference::GetCurrentPass() ==
        habana::ShapeInfo::InferencePass::MAX_SHAPE) {
      auto size_tensor = input.sizes().vec()[dim];
      std::vector<int32_t> d(size_tensor, 0);
      d[size_tensor - 1] = out_shape.sizes()[dim];
      impl->set_max<int32_t>(d);
    }
  }

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

static auto& RepeatKernelRegistry =
    habana::KernelRegistry()
        .add("aten::repeat", KERNEL_FN(RepeatOperator))
        .add("hpu::repeat", KERNEL_FN(RepeatOperator))
        .add("hpu::repeat_inlv", KERNEL_FN(RepeatInlvOperator))
        .add("hpu::repeat_ht", KERNEL_FN(RepeatOperatorHT));
