/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "repeat.h"
#include <perf_lib_layer_params.h>
#include <torch/script.h>
#include "habana_bridge/kernel/hpu_shape_inference.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/simple_generic_kernel.h"
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

  auto output = habana_helpers::createPTTensor(
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
      auto repeatsShape = habana_helpers::createPTTensor(
          input, repeats, input.options(), false);
      AllocateSynapseShapeTensor(
          graph, repeatsShape, INPUT_DESCRIBING_SHAPE_TENSOR);
    }
  } else {
    TORCH_CHECK(p_context_->syn_inputs_.back().ref().is_input_shape_tensor());
  }

  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

at::Tensor repeat_hpu(const at::Tensor& self, at::IntArrayRef repeats) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = self.scalar_type();
  size_t device_id = self.device().index();
  // Create the operator
  RepeatOperator Op(device_id, scalar_type);
  std::string node_type =
      "tile_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self};
  std::vector<c10::IValue> stack = {c10::IValue(self), c10::IValue(repeats)};
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    auto output = at::empty(
        RepeatOperator::compute_output_shape(self, repeats),
        self.options(),
        self.suggest_memory_format());
    Op.Execute(key, pt_inputs, output);
  } else {
    // Create Graph
    OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  auto output = out.at(0);
  PT_KERNEL_END;
  return output;
}

std::vector<int64_t> RepeatInlvOperator::compute_output_shape(
    const at::Tensor& input,
    int64_t dim,
    int64_t out_size) {
  auto outshape = input.sizes().vec();
  outshape[dim] = out_size;
  return outshape;
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

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)) {
    auto repeats_ht = inputs[1].toTensor();
    TORCH_CHECK(p_context_->syn_inputs_[1].ref().is_host_to_device_tensor());
    auto impl = habana_lazy::GetHbInternalTensorImpl(repeats_ht);
    HABANA_ASSERT(impl);
    TORCH_CHECK(
        impl->get_host_dt_type() == habana_lazy::HostDataType::INT32_T,
        "Incorrect datatype of HOST");

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
  auto output = habana_helpers::createPTTensor(
      input,
      out_shape.sizes(),
      input.options(),
      output_metadata.at(0).persistent);
  // throw away shape tensor before adding synapse node
  p_context_->syn_inputs_.pop_back();
  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add("aten::repeat", KERNEL_FN(RepeatOperator))
        .add("hpu::repeat", KERNEL_FN(RepeatOperator))
        .add("hpu::repeat_inlv", KERNEL_FN(RepeatInlvOperator));
