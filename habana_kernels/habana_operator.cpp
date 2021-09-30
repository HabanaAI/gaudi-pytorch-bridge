/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "habana_operator.h"
#include "habana_bridge/kernel/hpu_shape_inference.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/lazy_executor.h"
#include "synapse_helpers/device.h"
#include "synapse_helpers/env_flags.h"
#include "synapse_helpers/tensor_builder_base.h"

using tensor_name_generator = synapse_helpers::detail::tensor_name_generator;

const std::array<int64_t, 4>& habana::HabanaOperator::getPermuteOrder(
    const LayoutFormat target_layout,
    bool to_device) {
  static const std::
      unordered_map<const LayoutFormat, const std::array<int64_t, 4>>
          toDevicePermuteOrder = {
              // Host -> Device
              {LayoutFormat::NHWC, {0, 2, 3, 1}}, // NCHW -> NHWC
              {LayoutFormat::NCHW, {0, 1, 2, 3}}, // NCHW -> NCHW (No Change)
              {LayoutFormat::HWCK, {2, 3, 1, 0}}, // KCHW -> HWCK
              {LayoutFormat::ANY, {0, 1, 2, 3}} // XXXX -> XXXX (No Change)
          };

  static const std::
      unordered_map<const LayoutFormat, const std::array<int64_t, 4>>
          toHostPermuteOrder = {
              // Device -> Host
              {LayoutFormat::NCHW, {0, 1, 2, 3}}, // NCHW   -> NCHW (No Change)
              {LayoutFormat::NHWC, {0, 3, 1, 2}}, // NHWC   -> NCHW
              {LayoutFormat::HWCK, {3, 2, 0, 1}}, // HWCK   -> KCHW
              {LayoutFormat::ANY, {0, 1, 2, 3}} // XXXX   -> XXXX (No Change)
          };

  const auto& permuteOrder =
      (to_device ? toDevicePermuteOrder : toHostPermuteOrder);

  TORCH_CHECK(
      permuteOrder.find(target_layout) != permuteOrder.end(),
      "Unknown layout in getPermuteOrder");
  return permuteOrder.find(target_layout)->second;
}

void habana::HabanaOperator::Compile(synapse_helpers::graph& graph) {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    synapse_helpers::device& device = graph.get_device();
    auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
        (int)(device.id()));
    if (context != nullptr) {
      // Lazy mode shape inference call, early return without execution
      if (!(context->isExecutionInLoweringMode()))
        return;
    }
  }

  // compile the graph
  habana_helpers::compile_and_run(
      std::move(graph),
      habana_helpers::names(p_context_->syn_inputs_),
      habana_helpers::names(p_context_->syn_outputs_),
      habana_helpers::extract_data_ptrs(p_context_->pt_inputs_),
      habana_helpers::extract_data_ptrs(p_context_->pt_outputs_),
      habana_helpers::extract_storage_data_ptrs(p_context_->pt_inputs_),
      habana_helpers::extract_storage_data_ptrs(p_context_->pt_outputs_),
      p_context_->pt_inputs_,
      p_context_->device_id_,
      p_context_->recipe_key_);
}

void habana::HabanaOperator::Execute(size_t key) {
  static_cast<void>(key);
  //
  // Execute the graph
  habana_helpers::execute_recipe(
      habana_helpers::extract_data_ptrs(p_context_->pt_inputs_),
      habana_helpers::extract_data_ptrs(p_context_->pt_outputs_),
      habana_helpers::extract_storage_data_ptrs(p_context_->pt_inputs_),
      habana_helpers::extract_storage_data_ptrs(p_context_->pt_outputs_),
      p_context_->pt_inputs_,
      p_context_->device_id_,
      p_context_->recipe_key_);
}

void habana::HabanaOperator::SetPTInputs(
    const std::vector<at::Tensor>& inputs) {
  for (auto& input : inputs) {
    p_context_->pt_inputs_.emplace_back(input);
  }
}

void habana::HabanaOperator::SetPTOutput(const at::Tensor& output) {
  p_context_->pt_outputs_.emplace_back(output);
}

void habana::HabanaOperator::SetPTOutput(torch::jit::Stack& inputs) {
  static_cast<void>(inputs);
  TORCH_CHECK(0, "Should never reach this empty base SetPTOutput Stack");
}

void habana::HabanaOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  static_cast<void>(inputs);
  TORCH_CHECK(0, "Should never reach this empty base SetPTOutputs Stack");
}

void habana::HabanaOperator::SetPTOutputs(std::vector<at::Tensor>& outputs) {
  TORCH_CHECK(outputs.size() != 0, "Outputs cannot be null");

  for (auto& output : outputs) {
    p_context_->pt_outputs_.emplace_back(output);
  }
}

void habana::HabanaOperator::SetOutputMetadata(
    int index,
    const OutputMetaData& md) {
  output_metadata_[index] = md;
}

void habana::HabanaOperator::SetOutputMetadata(const OutputMetaDataVector& md) {
  output_metadata_ = md;
}

size_t habana::HabanaOperator::GetRecipeKey(
    std::string node,
    std::vector<c10::IValue> stack,
    bool inPlaceOp,
    bool outOp) {
  size_t key = habana_helpers::getRecipeKey(node, stack, inPlaceOp, outOp);
  p_context_->recipe_key_ = key;
  return key;
}

synapse_helpers::tensor& habana::HabanaOperator::AllocateSynapseInput(
    synapse_helpers::graph& graph,
    const at::Tensor& input,
    bool is_persistent,
    bool is_shape_tensor) {
  // TORCH_CHECK(input != nullptr, "Input cannot be null");

  if (is_shape_tensor == false) {
    if (p_context_->is_duplicate_input_) {
      uint64_t syn_offset = input.storage_offset() * input.itemsize();
      auto sizes = input.sizes().vec();
      auto strides = input.strides().vec();
      auto syn_tensor_input =
          habana_helpers::duplicate_tensor_in_memory_section_with_size(
              p_context_->syn_input_orig_[0],
              graph,
              sizes,
              strides,
              syn_offset);
      p_context_->syn_inputs_.emplace_back(std::move(syn_tensor_input));
    } else {
      auto syn_tensor_input = habana_helpers::create_tensor(
          input, graph, is_persistent, c10::nullopt);
      p_context_->syn_inputs_.emplace_back(std::move(syn_tensor_input));
    }
  } else {
    auto syn_shape_input =
        habana_helpers::create_shape_tensor(input, graph, is_persistent, false);
    p_context_->syn_inputs_.emplace_back(std::move(syn_shape_input));
  }
  PT_BRIDGE_DEBUG("AllocateSynapseInput ", p_context_->syn_inputs_.back());
  p_context_->pt_inputs_.emplace_back(input);
  return p_context_->syn_inputs_.back();
}

void habana::HabanaOperator::AllocateSynapseInputs(
    synapse_helpers::graph& graph,
    const std::vector<at::Tensor>& inputs,
    bool is_persistent) {
  // TORCH_CHECK(!inputs.empty(), "Inputs cannot be null");

  for (auto& input : inputs) {
    AllocateSynapseInput(graph, input, is_persistent);
  }
}

// Create synapse shape tensor without corresponding pt_tensor
synapse_helpers::tensor& habana::HabanaOperator::AllocateSynapseShapeTensor(
    synapse_helpers::graph& graph,
    const at::Tensor& input) {
  auto syn_shape_input =
      habana_helpers::create_shape_tensor(input, graph, false, false);
  p_context_->syn_inputs_.emplace_back(std::move(syn_shape_input));
  PT_BRIDGE_DEBUG(
      "AllocateSynapseShapeTensor ", p_context_->syn_inputs_.back());
  return p_context_->syn_inputs_.back();
}

void habana::HabanaOperator::AllocateSynapseOutput(
    synapse_helpers::graph& graph,
    const at::Tensor& output,
    bool is_persistent,
    bool is_shape_tensor,
    bool use_metadata) {
  const std::string& synName =
      use_metadata && output_allocation_index_ < output_metadata_.size()
      ? output_metadata_.at(output_allocation_index_++).name
      : guid_;

  if (is_shape_tensor == false) {
    p_context_->syn_outputs_.emplace_back(habana_helpers::create_tensor(
        output, graph, is_persistent, c10::nullopt, synName));
  } else {
    p_context_->syn_outputs_.emplace_back(habana_helpers::create_shape_tensor(
        output, graph, is_persistent, true, synName));
  }
  PT_BRIDGE_DEBUG("AllocateSynapseOutput ", p_context_->syn_outputs_.back());
  p_context_->pt_outputs_.emplace_back(output);
}

void habana::HabanaOperator::AllocateSynapseOutput(
    synapse_helpers::graph& graph,
    const at::Tensor& output,
    const synDataType synType,
    bool is_persistent,
    bool is_shape_tensor,
    bool use_metadata) {
  std::vector<int64_t> min_shape, max_shape;
  const std::string& synName =
      use_metadata && output_allocation_index_ < output_metadata_.size()
      ? output_metadata_.at(output_allocation_index_++).name
      : guid_;

  if (is_shape_tensor == false) {
    p_context_->syn_outputs_.emplace_back(habana_helpers::create_tensor(
        output, graph, is_persistent, synType, synName));
  } else {
    p_context_->syn_outputs_.emplace_back(habana_helpers::create_shape_tensor(
        output, graph, is_persistent, true, synName));
  }
  PT_BRIDGE_DEBUG("AllocateSynapseOutput ", p_context_->syn_outputs_.back());
  p_context_->pt_outputs_.emplace_back(output);
}

getDMAInputTensorCBType habana::HabanaOperator::getDMAInputTensorCB() {
  HABANA_ASSERT(false, "This call needs to be supported by the derived op");
  return {};
}

std::vector<std::pair<std::string, at::Tensor>> habana::HabanaOperator::
    getAppendedTensorInfos() {
  return appended_tensor_infos;
}

void habana::HabanaOperator::AllocateSynapseInplaceOutput(
    synapse_helpers::graph& graph) {
  static_cast<void>(graph);
  HABANA_ASSERT(p_context_->syn_inputs_.size() > 0);
  HABANA_ASSERT(p_context_->pt_inputs_.size() > 0);

  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[0], graph));

  p_context_->pt_outputs_.emplace_back(p_context_->pt_inputs_[0]);
}

void habana::HabanaOperator::AllocateSynapseOutputs(
    synapse_helpers::graph& graph,
    const std::vector<at::Tensor>& outputs,
    std::vector<bool> is_persistent,
    std::vector<bool> use_metadata) {
  TORCH_CHECK(outputs.size() != 0, "Outputs cannot be null");
  TORCH_CHECK(
      outputs.size() == is_persistent.size(),
      "#output should match #persistent flag");
  TORCH_CHECK(
      outputs.size() == use_metadata.size(),
      "#output should match #use_metadata flag");
  for (unsigned int i = 0; i < outputs.size(); ++i) {
    auto& output = outputs.at(i);
    AllocateSynapseOutput(
        graph, output, is_persistent[i], false, use_metadata[i]);
  }
}

void habana::HabanaOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  static_cast<void>(graph);
  static_cast<void>(inputs);
  static_cast<void>(is_output_persistent);
  TORCH_CHECK(
      0, "Should never reach this empty base AllocateAndAddSynapseNode");
}

void habana::HabanaOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  static_cast<void>(graph);
  static_cast<void>(inputs);
  static_cast<void>(is_output_persistent);
  TORCH_CHECK(
      0, "Should never reach this empty base AllocateAndAddSynapseNode");
}

synapse_helpers::tensor_or_ref& habana::HabanaOperator::SetSynapseInput(
    UNUSED synapse_helpers::tensor_or_ref&& tensor) {
  TORCH_CHECK(
      0, "Should never reach this SetSynapseInput, avoid using std::move");
}

synapse_helpers::tensor_or_ref& habana::HabanaOperator::SetSynapseInput(
    synapse_helpers::tensor& tensor) {
  //
  // The tensor already exists and hence we just add this to the context
  // no need to convert to synapse tensor
  p_context_->syn_inputs_.emplace_back(tensor);
  return p_context_->syn_inputs_.back();
}

synapse_helpers::tensor_or_ref& habana::HabanaOperator::SetSynapseOutput(
    synapse_helpers::tensor_or_ref&& tensor) {
  //
  // The tensor already exists and hence we just add this to the context
  // no need to convert to synapse tensor
  p_context_->syn_outputs_.emplace_back(std::move(tensor));
  return p_context_->syn_outputs_.back();
}

void habana::HabanaOperator::AddNodeToSynapseGraph(
    synapse_helpers::graph& graph,
    void* params,
    size_t params_size) {
  if (graph.is_dry_run()) {
    // Lazy mode shape inference call, early return without execution
    return;
  }
  std::vector<synTensor> syn_inputs;
  std::vector<synTensor> syn_outputs;

  if (kernel_meta_data_.tpc_input_order.size()) {
    auto no_inputs = kernel_meta_data_.tpc_input_order.size() == 1 &&
        NO_INPUTS == kernel_meta_data_.tpc_input_order[0];
    if (no_inputs == false) {
      for (auto index : kernel_meta_data_.tpc_input_order) {
        HABANA_ASSERT(index < p_context_->syn_inputs_.size());
        synapse_helpers::tensor& tensor = p_context_->syn_inputs_[index];
        syn_inputs.emplace_back(tensor.get());
      }
    }
    for (synapse_helpers::tensor& tensor : p_context_->syn_inputs_) {
      if (tensor.is_shape_tensor()) {
        syn_inputs.emplace_back(tensor.get());
      }
    }
  } else {
    for (synapse_helpers::tensor& tensor : p_context_->syn_inputs_) {
      syn_inputs.emplace_back(tensor.get());
    }
  }

  for (synapse_helpers::tensor& tensor : p_context_->syn_outputs_) {
    syn_outputs.emplace_back(tensor.get());
  }

  graph.add_node(
      std::move(syn_inputs),
      std::move(syn_outputs),
      params,
      params_size,
      std::move(guid_));
}

habana::RegisterKernel& habana::KernelRegistry() {
  static habana::RegisterKernel* Registry = new habana::RegisterKernel();
  return *Registry;
}

habana::HabanaOperator::~HabanaOperator() = default;
