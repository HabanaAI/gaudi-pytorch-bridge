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
#include "habana_helpers/graph.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/lazy_executor.h"
#include "synapse_helpers/device.h"
#include "synapse_helpers/env_flags.h"
#include "synapse_helpers/layout_utils.h"
#include "synapse_helpers/tensor_builder_base.h"

using tensor_name_generator = synapse_helpers::detail::tensor_name_generator;

const std::array<int64_t, 4>& habana::HabanaOperator::getPermuteOrder(
    const LayoutFormat target_layout,
    bool to_device) {
  static const std::
      unordered_map<const LayoutFormat, const std::array<int64_t, 4>>
          toDevicePermuteOrder = {
              // Host -> Device
              {LayoutFormat::NHWC,
               {LayoutFormatDims::N,
                LayoutFormatDims::H,
                LayoutFormatDims::W,
                LayoutFormatDims::C}}, // NCHW -> NHWC
              {LayoutFormat::NCHW,
               {LayoutFormatDims::N,
                LayoutFormatDims::C,
                LayoutFormatDims::H,
                LayoutFormatDims::W}}, // NCHW -> NCHW (No Change)
              {LayoutFormat::HWCK,
               {LayoutFormatDims::H,
                LayoutFormatDims::W,
                LayoutFormatDims::C,
                LayoutFormatDims::N}}, // KCHW -> HWCK
              {LayoutFormat::ANY,
               {LayoutFormatDims::N,
                LayoutFormatDims::C,
                LayoutFormatDims::H,
                LayoutFormatDims::W}} // XXXX -> XXXX (No Change)
          };

  static const std::
      unordered_map<const LayoutFormat, const std::array<int64_t, 4>>
          toHostPermuteOrder = {
              // Device -> Host
              {LayoutFormat::NCHW,
               {LayoutFormatDims::N,
                LayoutFormatDims::C,
                LayoutFormatDims::H,
                LayoutFormatDims::W}}, // NCHW   -> NCHW (No Change)
              {LayoutFormat::NHWC,
               {LayoutFormatDims::N,
                LayoutFormatDims::W,
                LayoutFormatDims::C,
                LayoutFormatDims::H}}, // NHWC   -> NCHW
              {LayoutFormat::HWCK,
               {LayoutFormatDims::W,
                LayoutFormatDims::H,
                LayoutFormatDims::N,
                LayoutFormatDims::C}}, // HWCK   -> KCHW
              {LayoutFormat::ANY,
               {LayoutFormatDims::N,
                LayoutFormatDims::C,
                LayoutFormatDims::H,
                LayoutFormatDims::W}} // XXXX   -> XXXX (No Change)
          };

  const auto& permuteOrder =
      (to_device ? toDevicePermuteOrder : toHostPermuteOrder);

  TORCH_CHECK(
      permuteOrder.find(target_layout) != permuteOrder.end(),
      "Unknown layout in getPermuteOrder");
  return permuteOrder.find(target_layout)->second;
}

std::vector<int64_t> habana::HabanaOperator::CalculateStrides(
    const at::IntArrayRef sizes,
    c10::MemoryFormat format) {
  switch (sizes.size()) {
    case 5: {
      if (c10::MemoryFormat::ChannelsLast3d == format) {
        return {
            sizes[1] * sizes[2] * sizes[3] * sizes[4],
            1,
            sizes[1] * sizes[3] * sizes[4],
            sizes[1] * sizes[4],
            sizes[1]};
      }
      return {
          sizes[1] * sizes[2] * sizes[3] * sizes[4],
          sizes[4] * sizes[3] * sizes[2],
          sizes[4] * sizes[3],
          sizes[4],
          1};
    }
    case 4: {
      if (c10::MemoryFormat::ChannelsLast == format) {
        return {
            sizes[1] * sizes[2] * sizes[3], 1, sizes[1] * sizes[3], sizes[1]};
      }
      return {sizes[1] * sizes[2] * sizes[3], sizes[3] * sizes[2], sizes[3], 1};
    }
    case 3:
      return {sizes[1] * sizes[2], sizes[2], 1};
    case 2:
      return {sizes[1], 1};
    case 1:
      return {1};
    case 0:
      return {};
    default:
      HABANA_ASSERT(0);
  };
  return {};
}

void habana::HabanaOperator::CreateGraphAndCompile(
    size_t key,
    const std::vector<at::Tensor>& inputs,
    torch::jit::Stack& stack,
    OutputMetaDataVector& output_meta_data,
    bool is_persistent) {
  PT_KERNEL_DEBUG("key:", key);
  //
  // Create Graph
  auto graph = habana_helpers::create_graph(
      p_context_->device_id_, p_context_->node_type_);
  AllocateSynapseInputs(graph, inputs, is_persistent);
  AllocateAndAddSynapseNode(graph, stack, output_meta_data);
  Compile(graph);
}

void habana::HabanaOperator::Compile(synapse_helpers::graph& graph) {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    if (!habana_lazy::isDeviceInLoweringMode()) {
      // Lazy mode shape inference call, early return without execution
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
  PT_KERNEL_DEBUG("Cache hit key:", key);
  habana_helpers::execute_recipe(
      habana_helpers::extract_data_ptrs(p_context_->pt_inputs_),
      habana_helpers::extract_data_ptrs(p_context_->pt_outputs_),
      habana_helpers::extract_storage_data_ptrs(p_context_->pt_inputs_),
      habana_helpers::extract_storage_data_ptrs(p_context_->pt_outputs_),
      p_context_->pt_inputs_,
      p_context_->device_id_,
      p_context_->recipe_key_);
}

void habana::HabanaOperator::Execute(
    size_t key,
    const std::vector<at::Tensor>& inputs) {
  SetPTInputs(inputs);
  Execute(key);
}

void habana::HabanaOperator::Execute(
    size_t key,
    const std::vector<at::Tensor>& inputs,
    const at::Tensor& output) {
  SetPTInputs(inputs);
  SetPTOutput(output);
  Execute(key);
}

void habana::HabanaOperator::Execute(
    size_t key,
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& outputs) {
  SetPTInputs(inputs);
  SetPTOutputs(outputs);
  Execute(key);
}

void habana::HabanaOperator::Execute(
    size_t key,
    const std::vector<at::Tensor>& inputs,
    torch::jit::Stack& output) {
  SetPTInputs(inputs);
  SetPTOutput(output);
  Execute(key);
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

void habana::HabanaOperator::SetPTOutputs(
    const std::vector<at::Tensor>& outputs) {
  TORCH_CHECK(outputs.size() != 0, "Outputs cannot be null");

  for (auto& output : outputs) {
    p_context_->pt_outputs_.emplace_back(output);
  }
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
    synTensorType shape_tensor_type,
    void* host_ptr) {
  PT_BRIDGE_TRACE;
  // TORCH_CHECK(input != nullptr, "Input cannot be null");

  if (!habana_helpers::is_shape_tensor(shape_tensor_type)) {
    if (p_context_->is_duplicate_input_) {
      uint64_t syn_offset = input.storage_offset() * input.itemsize();
      auto sizes = input.sizes().vec();
      auto strides = input.strides().vec();
      auto hb_impl = habana_lazy::GetHbInternalTensorImpl(input);
      TORCH_CHECK(hb_impl, " internal tensor missing for input tensor");
      auto permutation = hb_impl->GetMemoryPermutation();
      auto syn_tensor_input =
          habana_helpers::duplicate_tensor_in_memory_section_with_size(
              p_context_->syn_input_orig_[0],
              graph,
              sizes,
              strides,
              syn_offset,
              false,
              permutation);

      p_context_->syn_inputs_.emplace_back(std::move(syn_tensor_input));
    } else {
      auto syn_tensor_input = habana_helpers::create_tensor(
          input, graph, is_persistent, false, c10::nullopt);
      p_context_->syn_inputs_.emplace_back(std::move(syn_tensor_input));
    }
  } else {
    auto syn_shape_input = habana_helpers::create_shape_tensor(
        input, graph, is_persistent, shape_tensor_type, "", host_ptr);
    p_context_->syn_inputs_.emplace_back(std::move(syn_shape_input));
  }
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
    const at::Tensor& input,
    synTensorType shape_tensor_type,
    void* host_ptr) {
  HABANA_ASSERT(
      shape_tensor_type == SHAPE_TENSOR ||
      shape_tensor_type == HOST_TO_DEVICE_TENSOR ||
      shape_tensor_type == INPUT_DESCRIBING_SHAPE_TENSOR);
  auto syn_shape_input = habana_helpers::create_shape_tensor(
      input, graph, false, shape_tensor_type, "", host_ptr);
  syn_shape_input.set_intermediate_shape_tensor();
  p_context_->syn_inputs_.emplace_back(std::move(syn_shape_input));
  return p_context_->syn_inputs_.back();
}

// Create synapse shape tensor with  shape and device index
synapse_helpers::tensor& habana::HabanaOperator::AllocateSynapseShapeTensor(
    synapse_helpers::graph& graph,
    const at::IntArrayRef& input_shapes,
    synDeviceId syn_device,
    synTensorType shape_tensor_type,
    void* host_ptr) {
  HABANA_ASSERT(
      shape_tensor_type == SHAPE_TENSOR ||
      shape_tensor_type == HOST_TO_DEVICE_TENSOR ||
      shape_tensor_type == INPUT_DESCRIBING_SHAPE_TENSOR);
  auto syn_shape_input = habana_helpers::create_shape_tensor(
      input_shapes, syn_device, graph, false, shape_tensor_type, "", host_ptr);
  syn_shape_input.set_intermediate_shape_tensor();
  p_context_->syn_inputs_.emplace_back(std::move(syn_shape_input));
  return p_context_->syn_inputs_.back();
}

void habana::HabanaOperator::AllocateSynapseOutput(
    synapse_helpers::graph& graph,
    const at::Tensor& output,
    const OutputMetaData& output_metadata,
    bool is_shape_tensor) {
  if (is_shape_tensor == false) {
    p_context_->syn_outputs_.emplace_back(habana_helpers::create_tensor(
        output,
        graph,
        output_metadata.persistent,
        output_metadata.external,
        c10::nullopt,
        output_metadata.name,
        output_metadata.module_name));
  } else {
    p_context_->syn_outputs_.emplace_back(habana_helpers::create_shape_tensor(
        output,
        graph,
        output_metadata.persistent,
        DEVICE_SHAPE_TENSOR,
        output_metadata.name));
  }
  p_context_->pt_outputs_.emplace_back(output);
}

void habana::HabanaOperator::AllocateSynapseOutput(
    synapse_helpers::graph& graph,
    const at::Tensor& output,
    const synDataType synType,
    const OutputMetaData& output_metadata,
    bool is_shape_tensor) {
  std::vector<int64_t> min_shape, max_shape;
  if (is_shape_tensor == false) {
    p_context_->syn_outputs_.emplace_back(habana_helpers::create_tensor(
        output,
        graph,
        output_metadata.persistent,
        output_metadata.external,
        synType,
        output_metadata.name,
        output_metadata.module_name));
  } else {
    p_context_->syn_outputs_.emplace_back(habana_helpers::create_shape_tensor(
        output,
        graph,
        output_metadata.persistent,
        DEVICE_SHAPE_TENSOR,
        output_metadata.name));
  }
  p_context_->pt_outputs_.emplace_back(output);
}

DMAInputGeneratorType habana::HabanaOperator::getDMAInputGeneratorType() {
  HABANA_ASSERT(false, "This call needs to be supported by the derived op");
  return DMAInputGeneratorType::INVALID;
}

std::vector<std::tuple<std::string, at::Tensor, uint64_t>> habana::
    HabanaOperator::getAppendedTensorInfos() {
  return appended_tensor_infos;
}

void habana::HabanaOperator::AllocateSynapseInplaceOutput(
    synapse_helpers::graph& graph,
    bool external) {
  static_cast<void>(graph);
  HABANA_ASSERT(p_context_->syn_inputs_.size() > 0);
  HABANA_ASSERT(p_context_->pt_inputs_.size() > 0);

  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[0], graph, external));
  p_context_->pt_outputs_.emplace_back(p_context_->pt_inputs_[0]);
}

void habana::HabanaOperator::AllocateSynapseOutputs(
    synapse_helpers::graph& graph,
    const std::vector<at::Tensor>& outputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(outputs.size() != 0, "Outputs cannot be null");
  TORCH_CHECK(
      outputs.size() == output_metadata.size(),
      "#output should match #output_metadata");
  for (unsigned int i = 0; i < outputs.size(); ++i) {
    auto& output = outputs.at(i);
    AllocateSynapseOutput(graph, output, output_metadata.at(i), false);
  }
}

void habana::HabanaOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  static_cast<void>(graph);
  static_cast<void>(inputs);
  static_cast<void>(output_metadata);
  TORCH_CHECK(
      0, "Should never reach this empty base AllocateAndAddSynapseNode");
}

void habana::HabanaOperator::ReuseMemoryAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const std::vector<synapse_helpers::tensor_or_ref>& syn_t_vec,
    const OutputMetaDataVector& output_metadata) {
  static_cast<void>(graph);
  static_cast<void>(inputs);
  static_cast<void>(syn_t_vec);
  static_cast<void>(output_metadata);
  TORCH_CHECK(
      0, "Should never reach this empty base ReuseMemoryAndAddSynapseNode");
};

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

habana::OutputShapeInfRetType habana::HabanaOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  static_cast<void>(inputs);
  OutputShapeInfRetType ret(true);
  return ret;
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
        auto& tensor = SynInput(index).ref();
        syn_inputs.emplace_back(tensor.get());
      }
    }
    for (size_t i = 0; i < p_context_->syn_inputs_.size(); ++i) {
      auto& tensor = SynInput(i).ref();
      if (tensor.is_shape_tensor() || tensor.is_input_shape_tensor()) {
        syn_inputs.emplace_back(tensor.get());
      }
    }
  } else {
    for (size_t i = 0; i < p_context_->syn_inputs_.size(); ++i) {
      auto& tensor = SynInput(i).ref();
      syn_inputs.emplace_back(tensor.get());
    }
  }

  for (synapse_helpers::tensor& tensor : p_context_->syn_outputs_) {
    syn_outputs.emplace_back(tensor.get());
  }

  auto input_layouts = synapse_helpers::layouts::getSynapseLayoutFormat(
      kernel_meta_data_.synapse_input_layout);
  auto output_layouts = synapse_helpers::layouts::getSynapseLayoutFormat(
      kernel_meta_data_.synapse_output_layout);

  HABANA_ASSERT(
      input_layouts.empty() || input_layouts.size() >= syn_inputs.size(),
      "Missing layouts for inputs");
  HABANA_ASSERT(
      output_layouts.empty() || output_layouts.size() >= syn_outputs.size(),
      "Missing layouts for outputs");

  graph.add_node(
      std::move(syn_inputs),
      std::move(syn_outputs),
      params,
      params_size,
      guid_,
      nullptr,
      input_layouts.empty() ? nullptr : input_layouts.data(),
      output_layouts.empty() ? nullptr : output_layouts.data(),
      deterministic);
}

habana::RegisterKernel& habana::KernelRegistry() {
  static habana::RegisterKernel* Registry = new habana::RegisterKernel();
  return *Registry;
}

habana::HabanaOperator::~HabanaOperator() = default;

void habana::OutputShapeInfRetType::AddTensor(
    const TensorMetaData& data,
    std::vector<IdxTensorTup>& v) {
  auto sif_tensor_id_ = habana::ShapeInference::ReadAndIncrementSifTensorId();
  auto tensor = habana_helpers::nonPersistentTensor(
      data.sizes, data.strides, data.mf, scalarTypeToTypeMeta(data.dtype));

  v.emplace_back(std::make_tuple(sif_tensor_id_, tensor));
}

void habana::OutputShapeInfRetType::AddOutputTensor(
    const TensorMetaData& data) {
  AddTensor(data, output_tensors);
}

void habana::OutputShapeInfRetType::AddIntermediateTensor(
    const TensorMetaData& data) {
  OutputShapeInfRetType output;
  output.AddOutputTensor(data);
  kernel_outputs.emplace_back(std::make_shared<OutputShapeInfRetType>(output));
}

void habana::OutputShapeInfRetType::AddShapeTensor(const TensorMetaData& data) {
  AddTensor(data, shape_tensors);
}

void habana::OutputShapeInfRetType::AddDupTensor(
    const habana::TensorMetaData& data) {
  AddTensor(data, dup_tensors);
}

habana::OutputShapeInfRetType habana::OutputShapeInfRetType::
    call_ComputeOutputShape(
        HabanaOperatorPtr kernel,
        torch::jit::Stack& inputs) {
  HABANA_ASSERT(kernel.get() != nullptr, "kernel cannot be null");
  auto output = kernel->ComputeOutputShape(inputs);
  kernel_outputs.emplace_back(
      std::make_shared<habana::OutputShapeInfRetType>(output));
  return output;
}

const habana::IdxTensorTup& habana::OutputShapeInfRetType::GetOutputTensor(
    size_t index) {
  HABANA_ASSERT(index <= output_tensors.size(), "index out of range");
  return output_tensors.at(index);
}

const habana::IdxTensorTup& habana::OutputShapeInfRetType::GetShapeTensor(
    size_t index) {
  HABANA_ASSERT(index <= shape_tensors.size(), "index out of range");
  return shape_tensors.at(index);
}

void habana::OutputShapeInfRetType::MoveToOutput(habana::IdxTensorTup&& data) {
  output_tensors.emplace_back(data);
}

void habana::OutputShapeInfRetType::RemoveOutput(size_t index) {
  output_tensors.erase(output_tensors.begin() + index);
}
