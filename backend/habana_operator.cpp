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
#include "backend/habana_operator.h"
#include "backend/create_pt_tensor.h"
#include "backend/habana_device/HPUStream.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/graph.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/kernel/hpu_shape_inference.h"
#include "backend/kernel_recipe_signature.h"
#include "backend/lazy_to_backend.h"
#include "backend/synapse_helpers/device.h"
#include "backend/synapse_helpers/env_flags.h"
#include "backend/synapse_helpers/layout_utils.h"
#include "backend/synapse_helpers/tensor_builder_base.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/lazy_kernels_declarations.h"

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

bool habana::HabanaOperator::isFp8Op(const std::string& guid) {
  static const std::vector<std::string> fp8_ops{
      "cast_from_fp8_f32",
      "cast_from_fp8_bf16",
      "cast_from_fp8_i8",
      "cast_to_fp8_f32",
      "cast_to_fp8_bf16",
      "cast_to_fp8_v2_f32",
      "cast_to_fp8_v2_bf16",
      "fp8_cast_transpose_f32",
      "fp8_cast_transpose_bf16",
      "fp8_cast_transpose_bgrad_f32",
      "fp8_cast_transpose_bgrad_bf16",
      "fp8_cast_transpose_bgrad_dgelu_f32",
      "fp8_cast_transpose_bgrad_dgelu_bf16",
      "fp8_dropout_f32",
      "fp8_dropout_bf16",
      "fp8_gelu_f32",
      "fp8_gelu_bf16",
      "fp8_bgrad_dgelu_f32",
      "fp8_bgrad_dgelu_bf16",
      "fp8_gemm_i8",
      "fp8_gemm_v2_i8",
      "fp8_layernorm_f32",
      "fp8_layernorm_bf16",
      "fp8_reshape_i8",
      "fp8_transpose_i8",
      "fp8_permute_i8"};

  return std::any_of(fp8_ops.begin(), fp8_ops.end(), [&guid](const auto& op) {
    return op == guid;
  });
}

std::string habana::get_guid_with_precision(
    const std::string& guid,
    c10::ScalarType dtype,
    bool use_int64) {
  static const absl::flat_hash_set<std::string> synapse_guids = {
      // Matrix operations
      "batch_gemm",
      "batch_gemm_dedw",
      "batch_gemm_dedx",
      "spatial_convolution",
      "spatial_convolution3d",
      "dedw",
      "dedw3d",
      "dedx",
      "dedx3d",
      "gemm",
      "gemm_dedw",
      "gemm_dedx",
      "masked_batch_gemm",
      // Data movment guids
      "broadcast",
      "concat",
      "expand_dims",
      "flatten",
      "identity",
      "memcpy",
      "memset",
      "reinterpret_cast",
      "reshape",
      "slice",
      "slice_axis",
      "slice_bwd",
      "slice_insert",
      "split",
      "split_shape",
      "squeeze",
      "strided_insert",
      "strided_slice_grad",
      "strided_view",
      "transpose",
      // Normalization
      "cud_bn_bwd_ex",
      "cud_bn_fwd_ex",
      "frobenius_norm_fwd",
      "moments_fwd",
      // Misc
      "einsum",
      "topk",
  };
  // Synapse guids do not take precision type/suffix
  if (synapse_guids.count(guid)) {
    return guid;
  }

  auto string_or_error = synapse_helpers::graph::name_suffix_from_type(
      habana_helpers::pytorch_to_synapse_type(dtype), use_int64);
  HABANA_ASSERT(
      absl::holds_alternative<std::string>(string_or_error),
      "Error getting suffix/precision type: ",
      Logger::synStatusToStr(
          absl::get<synapse_helpers::synapse_error>(string_or_error).status));

  return guid + '_' + absl::get<std::string>(string_or_error);
}

std::vector<int64_t> habana::HabanaOperator::CalculateStrides(
    const at::IntArrayRef sizes,
    c10::MemoryFormat format) {
  std::vector<int64_t> result;
  if (((sizes.size() == 5) && (format == c10::MemoryFormat::ChannelsLast3d)) ||
      ((sizes.size() == 4) && (format == c10::MemoryFormat::ChannelsLast))) {
    std::vector<int64_t> prod(sizes.begin() + 2, sizes.end());
    prod.push_back(sizes[1]);
    for (int i = prod.size() - 2; i >= 0; --i) {
      prod[i] *= prod[i + 1];
    }

    result.push_back(prod[0]);
    result.push_back(1);
    result.insert(result.end(), prod.begin() + 1, prod.end());
  } else {
    if (!sizes.empty()) {
      result.insert(result.end(), sizes.begin() + 1, sizes.end());
      result.push_back(1);
    }

    for (int i = result.size() - 3; i >= 0; --i) {
      result[i] *= result[i + 1];
    }
  }
  return result;
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

namespace {
struct ResourceHolder {
  std::unique_ptr<synapse_helpers::device_ptr_lock> address_lock;
};
} // namespace

static size_t getRecipeKey(
    std::string node,
    std::vector<c10::IValue> stack,
    bool inPlaceOp,
    bool outOp) {
  habana_helpers::RecipeSignature rs(true, stack, {node}, inPlaceOp, outOp);
  return rs.hash();
}

static void launchRecipe(
    const std::vector<void*>& input_buffers,
    const std::vector<void*>& output_buffers,
    std::vector<synapse_helpers::device_ptr> in_event_addr,
    std::vector<synapse_helpers::device_ptr> out_event_addr,
    std::vector<at::Tensor>& pt_inputs,
    const uint32_t device_id,
    std::shared_ptr<synapse_helpers::recipe>& recipe) {
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  auto& stream_handle = device.get_stream(c10::hpu::getCurrentHPUStream());
  std::unique_ptr<synapse_helpers::device_ptr_lock> address_lock;
  if (device.IsStreamASyncEnabled()) {
    // wait for input DMA to complete before launching the compute.
    device.add_wait_events_on_stream(in_event_addr, stream_handle);

    auto& recipe_counter = device.get_active_recipe_counter();
    bool status = recipe->launch(
        input_buffers, output_buffers, address_lock, stream_handle);
    if (!status) {
      TORCH_CHECK(false, "syn launch failed");
    }
    recipe_counter.increase();
    auto holder = std::make_shared<ResourceHolder>();
    holder->address_lock = std::move(address_lock);
    const auto& recipe_ptr = recipe->getRecipeHandle();
    // Get the reference to the tensor it is operating on to prevent
    // it from being deallocated while the operation is still in flight.
    // so use copy of pt_input in callback
    // regsiter an event on the compute
    device.register_producer_on_stream(
        std::move(out_event_addr),
        stream_handle,
        [pt_inputs, recipe_ptr, &recipe_counter, holder]() {
          recipe_counter.decrease_and_notify();
          return;
        });
  } else {
    recipe->launch(input_buffers, output_buffers, address_lock, stream_handle);
    TORCH_HABANA_CHECK(
        synStreamSynchronize(stream_handle), "synStreamSynchronize failed");
  }
}

static void execute_recipe(
    const std::vector<void*>& input_buffers,
    const std::vector<void*>& output_buffers,
    std::vector<synapse_helpers::device_ptr> in_event_addr,
    std::vector<synapse_helpers::device_ptr> out_event_addr,
    std::vector<at::Tensor>& pt_inputs,
    const uint32_t device_id,
    size_t key) {
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  auto recipe = device.get_recipe_handle_cache().get_recipe(key);
  AT_ASSERT(recipe != nullptr);
  if (recipe != nullptr) {
    launchRecipe(
        input_buffers,
        output_buffers,
        in_event_addr,
        out_event_addr,
        pt_inputs,
        device_id,
        recipe);
  }
}
static void compile_and_run(
    synapse_helpers::graph&& graph,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<void*>& input_buffers,
    const std::vector<void*>& output_buffers,
    std::vector<synapse_helpers::device_ptr> in_event_addr,
    std::vector<synapse_helpers::device_ptr> out_event_addr,
    std::vector<at::Tensor>& pt_inputs,
    const uint32_t device_id,
    size_t key) {
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::shared_ptr<synapse_helpers::recipe> recipe = nullptr;
  if (key > 0 && device.IsCachingEnabled()) {
    recipe = device.get_recipe_handle_cache().get_recipe(key, graph);
  } else {
    recipe = std::make_shared<synapse_helpers::recipe>(device);
    recipe->create(graph);
  }
  AT_ASSERT(recipe != nullptr);
  if (recipe != nullptr) {
    recipe->set_inputs_outputs_names(input_names, output_names);
    launchRecipe(
        input_buffers,
        output_buffers,
        in_event_addr,
        out_event_addr,
        pt_inputs,
        device_id,
        recipe);
  }
}
void habana::HabanaOperator::Compile(synapse_helpers::graph& graph) {
  if (lazy_to_backend::is_lazy_inference_call_context())
    return;

  // compile the graph
  compile_and_run(
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
  execute_recipe(
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
  size_t key = getRecipeKey(node, stack, inPlaceOp, outOp);
  p_context_->recipe_key_ = key;
  return key;
}

synapse_helpers::tensor& habana::HabanaOperator::AllocateSynapseInput(
    synapse_helpers::graph& graph,
    const at::Tensor& input,
    bool is_persistent,
    synTensorType shape_tensor_type,
    void* host_ptr,
    const std::string& idx) {
  PT_BRIDGE_TRACE;
  // TORCH_CHECK(input != nullptr, "Input cannot be null");
  if (!habana_helpers::is_shape_tensor(shape_tensor_type)) {
    if (p_context_->is_duplicate_input_) {
      uint64_t syn_offset = input.storage_offset() * input.itemsize();
      auto sizes = input.sizes().vec();
      auto strides = input.strides().vec();
      std::vector<uint8_t> permutation;
      bool dont_allow_permutation = false;
      std::tie(permutation, dont_allow_permutation) =
          habana_helpers::get_tensor_memory_permutation(input);
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
    } else if (input.scalar_type() == c10::ScalarType::Char && isFp8Op(guid_)) {
      // fp8 tensors are exposed to Pytorch viatorch.uint8 type, therefor for
      // fp8 ops synTensors must have manually set syn_type_fp8_152 data type
      p_context_->syn_inputs_.emplace_back(habana_helpers::create_tensor(
          input, graph, is_persistent, false, syn_type_fp8_152));
    } else {
      p_context_->syn_inputs_.emplace_back(habana_helpers::create_tensor(
          input, graph, is_persistent, false, c10::nullopt, idx, idx));
    }
  } else {
    p_context_->syn_inputs_.emplace_back(habana_helpers::create_shape_tensor(
        input, graph, is_persistent, shape_tensor_type, "", host_ptr));
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
  // Increment count for shape tensors
  graph.increment_shape_tensors();
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
        output_metadata.module_name + '.' +
            std::to_string(p_context_->syn_outputs_.size())));
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

synapse_helpers::tensor_or_ref& habana::HabanaOperator::SetSynapseInput([
    [maybe_unused]] synapse_helpers::tensor_or_ref&& tensor) {
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

// Allocate constant synapse tensor of size '1' for handling scalars
synapse_helpers::tensor habana::HabanaOperator::AllocateConstantSynapseTensor(
    synapse_helpers::graph& graph,
    const c10::Scalar& scalar_val) {
  // Double data type not supported in synapse convert it to float value on host
  const auto& scalar_val_type = (scalar_val.type() == at::ScalarType::Double)
      ? at::ScalarType::Float
      : scalar_val.type();

  void* host_ptr = nullptr;
  const auto& host_ptr_size = elementSize(scalar_val_type);
  auto& device =
      synapse_helpers::HPURegistrar::get_device(p_context_->device_id_);
  auto status = device.get_host_memory().malloc(&host_ptr, host_ptr_size);
  HABANA_ASSERT(status == synSuccess, Logger::synStatusToStr(status));

  if (scalar_val.type() == at::ScalarType::Double) {
    // WA for copying float data to host_ptr
    // If c10::Scalar is initialized with Float value
    // its data type is still seen Double
    auto lval = scalar_val.to<float>();
    memcpy(host_ptr, (const char*)&lval, host_ptr_size);
  } else {
    memcpy(host_ptr, scalar_val.data_ptr(), host_ptr_size);
  }

  PT_KERNEL_DEBUG(
      "constant host_ptr: ",
      reinterpret_cast<size_t>(host_ptr),
      " scalar value: ",
      Logger::_str_wrapper(scalar_val),
      " size: ",
      host_ptr_size,
      " org data_type: ",
      scalar_val.type());

  auto const_syn_tensor = habana_helpers::create_const_tensor(
      {1},
      {1},
      graph,
      false,
      p_context_->device_id_,
      scalar_val_type,
      host_ptr,
      host_ptr_size);

  // Free host_ptr here only since copy_buffer is set true for const tensor
  device.get_host_memory().free(host_ptr);

  // Increment count for const tensors created for scalars
  graph.increment_const_tensors();

  return std::move(const_syn_tensor);
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
  auto tensor = habana::nonPersistentTensor(
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
