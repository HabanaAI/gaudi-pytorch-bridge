/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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
#include <algorithm>
#include "backend/create_pt_tensor.h"
#include "backend/habana_device/HPUStream.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/runtime_config.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/kernel/hpu_shape_inference.h"
#include "backend/kernel_recipe_signature.h"
#include "backend/lazy_to_backend.h"
#include "backend/synapse_helpers/device.h"
#include "backend/synapse_helpers/layout_utils.h"
#include "habana_helpers/logging.h"
#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/op_logger.h"

static const synDataType fp8_syn_type =
    GET_ENV_FLAG_NEW(PT_USE_FP8_143) ? syn_type_fp8_143 : syn_type_fp8_152;

bool habana::HabanaOperator::isFp8Op(const std::string_view guid) {
  using namespace std::literals;
  // Note: For 26 items, benchmark shown 6% improvement using constexpr array
  // (+any_of) over absl::flat_hash_set (+contains). Adding more items requires
  // reevaluation of results. For reference on why
  // https://www.youtube.com/watch?v=INn3xa4pMfg
  static constexpr std::array<std::string_view, 32> fp8_ops{
      "cast_from_fp8_f32"sv,
      "cast_from_fp8_bf16"sv,
      "cast_from_fp8_i8"sv,
      "cast_to_fp8_f32"sv,
      "cast_to_fp8_bf16"sv,
      "cast_to_fp8_v2_f32"sv,
      "cast_to_fp8_v2_bf16"sv,
      "fp8_cast_transpose_f32"sv,
      "fp8_cast_transpose_bf16"sv,
      "fp8_cast_transpose_bgrad_f32"sv,
      "fp8_cast_transpose_bgrad_bf16"sv,
      "fp8_cast_transpose_bgrad_dgelu_f32"sv,
      "fp8_cast_transpose_bgrad_dgelu_bf16"sv,
      "fp8_copy__i8"sv,
      "fp8_dropout_f32"sv,
      "fp8_dropout_bf16"sv,
      "fp8_gelu_f32"sv,
      "fp8_gelu_bf16"sv,
      "fp8_bgrad_dgelu_f32"sv,
      "fp8_bgrad_dgelu_bf16"sv,
      "fp8_gemm_i8"sv,
      "fp8_gemm_v2_i8"sv,
      "fp8_layernorm_f32"sv,
      "fp8_layernorm_bf16"sv,
      "fp8_reshape_i8"sv,
      "fp8_transpose_i8"sv,
      "fp8_permute_i8"sv,
      "fp8_kv_reorder_i8"sv,
      "fp8_index_copy__i8"sv,
      "fp8_repeat_v2_i8"sv,
      "fp8_index_select_v2_i8"sv,
      "in_place_interleave__i8"sv};

  return std::any_of(
      fp8_ops.begin(), fp8_ops.end(), [&guid](auto&& v) { return v == guid; });
}

std::string habana::get_guid_with_precision(
    const std::string_view guid,
    c10::ScalarType dtype,
    bool use_int64) {
  using namespace std::literals;
  static const absl::flat_hash_set<std::string_view> synapse_guids = {
      // Matrix operations
      "batch_gemm"sv,
      "batch_gemm_dedw"sv,
      "batch_gemm_dedx"sv,
      "spatial_convolution"sv,
      "spatial_convolution3d"sv,
      "dedw"sv,
      "dedw3d"sv,
      "dedx"sv,
      "dedx3d"sv,
      "gemm"sv,
      "gemm_dedw"sv,
      "gemm_dedx"sv,
      "masked_batch_gemm"sv,
      // Data movment guids
      "broadcast"sv,
      "concat"sv,
      "expand_dims"sv,
      "flatten"sv,
      "identity"sv,
      "memcpy"sv,
      "memset"sv,
      "reinterpret_cast"sv,
      "reshape"sv,
      "slice"sv,
      "slice_axis"sv,
      "slice_bwd"sv,
      "slice_insert"sv,
      "split"sv,
      "split_shape"sv,
      "squeeze"sv,
      "strided_insert"sv,
      "strided_slice_grad"sv,
      "strided_view"sv,
      "transpose"sv,
      // Normalization
      "cud_bn_bwd_ex"sv,
      "cud_bn_fwd_ex"sv,
      "frobenius_norm_fwd"sv,
      "moments_fwd"sv,
      // Misc
      "einsum"sv,
      "topk"sv,
  };
  // Synapse guids do not take precision type/suffix
  if (synapse_guids.contains(guid) or dtype == c10::ScalarType::Undefined) {
    return std::string{guid};
  }

  std::string updated_guid = std::string{guid};
  update_guid_trunc_mode(updated_guid, dtype);

  auto type_name = synapse_helpers::graph::name_suffix_from_type(
      habana_helpers::pytorch_to_synapse_type(dtype), use_int64);

  return updated_guid.append(1, '_').append(type_name);
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
  auto& device = habana::HPURegistrar::get_device(device_id).syn_device();
  auto& stream_handle = device.get_stream(c10::hpu::getCurrentHPUStream());
  std::unique_ptr<synapse_helpers::device_ptr_lock> address_lock;
  if (device.IsStreamASyncEnabled()) {
    // wait for input DMA to complete before launching the compute.
    device.add_wait_events_on_stream(in_event_addr, stream_handle);

    auto& recipe_counter = device.get_active_recipe_counter();
    recipe->launch(input_buffers, output_buffers, address_lock, stream_handle);
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
  auto& device = habana::HPURegistrar::get_device(device_id).syn_device();
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
  auto& device = habana::HPURegistrar::get_device(device_id).syn_device();
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
  if (input.scalar_type() == c10::ScalarType::Long &&
      !common::IsInt64Supported()) {
    auto tmeta{habana::get_tensor_extra_meta(input)};
    if (tmeta && tmeta->is_view_tensor() && input.storage_offset() &&
        guid_.find("gather_elements_fwd") != std::string::npos) {
      auto* impl = input.unsafeGetTensorImpl();
      impl->set_storage_and_dtype(
          input.storage(), c10::scalarTypeToTypeMeta(c10::ScalarType::Int));
    }
  }
  if (!habana_helpers::is_shape_tensor(shape_tensor_type)) {
    if (p_context_->is_duplicate_input_) {
      uint64_t syn_offset = input.storage_offset() * input.itemsize();
      auto sizes = input.sizes().vec();
      auto strides = input.strides().vec();
      synapse_helpers::layouts::MemoryPermutation permutation;
      std::tie(permutation, std::ignore) =
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
      // fp8 tensors are exposed to Pytorch via torch.int8 type, therefor for
      // fp8 ops synTensors must have manually set syn_type_fp8_152/143 type
      p_context_->syn_inputs_.emplace_back(habana_helpers::create_tensor(
          input, graph, is_persistent, false, fp8_syn_type));
    } else if (
        // int4/uint4 tensors are exposed to Pytorch via torch.int type,
        // therefor for int4 ops synTensors must have manually set
        // syn_type_int4/uint4 type
        (guid_ == "convert_from_int4_i32" ||
         guid_ == "convert_from_uint4_i32") &&
        input.scalar_type() == c10::ScalarType::Int) {
      auto syn_type =
          guid_ == "convert_from_int4_i32" ? syn_type_int4 : syn_type_uint4;
      p_context_->syn_inputs_.emplace_back(habana_helpers::create_tensor(
          input, graph, is_persistent, false, syn_type));
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
      shape_tensor_type == HOST_TO_DEVICE_TENSOR);
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
      shape_tensor_type == HOST_TO_DEVICE_TENSOR);
  auto syn_shape_input = habana_helpers::create_shape_tensor(
      input_shapes, syn_device, graph, false, shape_tensor_type, "", host_ptr);
  syn_shape_input.set_intermediate_shape_tensor();
  graph.increment_shape_tensors();
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

bool habana::HabanaOperator::STMeta(
    habana_helpers::IShapeList& inputs,
    habana_helpers::IShapeList& outputs) {
  static_cast<void>(inputs);
  static_cast<void>(outputs);
  PT_BRIDGE_DEBUG("ST meta called for base HabanaOperator!!!");
  return false;
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

habana::InferOutputMetaRetType habana::HabanaOperator::InferOutputMeta(
    torch::jit::Stack&) {
  return InferOutputMetaRetType(true);
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
      if (tensor.is_shape_tensor()) {
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
      deterministic,
      getContextHints());
}

#define CONVERT_SCALAR(type)                           \
  auto size = c10::elementSize(c10::ScalarType::type); \
  auto data = scalar_val.to##type();                   \
  val.resize(size);                                    \
  memcpy(val.data(), (const char*)&data, size);

#define CASE(type, ...)         \
  case c10::ScalarType::type: { \
    __VA_ARGS__;                \
    CONVERT_SCALAR(type);       \
  } break;

// Special case
#define CASE_DOUBLE(...)          \
  case c10::ScalarType::Double: { \
    __VA_ARGS__;                  \
    CONVERT_SCALAR(Float);        \
  } break;

// Allocate constant synapse tensor of size '1' for handling scalars
synapse_helpers::tensor habana::HabanaOperator::AllocateConstantSynapseTensor(
    synapse_helpers::graph& graph,
    const c10::Scalar& scalar_val,
    c10::optional<at::ScalarType> force_type) {
  auto val_type = scalar_val.type();

  const auto init_val_size = elementSize(val_type);
  std::vector<uint8_t> val(init_val_size);
  memcpy(val.data(), scalar_val.data_ptr(), init_val_size);
  c10::Scalar force_long_scalar_val;
  // Handle force type i.e. convert it to force data type
  if (force_type.has_value()) {
    val_type = force_type.value();
    switch (val_type) {
      CASE(Char);
      CASE(Byte);
      CASE(Short);
      CASE(Int);
      CASE(Long, force_long_scalar_val = scalar_val.toLong());
      CASE(Float);
      // Double data type not supported, always convert it to float
      CASE_DOUBLE(val_type = c10::ScalarType::Float);
      CASE(Half);
      CASE(Bool);
      CASE(BFloat16);
#if HAVE_FP8_SUPPORT
      CASE(Float8_e5m2);
      CASE(Float8_e4m3fn);
#endif
      default:
        HABANA_ASSERT(0, "Unsupported scalar type: ", val_type);
    }
  }

  // Double data type not supported in synapse convert it to float value on host
  const bool is_double_dtype = (val_type == at::ScalarType::Double);
  auto scalar_val_type = is_double_dtype ? at::ScalarType::Float : val_type;

  // Check if Long data type can be supported or else convert it to int32
  const bool is_long_dtype_and_not_supported =
      (val_type == at::ScalarType::Long) && !common::IsInt64Supported();
  scalar_val_type =
      is_long_dtype_and_not_supported ? at::ScalarType::Int : val_type;

  void* host_ptr = nullptr;
  const auto& host_ptr_size = elementSize(scalar_val_type);
  auto& device =
      habana::HPURegistrar::get_device(p_context_->device_id_).syn_device();
  auto status = device.get_host_memory().malloc(&host_ptr, host_ptr_size);
  HABANA_ASSERT(status == synSuccess, Logger::synStatusToStr(status));

  if (is_double_dtype) {
    auto lval = scalar_val.toFloat();
    memcpy(host_ptr, (const char*)&lval, host_ptr_size);
  } else if (is_long_dtype_and_not_supported) {
    auto lval = force_type.has_value() ? force_long_scalar_val.toInt()
                                       : scalar_val.toInt();
    memcpy(host_ptr, (const char*)&lval, host_ptr_size);
  } else {
    memcpy(host_ptr, val.data(), host_ptr_size);
  }

  PT_KERNEL_DEBUG(
      "constant host_ptr: ",
      reinterpret_cast<size_t>(host_ptr),
      " scalar value: ",
      Logger::_str_wrapper(scalar_val),
      " size: ",
      host_ptr_size,
      " org data_type: ",
      scalar_val.type(),
      " data_type: ",
      scalar_val_type,
      " is_force_dtype : ",
      force_type.has_value());

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

  return const_syn_tensor;
}

synapse_helpers::tensor& habana::HabanaOperator::AllocateSeed(
    synapse_helpers::graph& graph,
    const at::Tensor& seed_tensor) {
  p_context_->syn_seed_ =
      habana_helpers::create_tensor(seed_tensor, graph, true, false);
  return p_context_->syn_seed_.value();
}

habana::RegisterKernel& habana::KernelRegistry() {
  static habana::RegisterKernel* Registry = new habana::RegisterKernel();
  return *Registry;
}

habana::HabanaOperator::~HabanaOperator() = default;

void habana::HabanaOperator::dump(
    torch::jit::Node* node,
    const at::Stack& stack) {
  PT_OP_DEBUG([&]() {
    auto stack_printer = [](const at::Stack& stack) {
      std::ostringstream ss;
      std::string sep = "";
      for (const auto& s : stack) {
        ss << sep;
        sep = ", ";
        if (s.isTensor())
          ss << habana::to_string(s.toTensor());
        else if (s.isTensorList())
          ss << habana::to_string(s.toTensorVector());
        else
          ss << habana::to_string(s);
      }
      ss << "\n";
      return ss.str();
    };
    auto syn_tensor_printer =
        [](const std::deque<synapse_helpers::tensor_or_ref>& tensors) {
          std::ostringstream ss;
          for (const synapse_helpers::tensor& t : tensors) {
            ss << absl::StrFormat("\n%8s%s", "", t.DebugString(/*indent=*/5));
          }
          ss << "\n";
          return ss.str();
        };

    auto pt_tensor_printer = [](const std::vector<at::Tensor>& tensors) {
      std::ostringstream ss;
      std::string sep = "";
      for (const auto& t : tensors) {
        ss << sep << habana::to_string(t);
        sep = ", ";
      }
      ss << "\n";
      return ss.str();
    };

    auto node_io_printer = [](at::ArrayRef<torch::jit::Value*> vals) {
      std::ostringstream ss;
      for (auto val : vals) {
        ss << " " << *val->type();
      }
      ss << "\n";
      return ss.str();
    };

    std::ostringstream ss;
    ss << "Op: " << node->schema().operator_name() << "\n";
    ss << "  Inputs:\n";
    ss << "    stack: " << stack_printer(stack);
    ss << "    pt_inputs: " << pt_tensor_printer(GetInputs());
    ss << "    syn_inputs:" << syn_tensor_printer(GetSynInputs());
    ss << "    node_inputs:" << node_io_printer(node->inputs());
    ss << "  Outputs:\n";
    ss << "    pt_outputs: " << pt_tensor_printer(GetOutputs());
    ss << "    syn_outputs:" << syn_tensor_printer(GetSynOutputs());
    ss << "    node_outputs:" << node_io_printer(node->outputs());
    return ss.str();
  }());
}

void habana::InferOutputMetaRetType::AddTensor(
    const TensorMetaData& data,
    std::vector<IdxTensorTuple>& v) {
  auto sif_tensor_id_ = habana::ShapeInference::ReadAndIncrementSifTensorId();
  auto tensor = habana::nonPersistentTensor(
      data.sizes, data.strides, data.mf, scalarTypeToTypeMeta(data.dtype));

  v.emplace_back(std::make_tuple(sif_tensor_id_, tensor));
}

void habana::InferOutputMetaRetType::AddOutputTensor(
    const TensorMetaData& data) {
  AddTensor(data, output_tensors_);
}

void habana::InferOutputMetaRetType::AddIntermediateTensor(
    const TensorMetaData& data) {
  auto& output =
      kernel_outputs_.emplace_back(std::make_shared<InferOutputMetaRetType>());
  output->AddOutputTensor(data);
}

void habana::InferOutputMetaRetType::AddShapeTensor(
    const TensorMetaData& data) {
  AddTensor(data, shape_tensors_);
}

void habana::InferOutputMetaRetType::AddDupTensor(
    const habana::TensorMetaData& data) {
  AddTensor(data, dup_tensors_);
}

habana::InferOutputMetaRetType& habana::InferOutputMetaRetType::
    call_InferOutputMeta(HabanaOperatorPtr kernel, torch::jit::Stack& inputs) {
  HABANA_ASSERT(kernel.get() != nullptr, "kernel cannot be null");
  auto& output = kernel_outputs_.emplace_back(
      std::make_shared<habana::InferOutputMetaRetType>(
          kernel->InferOutputMeta(inputs)));
  return *output;
}

const habana::IdxTensorTuple& habana::InferOutputMetaRetType::GetOutputTensor(
    size_t index) const {
  HABANA_ASSERT(index < output_tensors_.size(), "index out of range");
  return output_tensors_.at(index);
}

const habana::IdxTensorTuple& habana::InferOutputMetaRetType::GetShapeTensor(
    size_t index) const {
  HABANA_ASSERT(index < shape_tensors_.size(), "index out of range");
  return shape_tensors_.at(index);
}

void habana::InferOutputMetaRetType::MoveToOutput(
    habana::IdxTensorTuple&& data) {
  output_tensors_.emplace_back(std::move(data));
}

void habana::InferOutputMetaRetType::RemoveOutput(size_t index) {
  HABANA_ASSERT(index < output_tensors_.size(), "index out of range");
  output_tensors_.erase(output_tensors_.begin() + index);
}

void habana::InferOutputMetaRetType::PushOutputTensorAtFront(
    IdxTensorTuple output_tensor) {
  output_tensors_.insert(output_tensors_.begin(), std::move(output_tensor));
}
