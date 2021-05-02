/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <torch/script.h>

#include <perf_lib_layer_params.h>
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_kernels/kernel_recipe_signature.h"
#include "kernel_utils.h"
#include "synapse_helpers/recipe.h"

using namespace torch;
/** @brief This data structure is used to encapsulate dtype promotion rules for
 *OPs with 2 inputs.
 * @param key: dtype tensor1, dtype tensor2
 * @param value: dtype promoted tensor
 **/
std::map<std::pair<c10::ScalarType, c10::ScalarType>, c10::ScalarType>
    habana_helpers::promote_dtype{
        {{c10::ScalarType::Char, c10::ScalarType::Int}, c10::ScalarType::Int},
        {{c10::ScalarType::Int, c10::ScalarType::Char}, c10::ScalarType::Int},
        {{c10::ScalarType::Byte, c10::ScalarType::Int}, c10::ScalarType::Int},
        {{c10::ScalarType::Int, c10::ScalarType::Byte}, c10::ScalarType::Int},
        {{c10::ScalarType::Float, c10::ScalarType::Int},
         c10::ScalarType::Float},
        {{c10::ScalarType::Int, c10::ScalarType::Float},
         c10::ScalarType::Float},
        {{c10::ScalarType::BFloat16, c10::ScalarType::Float},
         c10::ScalarType::Float},
        {{c10::ScalarType::Float, c10::ScalarType::BFloat16},
         c10::ScalarType::Float},
        {{c10::ScalarType::Byte, c10::ScalarType::Float},
         c10::ScalarType::Float},
        {{c10::ScalarType::Float, c10::ScalarType::Byte},
         c10::ScalarType::Float},
        {{c10::ScalarType::Float, c10::ScalarType::Char},
         c10::ScalarType::Float},
        {{c10::ScalarType::BFloat16, c10::ScalarType::Char},
         c10::ScalarType::BFloat16},
        {{c10::ScalarType::Char, c10::ScalarType::Float},
         c10::ScalarType::Float},
        {{c10::ScalarType::Char, c10::ScalarType::BFloat16},
         c10::ScalarType::BFloat16},
    };

/** @brief This data structure is used to map src & dst (for a cast) to
 *corresponding cast node guid.
 * @param key: dtype src, dtype dst
 * @param value: node guid
 **/
std::map<std::pair<c10::ScalarType, c10::ScalarType>, std::string>
    habana_helpers::cast_map{
        {{c10::ScalarType::Bool, c10::ScalarType::Float}, "cast_i8_to_f32"},
        {{c10::ScalarType::Char, c10::ScalarType::Float}, "cast_i8_to_f32"},
        {{c10::ScalarType::Float, c10::ScalarType::Bool}, "cast_f32_to_i8"},
        {{c10::ScalarType::Float, c10::ScalarType::Char}, "cast_f32_to_i8"},
        {{c10::ScalarType::Bool, c10::ScalarType::BFloat16}, "cast_i8_to_bf16"},
        {{c10::ScalarType::Char, c10::ScalarType::BFloat16}, "cast_i8_to_bf16"},
        {{c10::ScalarType::BFloat16, c10::ScalarType::Bool}, "cast_bf16_to_i8"},
        {{c10::ScalarType::BFloat16, c10::ScalarType::Char}, "cast_bf16_to_i8"},
        {{c10::ScalarType::Bool, c10::ScalarType::Int}, "cast_i8_to_i32"},
        {{c10::ScalarType::Char, c10::ScalarType::Int}, "cast_i8_to_i32"},
        {{c10::ScalarType::Int, c10::ScalarType::Bool}, "cast_i32_to_i8"},
        {{c10::ScalarType::Int, c10::ScalarType::Char}, "cast_i32_to_i8"},
        {{c10::ScalarType::Int, c10::ScalarType::Float}, "cast_i32_to_f32"},
        // c10::Long dtype is treated as Int for Synapse tensors,
        // therefore we are casting from i32 to f32
        {{c10::ScalarType::Long, c10::ScalarType::Float}, "cast_i32_to_f32"},
        {{c10::ScalarType::Float, c10::ScalarType::Int}, "cast_f32_to_i32"},
        // c10::Long dtype is treated as Int for Synapse tensors,
        // therefore we are casting to i32 from f32
        {{c10::ScalarType::Float, c10::ScalarType::Long}, "cast_f32_to_i32"},
        {{c10::ScalarType::BFloat16, c10::ScalarType::Float},
         "cast_bf16_to_f32"},
        {{c10::ScalarType::Float, c10::ScalarType::BFloat16},
         "cast_f32_to_bf16"},
        {{c10::ScalarType::Byte, c10::ScalarType::Int}, "cast_u8_to_i32"},
        {{c10::ScalarType::Int, c10::ScalarType::Byte}, "cast_i32_to_u8"},
        {{c10::ScalarType::Byte, c10::ScalarType::Float}, "cast_u8_to_f32"},
    };

/** @brief For OPs with two input arguments (e.g. binary, compare), we may get
 *input arguments with different dtypes. For such cases, this function
 *determines which input argument can be promoted to larger dtype. This function
 *takes IValue stack of input arguments as input and returns the position of
 *input argument to be promoted alongwith the dtype to which this argument needs
 *to be promoted.
 **/
void habana_helpers::type_promotion_for_two_tensor_inputs(
    std::vector<at::IValue>& inputs,
    int& pos,
    c10::ScalarType& dst_dtype) {
  if (inputs[0].isTensor() && inputs[1].isTensor()) {
    auto tensor1 = inputs[0].toTensor();
    auto tensor2 = inputs[1].toTensor();
    if ((tensor1.device().type() != c10::DeviceType::HABANA) ||
        (tensor2.device().type() != c10::DeviceType::HABANA)) {
      // Early return if one of the tensors is not on Habana device
      // in such cases we will not try type promotion.
      return;
    }
    auto type1 = (tensor1.scalar_type() == c10::ScalarType::Long)
        ? c10::ScalarType::Int
        : tensor1.scalar_type();
    auto type2 = (tensor2.scalar_type() == c10::ScalarType::Long)
        ? c10::ScalarType::Int
        : tensor2.scalar_type();
    // Generate key using input dtype(s)
    std::pair<ScalarType, ScalarType> type{type1, type2};
    // Check if we have this key to find the dtype to which smaller dtype
    // tensor should be promoted to
    auto iter = habana_helpers::promote_dtype.find(type);
    if (iter != habana_helpers::promote_dtype.end()) {
      dst_dtype = iter->second;
      // pos = position of tensor to be promoted (smaller dytpe)
      pos = (type.first == dst_dtype) ? 1 : 0;
    }
  }
}

/**
 * @brief This function computes the shape of output tensor resulting from a
 *binary operation. Shape is computed as per Pytorch broadcasting rules for such
 *operators.
 *https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
 **/
std::vector<int64_t> habana_helpers::compute_broadcast_shape(
    const Tensor& arg1,
    const Tensor& arg2) {
  std::vector<int64_t> out_size;
  auto sz1 = arg1.sizes().vec();
  auto sz2 = arg2.sizes().vec();
  // reverse sizes to start from FCD
  std::reverse(sz1.begin(), sz1.end());
  std::reverse(sz2.begin(), sz2.end());
  // compare sizes of input tensors along each dim starting from FCD
  for (auto i = 0; i < std::min(arg1.ndimension(), arg2.ndimension()); i++) {
    if (sz1[i] == sz2[i]) {
      // sizes match, add either input size to output size
      out_size.push_back(sz1[i]);
    } else if (sz1[i] == 1 || sz2[i] == 1) {
      // sizes do not match, but one of the input sizes is 1 => push other input
      // size to output size
      out_size.push_back(std::max(sz1[i], sz2[i]));
    } else {
      // sizes do not match and none of the input sizes is 1 => sizes
      // inconsistent for broadcast
      TORCH_CHECK(
          0,
          "Incompatible input shapes, broadcast not possible. Tensor1 Size: ",
          sz1,
          " Tensor2 Size: ",
          sz2);
    }
  }

  if (arg1.ndimension() > arg2.ndimension()) {
    // add remaining input1 sizes to output_size
    out_size.insert(out_size.end(), sz1.begin() + arg2.ndimension(), sz1.end());
  } else if (arg1.ndimension() < arg2.ndimension()) {
    // add remaining input2 sizes to output_size
    out_size.insert(out_size.end(), sz2.begin() + arg1.ndimension(), sz2.end());
  }

  // reverse output sizes to natural Pytorch order
  std::reverse(out_size.begin(), out_size.end());
  return out_size;
}

std::vector<synLaunchTensorInfo> habana_helpers::
    generate_syn_launch_tensor_info(
        const std::vector<std::string>& in_names,
        const std::vector<void*>& in_buffers,
        const std::vector<std::string>& out_names,
        const std::vector<void*>& out_buffers) {
  TORCH_CHECK(in_names.size() == in_buffers.size());
  TORCH_CHECK(out_names.size() == out_buffers.size());

  std::vector<synLaunchTensorInfo> syn_info;
  syn_info.reserve(in_names.size() + out_names.size());

  for (size_t i = 0; i < in_names.size(); ++i)
    syn_info.emplace_back(synLaunchTensorInfo{
        in_names[i].c_str(),
        reinterpret_cast<uint64_t>(in_buffers[i]),
        DATA_TENSOR,
        {0}});
  for (size_t i = 0; i < out_names.size(); ++i)
    syn_info.emplace_back(synLaunchTensorInfo{
        out_names[i].c_str(),
        reinterpret_cast<uint64_t>(out_buffers[i]),
        DATA_TENSOR,
        {0}});

  return syn_info;
}

std::string habana_helpers::unique_recipe_name_generator(
    std::string recipe_name) {
  static std::unordered_map<std::string, unsigned> map;
  return recipe_name + std::to_string(map[recipe_name]++);
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
  auto& stream_handle = device.get_compute_stream();
  if (device.IsStreamASyncEnabled()) {
    // wait for input DMA to complete before launching the compute.
    device.add_wait_events_on_stream(in_event_addr, stream_handle);

    auto& recipe_counter = device.get_active_recipe_counter();
    recipe_counter.increase();
    bool status = recipe->launch(input_buffers, output_buffers);
    if (!status) {
      recipe_counter.decrease_and_notify();
      TORCH_CHECK(false, "syn launch failed");
    }
    const auto& recipe_ptr = recipe->getRecipeHandle();
    // Get the reference to the tensor it is operating on to prevent
    // it from being deallocated while the operation is still in flight.
    // so use copy of pt_input in callback
    // regsiter an event on the compute
    device.register_producer_on_stream(
        std::move(out_event_addr),
        stream_handle,
        [pt_inputs, recipe_ptr, &recipe_counter]() {
          recipe_counter.decrease_and_notify();
          return;
        });
  } else {
    recipe->launch(input_buffers, output_buffers);
    TORCH_HABANA_CHECK(
        synStreamSynchronize(stream_handle), "synStreamSynchronize failed");
  }
}

void habana_helpers::compile_and_run(
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

void habana_helpers::execute_recipe(
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

size_t habana_helpers::getRecipeKey(
    std::string node,
    std::vector<c10::IValue> stack,
    bool inPlaceOp,
    bool outOp) {
  RecipeSignature rs(true, stack, {node}, inPlaceOp, outOp);
  return rs.hash();
}

/**
 * @brief CastKernel params structure
 */
ns_CastKernel::Params CastOutOperator::synapse_cast_params_builder() {
  ns_CastKernel::Params cast_params{};
  cast_params.round_mode = CAST_ROUND_HALF_NE;

  return cast_params;
}

void CastOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for cast operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for cast operator");

  auto self = inputs[0].toTensor();
  auto type = inputs[1].toScalarType();
  auto output = habana_helpers::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      type,
      is_output_persistent);

  ns_CastKernel::Params params = synapse_cast_params_builder();
  p_context_->params_.emplace<ns_CastKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void CastOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for cast operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for cast operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for cast operator");

  static_cast<void>(is_output_persistent);
  auto self = inputs[0].toTensor();
  auto output = inputs[1].toTensor();

  ns_CastKernel::Params params = synapse_cast_params_builder();
  p_context_->params_.emplace<ns_CastKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);
  p_context_->syn_outputs_.emplace_back(std::move(p_context_->syn_inputs_[1]));
  p_context_->pt_outputs_.emplace_back(output);
  // Cast requires only 1 input popping second as it is output
  p_context_->syn_inputs_.pop_back();
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void ConstantOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  static_cast<void>(is_output_persistent);
  TORCH_CHECK(
      inputs.size() >= 2,
      "Incorrect size of inputs expected for constant operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for constant operator");
  TORCH_CHECK(
      inputs[1].isScalar(),
      "Input arg2 expected to be scalar for constant operator");

  auto output = inputs[0].toTensor();
  auto value = inputs[1].toScalar();

  TORCH_CHECK(
      (output.scalar_type() == c10::ScalarType::BFloat16) ||
          (output.scalar_type() == c10::ScalarType::Int) ||
          (output.scalar_type() == c10::ScalarType::Char) ||
          (output.scalar_type() == c10::ScalarType::Bool) ||
          (output.scalar_type() == c10::ScalarType::Byte) ||
          (output.scalar_type() == c10::ScalarType::Float),
      "Unsupported dtype provided for ConstantOut kernel Input.scalar_type() = ",
      output.scalar_type());

  ns_ConstantKernel::Params params;
  if (output.scalar_type() == c10::ScalarType::Int) {
    params.constant.i = value.to<int32_t>();
  } else {
    params.constant.f = value.to<float>();
  }

  p_context_->params_.emplace<ns_ConstantKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  if (output.dim() == 0) {
    output.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  // Note that Constant TPC kernel does not need any tensor inputs
  // therefore we can move the input tensor(s) to corresponding
  // output tensors without any problems.
  HABANA_ASSERT(p_context_->syn_inputs_.size() == 1);
  synapse_helpers::tensor_or_ref& input_tensor = p_context_->syn_inputs_.back();
  p_context_->syn_outputs_.emplace_back(std::move(input_tensor));
  p_context_->pt_outputs_.emplace_back(output);
  // Adding a clear for inputs as constant kernel expects no inputs
  // AS we get inputs from PT kernel, graph mode creates a syn tensor anyway
  // It was observed if we let that syn tensor remain, the kernel gives wrong
  // outputs
  p_context_->syn_inputs_.clear();
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void ConstantOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() >= 2,
      "Incorrect size of inputs expected for constant operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for constant operator");
  TORCH_CHECK(
      inputs[1].isScalar(),
      "Input arg2 expected to be scalar for constant operator");

  auto input = inputs[0].toTensor();
  auto value = inputs[1].toScalar();

  TORCH_CHECK(
      (input.scalar_type() == c10::ScalarType::BFloat16) ||
          (input.scalar_type() == c10::ScalarType::Int) ||
          (input.scalar_type() == c10::ScalarType::Char) ||
          (input.scalar_type() == c10::ScalarType::Bool) ||
          (input.scalar_type() == c10::ScalarType::Byte) ||
          (input.scalar_type() == c10::ScalarType::Float),
      "Unsupported dtype provided for Constant kernel Input.scalar_type() = ",
      input.scalar_type());

  ns_ConstantKernel::Params params;
  if (input.scalar_type() == c10::ScalarType::Int) {
    params.constant.i = value.to<int32_t>();
  } else {
    params.constant.f = value.to<float>();
  }

  p_context_->params_.emplace<ns_ConstantKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  if (input.dim() == 0) {
    input.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto output = habana_helpers::createPTTensor(input, is_output_persistent);
  AllocateSynapseOutput(graph, output, is_output_persistent);
  // Adding a clear for inputs as constant kernel expects no inputs
  // AS we get inputs from PT kernel, graph mode creates a syn tensor anyway
  // It was observed if we let that syn tensor remain, the kernel gives wrong
  // outputs
  p_context_->syn_inputs_.clear();
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

static auto& KernelRegistry = habana::KernelRegistry().add(
    "aten::ones_like",
    [](const int device_id, c10::ScalarType node_type) {
      return std::make_shared<OnesLikeOperator>(device_id, node_type);
    });
