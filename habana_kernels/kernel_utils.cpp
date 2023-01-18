/******************************************************************************
 * Copyright (C) 2020-2022 Habana Labs, Ltd. an Intel Company
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
#include <torch/script.h>

#include <perf_lib_layer_params.h>
#include "backend/helpers/create_tensor.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/HPUStream.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_kernels/compare_kernels.h"
#include "habana_kernels/kernel_recipe_signature.h"
#include "hpu_ops/lazy_cast.h"
#include "kernel_utils.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"
#include "synapse_helpers/recipe.h"

using namespace torch;

namespace {
using CastMap =
    std::map<std::pair<c10::ScalarType, c10::ScalarType>, std::string>;

void insert_long_casts(CastMap& map) {
  if (GET_ENV_FLAG_NEW(PT_ENABLE_INT64_SUPPORT)) {
    map.insert(
        {{c10::ScalarType::Long, c10::ScalarType::Float}, "cast_i64_to_f32"});
    map.insert(
        {{c10::ScalarType::Float, c10::ScalarType::Long}, "cast_f32_to_i64"});
  } else {
    map.insert(
        {{c10::ScalarType::Long, c10::ScalarType::Float}, "cast_i32_to_f32"});
    map.insert(
        {{c10::ScalarType::Float, c10::ScalarType::Long}, "cast_f32_to_i32"});
  }
}
} // namespace

at::ScalarType habana_helpers::getInternalDtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::kLong: {
      if (GET_ENV_FLAG_NEW(PT_ENABLE_INT64_SUPPORT)) {
        return dtype;
      }
      return at::kInt;
    }
    case at::kDouble:
      return at::kFloat;
    case at::kBool:
      return at::kChar;
    default:
      return dtype;
  }
}

/**
 * @brief Prepare cast map for current platform
 **/
static auto get_platform_cast_map() {
  // initialize with g1
  CastMap cast_map{
      {{c10::ScalarType::Char, c10::ScalarType::Bool}, "cast_identity"},
      {{c10::ScalarType::Bool, c10::ScalarType::Char}, "cast_identity"},
      {{c10::ScalarType::Float, c10::ScalarType::Float}, "cast_identity"},
      {{c10::ScalarType::Int, c10::ScalarType::Int}, "cast_identity"},
      {{c10::ScalarType::Bool, c10::ScalarType::Float}, "cast_i8_to_f32"},
      {{c10::ScalarType::Char, c10::ScalarType::Float}, "cast_i8_to_f32"},
      {{c10::ScalarType::Float, c10::ScalarType::Bool}, "cast_f32_to_i8"},
      {{c10::ScalarType::Float, c10::ScalarType::Char}, "cast_f32_to_i8"},
      {{c10::ScalarType::Bool, c10::ScalarType::BFloat16}, "cast_i8_to_bf16"},
      {{c10::ScalarType::Char, c10::ScalarType::BFloat16}, "cast_i8_to_bf16"},
      {{c10::ScalarType::BFloat16, c10::ScalarType::Bool}, "cast_bf16_to_i8"},
      {{c10::ScalarType::BFloat16, c10::ScalarType::Char}, "cast_bf16_to_i8"},
      // TPC GUID doesn't support BF16->Int cast, hence using it
      // to realize it through a 2 level cast internally
      {{c10::ScalarType::BFloat16, c10::ScalarType::Int}, "cast_bf16_to_i32"},
      {{c10::ScalarType::Bool, c10::ScalarType::Int}, "cast_i8_to_i32"},
      {{c10::ScalarType::Char, c10::ScalarType::Int}, "cast_i8_to_i32"},
      {{c10::ScalarType::Int, c10::ScalarType::Bool}, "cast_i32_to_i8"},
      {{c10::ScalarType::Bool, c10::ScalarType::Short}, "cast_i8_to_i16"},
      {{c10::ScalarType::Short, c10::ScalarType::Bool}, "cast_i16_to_i8"},
      {{c10::ScalarType::Short, c10::ScalarType::Char}, "cast_i16_to_i8"},
      {{c10::ScalarType::Short, c10::ScalarType::Short}, "cast_identity"},
      {{c10::ScalarType::Short, c10::ScalarType::Int}, "cast_i16_to_i32"},
      {{c10::ScalarType::Int, c10::ScalarType::Char}, "cast_i32_to_i8"},
      {{c10::ScalarType::Int, c10::ScalarType::Short}, "cast_i32_to_i16"},
      {{c10::ScalarType::Int, c10::ScalarType::BFloat16}, "cast_i32_to_bf16"},
      {{c10::ScalarType::Int, c10::ScalarType::Float}, "cast_i32_to_f32"},
      {{c10::ScalarType::Float, c10::ScalarType::Int}, "cast_f32_to_i32"},
      {{c10::ScalarType::BFloat16, c10::ScalarType::Float}, "cast_bf16_to_f32"},
      {{c10::ScalarType::Float, c10::ScalarType::BFloat16}, "cast_f32_to_bf16"},
      {{c10::ScalarType::Byte, c10::ScalarType::Int}, "cast_u8_to_i32"},
      {{c10::ScalarType::Byte, c10::ScalarType::Bool}, "cast_u8_to_i8"},
      {{c10::ScalarType::Byte, c10::ScalarType::Float}, "cast_u8_to_f32"},
      // TPC GUID doesn't support Byte->BF16, hence using it
      // to realize it through a 2 level cast internally
      {{c10::ScalarType::Byte, c10::ScalarType::BFloat16}, "cast_u8_to_bf16"},
      {{c10::ScalarType::Int, c10::ScalarType::Byte}, "cast_i32_to_u8"},
      {{c10::ScalarType::Int, c10::ScalarType::Short}, "cast_i32_to_i16"},
  };

  insert_long_casts(cast_map);

  auto type{synapse_helpers::HPURegistrar::get_device().type()};
  switch (type) {
    case synDeviceGaudi2:
    case synDeviceGaudi3:
    case synDeviceGreco:
      // Half
      cast_map.insert(
          {{c10::ScalarType::Float, c10::ScalarType::Half}, "cast_f32_to_f16"});
      cast_map.insert(
          {{c10::ScalarType::Half, c10::ScalarType::Float}, "cast_f16_to_f32"});
      cast_map.insert(
          {{c10::ScalarType::BFloat16, c10::ScalarType::Half},
           "cast_bf16_to_f16"});
      cast_map.insert(
          {{c10::ScalarType::Half, c10::ScalarType::BFloat16},
           "cast_f16_to_bf16"});
      cast_map.insert(
          {{c10::ScalarType::Short, c10::ScalarType::Half}, "cast_i16_to_f16"});
      cast_map.insert(
          {{c10::ScalarType::Half, c10::ScalarType::Short}, "cast_f16_to_i16"});
      cast_map.insert(
          {{c10::ScalarType::Int, c10::ScalarType::Half}, "cast_i32_to_f16"});
      cast_map.insert(
          {{c10::ScalarType::Half, c10::ScalarType::Int}, "cast_f16_to_i32"});
      cast_map.insert(
          {{c10::ScalarType::Bool, c10::ScalarType::Half}, "cast_i8_to_f16"});
      cast_map.insert(
          {{c10::ScalarType::Char, c10::ScalarType::Half}, "cast_i8_to_f16"});
      cast_map.insert(
          {{c10::ScalarType::Half, c10::ScalarType::Bool}, "cast_f16_to_i8"});
      cast_map.insert(
          {{c10::ScalarType::Half, c10::ScalarType::Char}, "cast_f16_to_i8"});
      break;
    default:
      break;
  }

#if IS_PYTORCH_FORK_AT_LEAST(1, 0)
  if (type == synDeviceGaudi2 || type == synDeviceGaudi3) {
    // fp8r152
    cast_map.insert(
        {{c10::ScalarType::Float, c10::ScalarType::Fp8r152}, "cast_f32_to_f8"});
    cast_map.insert(
        {{c10::ScalarType::BFloat16, c10::ScalarType::Fp8r152},
         "cast_bf16_to_f8"});
    cast_map.insert(
        {{c10::ScalarType::Fp8r152, c10::ScalarType::Float}, "cast_f8_to_f32"});
    cast_map.insert(
        {{c10::ScalarType::Fp8r152, c10::ScalarType::BFloat16},
         "cast_f8_to_bf16"});
  }
#endif
  return cast_map;
}

std::optional<std::string> habana_helpers::direct_cast_guid(
    std::pair<c10::ScalarType, c10::ScalarType> type_key) {
  if (type_key.first == type_key.second)
    return "cast_identity";
  static auto cast_map{get_platform_cast_map()};
  auto iter = cast_map.find(type_key);
  if (iter != cast_map.end()) {
    return iter->second;
  }
  return {};
}

CastF32RoundMode_t habana_helpers::get_cast_rounding_mode(
    c10::ScalarType dst_dtype,
    const bool stochastic_rounding_override) {
#if IS_PYTORCH_FORK_AT_LEAST(1, 0)
  if ((stochastic_rounding_override ||
       GET_ENV_FLAG_NEW(PT_ENABLE_FP8_CAST_STOCHASTIC_ROUNDING)) &&
      dst_dtype == at::kFp8r152) {
    return CAST_ROUND_SR;
  }
#else
  (void)stochastic_rounding_override;
#endif

  if (c10::isIntegralType(dst_dtype, true)) {
    return CAST_ROUND_ZERO;
  }

  return CAST_ROUND_HALF_NE;
}

bool habana_helpers::isLongTypeSupported(const std::string& guid) {
  // Notice: We have no way of checking at runtime which kernels are supported
  // in i64 version, so the list has to be hardcoded here.
  // This is a temporary solution, in the future Synapse will allow us to call
  // i64 version for all kernels and will add i64->i32 cast when needed.
  std::unordered_set<std::string> supported_guids{"cast_"};
  return supported_guids.find(guid) != supported_guids.end();
}

/** @brief For OPs with two input arguments (e.g. binary, compare), we may get
 *input arguments with different dtypes. For such cases, this function
 *determines which input argument can be promoted to larger dtype. This function
 *takes IValue stack of input arguments as input and returns the position of
 *input argument to be promoted alongwith the dtype to which this argument needs
 *to be promoted.
 **/
void habana_helpers::type_promotion_for_two_tensor_inputs(
    std::vector<at::IValue>& inputs,
    int& position_of_promoted_tensor,
    c10::ScalarType& compute_dtype,
    c10::ScalarType& dst_dtype) {
  if (inputs[0].isTensor() && inputs[1].isTensor()) {
    auto tensor1 = inputs[0].toTensor();
    auto tensor2 = inputs[1].toTensor();
    if ((tensor1.device().type() != c10::DeviceType::HPU) ||
        (tensor2.device().type() != c10::DeviceType::HPU)) {
      // Early return if one of the tensors is not on Habana device
      // in such cases we will not try type promotion.
      return;
    }
    auto dtype_helper =
        habana_helpers::DTypeHelper::binary_op_with_type_promotion(
            inputs, c10::nullopt, false);

    compute_dtype = dst_dtype = dtype_helper.get_result_dtype();

    // Temporary W/A. The result dtype is converted from double to float and
    // from int64 to int32.
    auto type1 = getInternalDtype(tensor1.scalar_type());
    auto type2 = getInternalDtype(tensor2.scalar_type());

    compute_dtype = getInternalDtype(compute_dtype);

    // pos = position of tensor to be promoted (smaller dtype)
    if (type1 != compute_dtype) {
      position_of_promoted_tensor = 0;
    } else if (type2 != compute_dtype) {
      position_of_promoted_tensor = 1;
    }
  }
}

void habana_helpers::type_promotion_for_two_tensor_inputs(
    std::vector<at::IValue>& inputs,
    int& position_of_promoted_tensor,
    c10::ScalarType& compute_dtype) {
  c10::ScalarType dst_dtype = c10::ScalarType::Undefined;
  return type_promotion_for_two_tensor_inputs(
      inputs, position_of_promoted_tensor, compute_dtype, dst_dtype);
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
    } else if (sz1[i] == 0 || sz2[i] == 0) {
      // sizes do not match, but one of the input sizes is 0 => output size on
      // this dim will also be 0
      out_size.push_back(0);
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

std::string habana_helpers::unique_recipe_name_generator(
    std::string recipe_name) {
  static std::unordered_map<std::string, unsigned> map;
  return recipe_name + std::to_string(map[recipe_name]++);
}
namespace {
struct ResourceHolder {
  std::unique_ptr<synapse_helpers::device_ptr_lock> address_lock;
};
} // namespace

static void launchRecipe(
    const std::vector<void*>& input_buffers,
    const std::vector<void*>& output_buffers,
    std::vector<synapse_helpers::device_ptr> in_event_addr,
    std::vector<synapse_helpers::device_ptr> out_event_addr,
    std::vector<at::Tensor>& pt_inputs,
    const uint32_t device_id,
    std::shared_ptr<synapse_helpers::recipe>& recipe) {
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  auto& stream_handle =
      device.get_compute_stream(c10::hpu::getCurrentHPUStream());
  std::unique_ptr<synapse_helpers::device_ptr_lock> address_lock;
  if (device.IsStreamASyncEnabled()) {
    // wait for input DMA to complete before launching the compute.
    device.add_wait_events_on_stream(in_event_addr, stream_handle);

    auto& recipe_counter = device.get_active_recipe_counter();
    recipe_counter.increase();
    bool status = recipe->launch(
        input_buffers, output_buffers, address_lock, stream_handle);
    if (!status) {
      recipe_counter.decrease_and_notify();
      TORCH_CHECK(false, "syn launch failed");
    }
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
ns_CastKernel::Params CastOutOperator::synapse_cast_params_builder(
    c10::ScalarType dst_dtype,
    bool stochastic_rounding_override = false) {
  ns_CastKernel::Params params{};
  params.round_mode = stochastic_rounding_override
      ? CAST_ROUND_SR
      : habana_helpers::get_cast_rounding_mode(dst_dtype);
  return params;
}

ns_CastKernel::ParamsV2 CastOutOperator::synapse_cast_params_v2_builder(
    c10::ScalarType dst_dtype,
    bool stochastic_rounding_override = false,
    int seed = 0) {
  ns_CastKernel::ParamsV2 params{};
  params.round_mode = stochastic_rounding_override
      ? CAST_ROUND_SR
      : habana_helpers::get_cast_rounding_mode(dst_dtype);
  params.seed = seed;
  return params;
}

habana::OutputShapeInfRetType CastOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  auto castOp = make_operator<habana::LazyCast>(
      p_context_->device_id_, inputs[1].toScalarType());
  return castOp->ComputeOutputShape(inputs);
}

void CastOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  auto type = inputs[1].toScalarType();
  auto castOp = make_operator<habana::LazyCast>(p_context_->device_id_, type);
  castOp->SetSynapseInput(p_context_->syn_inputs_[0]);
  castOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);
  p_context_->syn_outputs_.emplace_back(std::move(castOp->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(castOp->GetOutputs()[0]));
}

habana::OutputShapeInfRetType CastOutOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  auto output = inputs[1].toTensor();
  habana::OutputShapeInfRetType out;
  out.AddDupTensor(habana::TensorMetaData(
      output.sizes().vec(),
      HabanaOperator::CalculateStrides(
          output.sizes(), output.suggest_memory_format()),
      output.scalar_type(),
      output.suggest_memory_format()));
  return out;
}

void CastOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for cast operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for cast operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for cast operator");

  static_cast<void>(output_metadata);
  auto self = inputs[0].toTensor();
  auto output = inputs[1].toTensor();

  ns_CastKernel::Params params =
      synapse_cast_params_builder(output.scalar_type());
  p_context_->params_.emplace<ns_CastKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[1], graph, output_metadata.at(0).external));
  p_context_->pt_outputs_.emplace_back(output);
  // Cast requires only 1 input popping second as it is output
  p_context_->syn_inputs_.pop_back();
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

habana::OutputShapeInfRetType ConstantOutOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  auto output = inputs[0].toTensor();
  auto tensor_meta_data = habana::TensorMetaData(
      output.sizes().vec(),
      HabanaOperator::CalculateStrides(
          output.sizes(), output.suggest_memory_format()),
      output.scalar_type(),
      output.suggest_memory_format());
  habana::OutputShapeInfRetType out;
  out.AddOutputTensor(tensor_meta_data);
  out.AddShapeTensor(tensor_meta_data);
  return out;
}

void ConstantOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  static_cast<void>(output_metadata);
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

  ns_ConstantKernel::Params params{};
  if (output.scalar_type() == c10::ScalarType::Int) {
    params.constant.i = value.to<int32_t>();
  } else {
    params.constant.f = value.to<float>();
  }

  p_context_->params_.emplace<ns_ConstantKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  if (output.dim() == 0) {
    SET_SIZE_STRIDE_1D(output);
  }

  HABANA_ASSERT(p_context_->syn_inputs_.size() == 1);
  synapse_helpers::tensor_or_ref& input_tensor = p_context_->syn_inputs_.back();
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          input_tensor, graph, output_metadata.at(0).external));
  p_context_->pt_outputs_.emplace_back(output);
  // Adding a clear for inputs as constant kernel expects no inputs
  // AS we get inputs from PT kernel, graph mode creates a syn tensor anyway
  // It was observed if we let that syn tensor remain, the kernel gives wrong
  // outputs
  p_context_->syn_inputs_.clear();

  // Allocate Shape Tensor
  if (graph.is_dynamic_graph()) {
    AllocateSynapseShapeTensor(graph, output);
  }

  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

habana::OutputShapeInfRetType ConstantOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  auto input = inputs[0].toTensor();
  auto tensor_meta_data = habana::TensorMetaData(
      input.sizes().vec(),
      HabanaOperator::CalculateStrides(
          input.sizes(), input.suggest_memory_format()),
      input.scalar_type(),
      input.suggest_memory_format());
  habana::OutputShapeInfRetType out;
  out.AddOutputTensor(tensor_meta_data);
  out.AddShapeTensor(tensor_meta_data);
  return out;
}

void ConstantOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
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

  ns_ConstantKernel::Params params{};
  if (input.scalar_type() == c10::ScalarType::Int) {
    params.constant.i = value.to<int32_t>();
  } else {
    params.constant.f = value.to<float>();
  }

  p_context_->params_.emplace<ns_ConstantKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  if (input.dim() == 0) {
    SET_SIZE_STRIDE_1D(input);
  }

  auto output =
      habana_helpers::createPTTensor(input, output_metadata.at(0).persistent);
  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  // Adding a clear for inputs as constant kernel expects no inputs
  // AS we get inputs from PT kernel, graph mode creates a syn tensor anyway
  // It was observed if we let that syn tensor remain, the kernel gives wrong
  // outputs
  p_context_->syn_inputs_.clear();

  // Allocate Shape Tensor
  if (graph.is_dynamic_graph()) {
    AllocateSynapseShapeTensor(graph, output);
  }

  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

static const auto& KernelUtilsKernelRegistry = habana::KernelRegistry().add(
    "aten::ones_like",
    [](const int device_id, c10::ScalarType node_type) {
      return std::make_shared<OnesLikeOperator>(device_id, node_type);
    });
