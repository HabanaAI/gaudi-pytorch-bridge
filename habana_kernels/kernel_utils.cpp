/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include <torch/script.h>

#include <perf_lib_layer_params.h>
#include "backend/create_pt_tensor.h"
#include "backend/habana_device/HPUStream.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/cast_sequence.h"
#include "backend/helpers/create_tensor.h"
#include "backend/synapse_helpers/device_helpers.h"
#include "backend/synapse_helpers/recipe.h"
#include "common/utils.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/compare_kernels.h"
#include "hpu_ops/lazy_cast.h"
#include "kernel_utils.h"

#include <string_view>

using namespace torch;

namespace {
using CastMap =
    std::map<std::pair<c10::ScalarType, c10::ScalarType>, std::string>;

void insert_long_casts(CastMap& map) {
  if (common::IsInt64Supported()) {
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
      if (common::IsInt64Supported()) {
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

  auto type{habana::HPUDeviceContext::get_device().type()};
  switch (type) {
    case synDeviceGaudi2:
    case synDeviceGaudi3:
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

  if (synapse_helpers::device_supports_fp8(type)) {
    // float8_e5m2
    cast_map.insert(
        {{c10::ScalarType::Float, c10::ScalarType::Float8_e5m2},
         "cast_f32_to_f8"});
    cast_map.insert(
        {{c10::ScalarType::BFloat16, c10::ScalarType::Float8_e5m2},
         "cast_bf16_to_f8"});
    cast_map.insert(
        {{c10::ScalarType::Float8_e5m2, c10::ScalarType::Float},
         "cast_f8_to_f32"});
    cast_map.insert(
        {{c10::ScalarType::Float8_e5m2, c10::ScalarType::BFloat16},
         "cast_f8_to_bf16"});
    // float8_e4m3fn
    cast_map.insert(
        {{c10::ScalarType::Float, c10::ScalarType::Float8_e4m3fn},
         "cast_f32_to_hf8"});
    cast_map.insert(
        {{c10::ScalarType::BFloat16, c10::ScalarType::Float8_e4m3fn},
         "cast_bf16_to_hf8"});
    cast_map.insert(
        {{c10::ScalarType::Float8_e4m3fn, c10::ScalarType::Float},
         "cast_hf8_to_f32"});
    cast_map.insert(
        {{c10::ScalarType::Float8_e4m3fn, c10::ScalarType::BFloat16},
         "cast_hf8_to_bf16"});
  }
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

bool habana_helpers::isLongTypeSupported(const std::string_view guid) {
  // Notice: We have no way of checking at runtime which kernels are supported
  // in i64 version, so the list has to be hardcoded here.
  // This is a temporary solution, in the future Synapse will allow us to call
  // i64 version for all kernels and will add i64->i32 cast when needed.
  using namespace std::literals;
  static const std::unordered_set<std::string_view> supported_guids{
      "cast_"sv, "random_uniform_fwd"sv};

  return supported_guids.find(guid) != supported_guids.end();
}

std::string_view habana_helpers::GetPrecisionString(
    const c10::ScalarType& dtype) {
  // Map datatype to TPC supported precision
  // If INT64 is not supported, then Long and Int point to i32 precision
  using namespace std::literals;
  static const std::string_view prefix = "cast_"sv;
  return synapse_helpers::graph::name_suffix_from_type(
      habana_helpers::pytorch_to_synapse_type(dtype),
      habana_helpers::isLongTypeSupported(prefix));
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

habana::InferOutputMetaRetType CastOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  auto castOp = make_operator<habana::LazyCast>(
      p_context_->device_id_, inputs[1].toScalarType());
  return castOp->InferOutputMeta(inputs);
}

void CastOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  auto type = inputs[1].toScalarType();
  std::shared_ptr<HabanaOperator> castOp =
      make_operator<habana::LazyCast>(p_context_->device_id_, type);
  castOp->SetSynapseInput(p_context_->syn_inputs_[0]);
  castOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);
  p_context_->syn_outputs_.emplace_back(std::move(castOp->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(castOp->GetOutputs()[0]));
}

habana::InferOutputMetaRetType CastOutOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  auto output = inputs[1].toTensor();
  habana::InferOutputMetaRetType out;
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

habana::InferOutputMetaRetType ConstantOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  auto input = inputs[0].toTensor();
  auto tensor_meta_data = habana::TensorMetaData(
      input.sizes().vec(),
      HabanaOperator::CalculateStrides(
          input.sizes(), input.suggest_memory_format()),
      input.scalar_type(),
      input.suggest_memory_format());
  habana::InferOutputMetaRetType out;
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

  // For lazy eager mode, Allocate constant synapse tensor
  // in case of non-persistent tensor of size {1}.
  const auto is_persistent = output_metadata.at(0).persistent;
  if (GetExecutionMode() == habana_helpers::HabanaFrontendTypes::EAGER &&
      !is_persistent && input.sizes().equals({1})) {
    auto const_syn_tensor = AllocateConstantSynapseTensor(graph, value);
    p_context_->syn_outputs_.emplace_back(std::move(const_syn_tensor));

    // non-persistent PT tensor created to maintain pt_outputs
    auto output = habana::createPTTensor(input, false);
    p_context_->pt_outputs_.emplace_back(output);
    return;
  }

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

  auto output = habana::createPTTensor(input, is_persistent);
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
