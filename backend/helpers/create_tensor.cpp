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

#include "backend/helpers/create_tensor.h"
#include <string>
#include "backend/backend_meta.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/habana_device/tensor_builder.h"
#include "backend/helpers/runtime_config.h"
#include "backend/helpers/tensor_info.h"
#include "backend/kernel/hpu_shape_inference.h"
#include "backend/synapse_helpers/tcmalloc_helper.h"
#include "common/utils.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_helpers/logging.h"

namespace {
void handle_const_section_tensor(const at::Tensor& tensor) {
  if (habana_helpers::IsConstSectionSerialization()) {
    auto tmeta{habana::get_tensor_extra_meta(tensor)};
    if (tmeta->is_const_tensor()) {
      if (!tmeta->get_const_section_data_serializer()->isSerialized(
              tmeta->get_const_id())) {
        tmeta->get_const_section_data_serializer()->serialize(
            tmeta->get_host_ptr(),
            tmeta->get_host_size(),
            tmeta->get_const_id());
      } else {
        PT_CONST_SECTION_DEBUG(
            __func__,
            " const: ",
            tmeta->get_const_id(),
            "already serialized...");
      }
      auto& device = habana::HPURegistrar::get_device();
      device.get_host_memory().free(tmeta->get_host_ptr());
      tmeta->set_host_ptr(nullptr);
      tmeta->set_data_in_host_memory(false);
      // Call TcMalloc extension to release memory
      synapse_helpers::ReleaseFreeMemory();
    }
  }
}
} // namespace

namespace habana_helpers {

inline uint64_t get_syn_offset(const at::Tensor& t) {
  return !habana::is_ZST(t) ? (t.storage_offset() * t.itemsize()) : 0;
}

synapse_helpers::tensor create_tensor(
    const c10::IntArrayRef& shape,
    [[maybe_unused]] const c10::IntArrayRef& stride,
    synapse_helpers::graph& graph,
    bool persistent,
    bool external,
    int devid,
    const c10::ScalarType dtype,
    const std::string& name) {
  uint64_t tensor_id{synapse_helpers::INVALID_SYN_TENSOR_ID};
  // In case of dynamic graph update the name shape map
  if (graph.is_dynamic_graph() &&
      (graph.is_optim_output_sif_enabled() == false ||
       habana::ShapeInference::GetCurrentPass() !=
           habana::ShapeInfo::InferencePass::OUTPUT_SHAPE)) {
    tensor_id = habana::ShapeInference::UpdateShapeInfo(graph, shape.vec());
  }
  if (graph.is_dry_run()) {
    // For dry run mode, just create a placeholder tensor
    return synapse_helpers::tensor::create_placeholder(
        devid, shape.vec(), calculate_strides(shape.vec()), persistent, name);
  }

  std::vector<int64_t> min, max;
  if (graph.is_dynamic_graph()) {
    std::tie(min, max) = habana::ShapeInference::GetMinMaxShape(tensor_id);
  }

  if (min.size() && max.size() && (min != max)) {
    // if the min represents the max value and if the max represents the min
    // value, swap them during tensor creation
    if (min > max) {
      std::swap(min, max);
    }
    auto dynamic_shape = synapse_helpers::tensor::dynamic_shape_t{
        synapse_helpers::to_shape_t(min), synapse_helpers::to_shape_t(max)};
    // Create the max stride
    std::vector<int64_t> max_stride(max.size());
    max_stride[max.size() - 1] = 1;
    for (size_t d = max.size() - 1; d > 0; --d) {
      max_stride[d - 1] = max_stride[d] * max[d];
    }
    auto variant = synapse_helpers::tensor_builder(
                       max, max_stride, pytorch_to_synapse_type(dtype))
                       .mark_persistence(persistent)
                       .mark_external(external)
                       .with_dynamic_shape(dynamic_shape)
                       .build(
                           habana::HPURegistrar::get_device(devid).syn_device(),
                           graph.get_graph_handle());
    synapse_helpers::tensor syn_tensor =
        absl::get<synapse_helpers::tensor>(std::move(variant));
    syn_tensor.set_pt_info(shape.vec(), calculate_strides(stride.vec()));
    PT_DYNAMIC_SHAPE_DEBUG("create_tensor ", syn_tensor);
    return syn_tensor;
  }

  auto builder =
      synapse_helpers::tensor_builder(
          shape, calculate_strides(shape.vec()), pytorch_to_synapse_type(dtype))
          .mark_persistence(persistent)
          .mark_external(external)
          .with_is_shape_agnostic_on(graph.is_shape_agnostic_graph());
  if (!name.empty()) {
    builder.use_suffix(name);
  }
  auto variant = builder.build(
      habana::HPURegistrar::get_device(devid).syn_device(),
      graph.get_graph_handle());
  synapse_helpers::tensor syn_tensor =
      absl::get<synapse_helpers::tensor>(std::move(variant));
  syn_tensor.set_pt_info(shape.vec(), calculate_strides(stride.vec()));
  PT_DYNAMIC_SHAPE_DEBUG("create_tensor ", syn_tensor);
  return syn_tensor;
}

synapse_helpers::tensor create_tensor(
    const at::Tensor& tensor,
    synapse_helpers::graph& graph,
    bool persistent,
    bool external,
    const c10::optional<c10::ScalarType> dtype,
    const std::string& name,
    const std::string& inference_name) {
  PT_BRIDGE_DEBUG("[create_tensor-1] name: ", name);
  PT_BRIDGE_DEBUG("[create_tensor-1] inference_name: ", inference_name);
  uint64_t tensor_id{synapse_helpers::INVALID_SYN_TENSOR_ID};
  // In case of dynamic graph update the name shape map
  if (graph.is_dynamic_graph() &&
      (graph.is_optim_output_sif_enabled() == false ||
       habana::ShapeInference::GetCurrentPass() !=
           habana::ShapeInfo::InferencePass::OUTPUT_SHAPE)) {
    tensor_id =
        habana::ShapeInference::UpdateShapeInfo(graph, tensor.sizes().vec());
  }

  auto syn_dtype =
      pytorch_to_synapse_type(dtype.value_or(tensor.scalar_type()));
  auto tmeta{habana::get_tensor_extra_meta(tensor)};

  if (graph.is_dry_run()) {
    // For dry run mode, just create a placeholder tensor
    return synapse_helpers::tensor::create_placeholder(
        tensor.device().index(),
        tensor.sizes().vec(),
        calculate_strides(tensor.sizes().vec()),
        syn_dtype,
        persistent,
        name,
        DATA_TENSOR);
  }

  std::vector<int64_t> min, max;
  if (graph.is_dynamic_graph()) {
    std::tie(min, max) = habana::ShapeInference::GetMinMaxShape(tensor_id);
  }

  if (min.size() && max.size() && (min != max)) {
    // if the min represents the max value and if the max represents the min
    // value, swap them during tensor creation
    if (min > max) {
      std::swap(min, max);
    }
    auto dynamic_shape = synapse_helpers::tensor::dynamic_shape_t{
        synapse_helpers::to_shape_t(min), synapse_helpers::to_shape_t(max)};
    // Create the max stride
    std::vector<int64_t> max_stride(max.size());
    max_stride[max.size() - 1] = 1;
    for (size_t d = max.size() - 1; d > 0; --d) {
      max_stride[d - 1] = max_stride[d] * max[d];
    }

    auto [permutation, dont_allow_permutation] =
        get_tensor_memory_permutation(tensor);
    if (!permutation.empty()) {
      PT_LAZY_DEBUG(
          "Setting permutation to tensor: ",
          name,
          " id: ",
          tensor_id,
          " permutation: ",
          VecToString(permutation));
    }

    const uint64_t syn_offset = get_syn_offset(tensor);
    auto builder = synapse_helpers::tensor_builder(max, max_stride, syn_dtype)
                       .set_offset(syn_offset)
                       .mark_persistence(persistent)
                       .mark_external(external)
                       .with_dynamic_shape(dynamic_shape)
                       .with_permutation(permutation)
                       .with_dont_allow_permutation(dont_allow_permutation);
    if (!name.empty()) {
      builder.use_suffix(name);
    }

    if (auto exp_bias = tmeta->get_exp_bias()) {
      builder.with_quant_params(*exp_bias);
    }

    // To avoid special device index(-1) when it use
    // StorageLessWrapperTensorImpl to create tensor
    auto device_index =
        tensor.device().index() == -1 ? 0 : tensor.device().index();
    auto variant = builder.build(
        habana::HPURegistrar::get_device(device_index).syn_device(),
        graph.get_graph_handle());
    synapse_helpers::tensor syn_tensor =
        absl::get<synapse_helpers::tensor>(std::move(variant));
    syn_tensor.set_pt_info(
        tensor.sizes().vec(), calculate_strides(tensor.sizes().vec()));
    PT_DYNAMIC_SHAPE_DEBUG("create_tensor ", syn_tensor);
    return syn_tensor;
  }

  const uint64_t syn_offset = get_syn_offset(tensor);
  std::vector<int64_t> strides = calculate_strides(tensor.sizes().vec());
  synapse_helpers::layouts::MemoryPermutation permutation;
  bool dont_allow_permutation = false;
  std::tie(permutation, dont_allow_permutation) =
      get_tensor_memory_permutation(tensor);
  if (!permutation.empty()) {
    // If we pass permutation to Synapse tensor, we must pass empty strides
    strides.clear();
    PT_LAZY_DEBUG(
        "Setting permutation to tensor: ",
        name,
        " id: ",
        tensor_id,
        " permutation: ",
        VecToString(permutation));
  }

  bool const_section = false;
  void* host_ptr = nullptr;
  if (habana_helpers::IsInferenceMode() && tensor.has_storage()) {
    const_section = tmeta->is_const_tensor();
    if (const_section) {
      host_ptr = tmeta->get_host_ptr();
    }
  }

  auto builder =
      synapse_helpers::tensor_builder(tensor.sizes(), strides, syn_dtype)
          .set_offset(syn_offset)
          .mark_persistence(persistent)
          .mark_external(external)
          .with_permutation(permutation)
          .with_dont_allow_permutation(dont_allow_permutation)
          .mark_const_section(const_section, host_ptr)
          .with_is_shape_agnostic_on(graph.is_shape_agnostic_graph());
  // Add a check to validate the inference_range
  if (habana_helpers::IsInferenceMode()) {
    bool range_found_with_module_name = false;
    bool range_found = false;
    std::string module_name = std::string();
    PtTensorInferenceData::InferenceRangePair inference_range;
    if (name.size() > 0 &&
        inference_name.find("placeholder") == std::string::npos) {
      module_name =
          PtTensorInferenceData::get_instance().extract_key_name(name, "/");
      inference_range =
          PtTensorInferenceData::get_instance().GetInferenceTensorRange(
              module_name.c_str(), range_found_with_module_name);
    }
    if (!range_found_with_module_name) {
      inference_range =
          PtTensorInferenceData::get_instance().GetInferenceTensorRange(
              inference_name.c_str(), range_found);
      if (range_found && (name.size() > 0)) {
        PtTensorInferenceData::get_instance().duplicate_key(
            inference_name.c_str(), name.c_str());
      }
    }
    if (range_found_with_module_name || range_found) {
      builder = builder.with_inference_range(
          inference_range.first, inference_range.second);
    }
  }

  if (!name.empty()) {
    builder.use_suffix(name);
  }

  if (auto exp_bias = tmeta->get_exp_bias()) {
    builder.with_quant_params(*exp_bias);
  }

  // To avoid special device index(-1) when it use
  // StorageLessWrapperTensorImpl to create tensor
  auto device_index =
      tensor.device().index() == -1 ? 0 : tensor.device().index();
  auto variant = builder.build(
      habana::HPURegistrar::get_device(device_index).syn_device(),
      graph.get_graph_handle());
  if (absl::holds_alternative<synapse_helpers::synapse_error>(variant)) {
    auto error = absl::get<synapse_helpers::synapse_error>(variant);
    TORCH_HABANA_CHECK(error.status, error.error);
  }
  synapse_helpers::tensor syn_tensor =
      absl::get<synapse_helpers::tensor>(std::move(variant));
  syn_tensor.set_pt_info(
      tensor.sizes().vec(), calculate_strides(tensor.sizes().vec()));
  PT_DYNAMIC_SHAPE_DEBUG("create_tensor ", syn_tensor);
  handle_const_section_tensor(tensor);
  return syn_tensor;
}

synapse_helpers::tensor create_tensor(
    const at::Tensor& tensor,
    synapse_helpers::graph& graph,
    bool persistent,
    bool external,
    const synDataType synType,
    const std::string& name,
    const std::string& inference_name) {
  PT_BRIDGE_DEBUG("[create_tensor-2] name: ", name);
  PT_BRIDGE_DEBUG("[create_tensor-2] inference_name: ", inference_name);
  uint64_t tensor_id{synapse_helpers::INVALID_SYN_TENSOR_ID};
  auto tensor_shape = tensor.sizes().vec();

  // int4/uint4 data comes to the bridge packed into int32 tensors,
  // so the real tensor shape must have FCD dimension multiplied by 8
  if (synType == syn_type_int4 || synType == syn_type_uint4) {
    tensor_shape.back() *= 8;
  }

  // In case of dynamic graph update the name shape map
  if (graph.is_dynamic_graph() &&
      (graph.is_optim_output_sif_enabled() == false ||
       habana::ShapeInference::GetCurrentPass() !=
           habana::ShapeInfo::InferencePass::OUTPUT_SHAPE)) {
    tensor_id = habana::ShapeInference::UpdateShapeInfo(graph, tensor_shape);
  }

  if (graph.is_dry_run()) {
    // For dry run mode, just create a placeholder tensor
    return synapse_helpers::tensor::create_placeholder(
        tensor.device().index(),
        tensor_shape,
        calculate_strides(tensor_shape),
        synType,
        persistent,
        name);
  }

  std::vector<int64_t> min, max;
  if (graph.is_dynamic_graph()) {
    std::tie(min, max) = habana::ShapeInference::GetMinMaxShape(tensor_id);
  }

  if (min.size() && max.size() && (min != max)) {
    // if the min represents the max value and if the max represents the min
    // value, swap them during tensor creation
    if (min > max) {
      std::swap(min, max);
    }
    auto dynamic_shape = synapse_helpers::tensor::dynamic_shape_t{
        synapse_helpers::to_shape_t(min), synapse_helpers::to_shape_t(max)};
    // Create the max stride
    std::vector<int64_t> max_stride(max.size());
    max_stride[max.size() - 1] = 1;
    for (size_t d = max.size() - 1; d > 0; --d) {
      max_stride[d - 1] = max_stride[d] * max[d];
    }
    const uint64_t syn_offset = get_syn_offset(tensor);
    auto builder = synapse_helpers::tensor_builder(max, max_stride, synType)
                       .set_offset(syn_offset)
                       .mark_persistence(persistent)
                       .mark_external(external)
                       .with_dynamic_shape(dynamic_shape);
    if (!name.empty()) {
      builder.use_suffix(name);
    }
    auto variant = builder.build(
        habana::HPURegistrar::get_device(tensor.device().index()).syn_device(),
        graph.get_graph_handle());
    synapse_helpers::tensor syn_tensor =
        absl::get<synapse_helpers::tensor>(std::move(variant));
    syn_tensor.set_pt_info(tensor_shape, calculate_strides(tensor_shape));
    PT_DYNAMIC_SHAPE_DEBUG("create_tensor ", syn_tensor);
    return syn_tensor;
  }

  std::vector<int64_t> strides = calculate_strides(tensor_shape);
  const uint64_t syn_offset = get_syn_offset(tensor);
  auto [permutation, dont_allow_permutation] =
      get_tensor_memory_permutation(tensor);
  if (!permutation.empty()) {
    // If we pass permutation to Synapse tensor, we must pass empty strides
    strides.clear();
    PT_LAZY_DEBUG(
        "Setting permutation to tensor: ",
        name,
        " id: ",
        tensor_id,
        " permutation: ",
        VecToString(permutation));
  }

  auto builder =
      synapse_helpers::tensor_builder(tensor_shape, strides, synType)
          .set_offset(syn_offset)
          .mark_persistence(persistent)
          .mark_external(external)
          .with_permutation(permutation)
          .with_dont_allow_permutation(dont_allow_permutation)
          .with_is_shape_agnostic_on(graph.is_shape_agnostic_graph());

  if (habana_helpers::IsInferenceMode()) {
    bool range_found_with_module_name = false;
    bool range_found = false;
    std::string module_name = std::string();
    PtTensorInferenceData::InferenceRangePair inference_range;
    if (name.size() > 0 &&
        inference_name.find("placeholder") == std::string::npos) {
      module_name =
          PtTensorInferenceData::get_instance().extract_key_name(name, "/");
      inference_range =
          PtTensorInferenceData::get_instance().GetInferenceTensorRange(
              module_name.c_str(), range_found_with_module_name);
    }
    if (!range_found_with_module_name) {
      inference_range =
          PtTensorInferenceData::get_instance().GetInferenceTensorRange(
              inference_name.c_str(), range_found);
      if (range_found && (name.size() > 0)) {
        PtTensorInferenceData::get_instance().duplicate_key(
            inference_name.c_str(), name.c_str());
      }
    }
    if (range_found_with_module_name || range_found) {
      builder = builder.with_inference_range(
          inference_range.first, inference_range.second);
    }
  }
  if (!name.empty()) {
    builder.use_suffix(name);
  }
  auto variant = builder.build(
      habana::HPURegistrar::get_device(tensor.device().index()).syn_device(),
      graph.get_graph_handle());
  synapse_helpers::tensor syn_tensor =
      absl::get<synapse_helpers::tensor>(std::move(variant));
  syn_tensor.set_pt_info(tensor_shape, calculate_strides(tensor_shape));
  PT_DYNAMIC_SHAPE_DEBUG("create_tensor ", syn_tensor);
  return syn_tensor;
}

synapse_helpers::tensor create_shape_tensor(
    const c10::IntArrayRef& input_shapes,
    synDeviceId syn_device,
    synapse_helpers::graph& graph,
    bool persistent,
    synTensorType shape_tensor_type,
    const std::string& name,
    void* host_ptr) {
  uint64_t tensor_id{synapse_helpers::INVALID_SYN_TENSOR_ID};
  // In case of dynamic graph update the name shape map
  if (graph.is_dynamic_graph()) {
    tensor_id = synapse_helpers::detail::tensor_name_generator::get_tensor_id();
    uint64_t shape_tensor_id =
        habana::ShapeInference::ReadAndIncrementShapeTensorId();
    if (graph.is_optim_output_sif_enabled() == true &&
        habana::ShapeInference::GetCurrentPass() ==
            habana::ShapeInfo::InferencePass::OUTPUT_SHAPE) {
      tensor_id = habana::ShapeInference::GetSTMappedTensorIdx(shape_tensor_id);
      PT_DYNAMIC_SHAPE_DEBUG(
          "OUTPUT PASS: ST_ID = ", shape_tensor_id, " TID = ", tensor_id);
      habana::ShapeInference::UpdateShapeInfo(
          graph, tensor_id, input_shapes.vec());
    } else {
      PT_DYNAMIC_SHAPE_DEBUG("ST_ID = ", shape_tensor_id, " TID = ", tensor_id);
      habana::ShapeInference::SaveSTAndTensorIdxMapping(
          shape_tensor_id, tensor_id);
      habana::ShapeInference::UpdateShapeInfo(graph, input_shapes.vec());
    }
  }

  if (graph.is_dry_run()) {
    // For dry run mode, just create a placeholder tensor
    return synapse_helpers::tensor::create_placeholder(
        syn_device,
        input_shapes.vec(),
        calculate_strides(input_shapes.vec()),
        persistent,
        name,
        shape_tensor_type);
  }

  std::vector<int64_t> min, max;
  if (graph.is_dynamic_graph()) {
    std::tie(min, max) = habana::ShapeInference::GetMinMaxShape(tensor_id);
  }

  if (min.size() && max.size() && (min != max)) {
    // if the min represents the max value and if the max represents the min
    // value, swap them during tensor creation
    if (min > max) {
      std::swap(min, max);
    }
    auto dynamic_shape = synapse_helpers::tensor::dynamic_shape_t{
        synapse_helpers::to_shape_t(min), synapse_helpers::to_shape_t(max)};

    auto builder = synapse_helpers::tensor_builder(
                       input_shapes, synDataType::syn_type_uint32)
                       .with_dynamic_shape(dynamic_shape);
    switch (shape_tensor_type) {
      case SHAPE_TENSOR:
        builder.mark_shape_tensor();
        break;
      case DEVICE_SHAPE_TENSOR:
        builder.mark_device_shape_tensor();
        builder.mark_persistence(persistent);
        break;
      case HOST_TO_DEVICE_TENSOR:
        builder.mark_host_to_device_tensor(host_ptr);
        break;
      default:
        HABANA_ASSERT(0 && "Invalid shape_tensor_type");
        break;
    }
    auto variant = builder.build(
        habana::HPURegistrar::get_device(syn_device).syn_device(),
        graph.get_graph_handle());
    synapse_helpers::tensor syn_tensor =
        absl::get<synapse_helpers::tensor>(std::move(variant));
    // GC requires strides to be 0 for shape tensors though this should not
    // affect our tensor shape patching.
    syn_tensor.set_pt_info(
        input_shapes.vec(), calculate_strides(input_shapes.vec()));
    PT_DYNAMIC_SHAPE_DEBUG("create_shape_tensor ", syn_tensor);
    return syn_tensor;
  }
  uint64_t syn_offset = 0;
  auto builder = synapse_helpers::tensor_builder(
                     input_shapes, synDataType::syn_type_uint32)
                     .set_offset(syn_offset);
  switch (shape_tensor_type) {
    case SHAPE_TENSOR:
      builder.mark_shape_tensor();
      break;
    case DEVICE_SHAPE_TENSOR:
      builder.mark_device_shape_tensor();
      builder.mark_persistence(persistent);
      break;
    case HOST_TO_DEVICE_TENSOR:
      builder.mark_host_to_device_tensor(host_ptr);
      break;
    default:
      HABANA_ASSERT(0 && "Invalid shape_tensor_type");
      break;
  }
  if (!name.empty()) {
    builder.use_suffix(name);
  }
  auto variant = builder.build(
      habana::HPURegistrar::get_device(syn_device).syn_device(),
      graph.get_graph_handle());
  synapse_helpers::tensor syn_tensor =
      absl::get<synapse_helpers::tensor>(std::move(variant));
  syn_tensor.set_pt_info(
      input_shapes.vec(), calculate_strides(input_shapes.vec()));
  PT_DYNAMIC_SHAPE_DEBUG("create_shape_tensor ", syn_tensor);
  return syn_tensor;
}

synapse_helpers::tensor create_shape_tensor(
    const at::Tensor& tensor,
    synapse_helpers::graph& graph,
    bool persistent,
    synTensorType shape_tensor_type,
    const std::string& name,
    void* host_ptr) {
  uint64_t tensor_id{synapse_helpers::INVALID_SYN_TENSOR_ID};
  // In case of dynamic graph update the name shape map
  if (graph.is_dynamic_graph()) {
    tensor_id = synapse_helpers::detail::tensor_name_generator::get_tensor_id();
    uint64_t shape_tensor_id =
        habana::ShapeInference::ReadAndIncrementShapeTensorId();
    if (graph.is_optim_output_sif_enabled() == true &&
        habana::ShapeInference::GetCurrentPass() ==
            habana::ShapeInfo::InferencePass::OUTPUT_SHAPE) {
      tensor_id = habana::ShapeInference::GetSTMappedTensorIdx(shape_tensor_id);
      PT_DYNAMIC_SHAPE_DEBUG(
          "OUTPUT PASS: ST_ID = ", shape_tensor_id, " TID = ", tensor_id);
      habana::ShapeInference::UpdateShapeInfo(
          graph, tensor_id, tensor.sizes().vec());
    } else {
      PT_DYNAMIC_SHAPE_DEBUG("ST_ID = ", shape_tensor_id, " TID = ", tensor_id);
      habana::ShapeInference::SaveSTAndTensorIdxMapping(
          shape_tensor_id, tensor_id);
      habana::ShapeInference::UpdateShapeInfo(graph, tensor.sizes().vec());
    }
  }

  if (graph.is_dry_run()) {
    // For dry run mode, just create a placeholder tensor
    return synapse_helpers::tensor::create_placeholder(
        tensor.device().index(),
        tensor.sizes().vec(),
        calculate_strides(tensor.sizes().vec()),
        persistent,
        name,
        shape_tensor_type);
  }

  std::vector<int64_t> min, max;
  if (graph.is_dynamic_graph()) {
    std::tie(min, max) = habana::ShapeInference::GetMinMaxShape(tensor_id);
  }

  if (min.size() && max.size() && (min != max)) {
    // if the min represents the max value and if the max represents the min
    // value, swap them during tensor creation
    if (min > max) {
      std::swap(min, max);
    }
    auto dynamic_shape = synapse_helpers::tensor::dynamic_shape_t{
        synapse_helpers::to_shape_t(min), synapse_helpers::to_shape_t(max)};
    auto builder = synapse_helpers::tensor_builder(
                       tensor.sizes(), synDataType::syn_type_uint32)
                       .with_dynamic_shape(dynamic_shape);
    switch (shape_tensor_type) {
      case SHAPE_TENSOR:
        builder.mark_shape_tensor();
        break;
      case DEVICE_SHAPE_TENSOR:
        builder.mark_device_shape_tensor();
        builder.mark_persistence(persistent);
        break;
      case HOST_TO_DEVICE_TENSOR:
        builder.mark_host_to_device_tensor(host_ptr);
        break;
      default:
        HABANA_ASSERT(0 && "Invalid shape_tensor_type");
        break;
    }
    auto variant = builder.build(
        habana::HPURegistrar::get_device(tensor.device().index()).syn_device(),
        graph.get_graph_handle());
    synapse_helpers::tensor syn_tensor =
        absl::get<synapse_helpers::tensor>(std::move(variant));
    // GC requires strides to be 0 for shape tensors though this should not
    // affect our tensor shape patching.
    syn_tensor.set_pt_info(
        tensor.sizes().vec(), calculate_strides(tensor.sizes().vec()));
    PT_DYNAMIC_SHAPE_DEBUG("create_shape_tensor ", syn_tensor);
    return syn_tensor;
  }
  uint64_t syn_offset = tensor.storage_offset() * tensor.itemsize();
  auto builder = synapse_helpers::tensor_builder(
                     tensor.sizes(), synDataType::syn_type_uint32)
                     .set_offset(syn_offset);
  switch (shape_tensor_type) {
    case SHAPE_TENSOR:
      builder.mark_shape_tensor();
      break;
    case DEVICE_SHAPE_TENSOR:
      builder.mark_device_shape_tensor();
      builder.mark_persistence(persistent);
      break;
    case HOST_TO_DEVICE_TENSOR:
      builder.mark_host_to_device_tensor(host_ptr);
      break;
    default:
      HABANA_ASSERT(0 && "Invalid shape_tensor_type");
      break;
  }
  if (!name.empty()) {
    builder.use_suffix(name);
  }
  auto variant = builder.build(
      habana::HPURegistrar::get_device(tensor.device().index()).syn_device(),
      graph.get_graph_handle());
  synapse_helpers::tensor syn_tensor =
      absl::get<synapse_helpers::tensor>(std::move(variant));
  syn_tensor.set_pt_info(
      tensor.sizes().vec(), calculate_strides(tensor.sizes().vec()));
  PT_DYNAMIC_SHAPE_DEBUG("create_shape_tensor ", syn_tensor);
  return syn_tensor;
}

synapse_helpers::tensor create_const_tensor(
    const c10::IntArrayRef& shape,
    const c10::IntArrayRef& stride,
    synapse_helpers::graph& graph,
    bool persistent,
    int devid,
    const c10::ScalarType dtype,
    void* host_ptr,
    const uint64_t host_ptr_size,
    const std::string& name) {
  if (graph.is_dry_run()) {
    // For dry run mode, just create a placeholder tensor
    return synapse_helpers::tensor::create_placeholder(
        devid,
        shape.vec(),
        stride.vec(),
        pytorch_to_synapse_type(dtype),
        persistent,
        name);
  }
  auto builder =
      synapse_helpers::tensor_builder(
          shape, stride, pytorch_to_synapse_type(dtype))
          .mark_persistence(persistent)
          .mark_const(true, host_ptr, host_ptr_size)
          .with_is_shape_agnostic_on(graph.is_shape_agnostic_graph());
  if (!name.empty()) {
    builder.use_suffix(name);
  }
  auto variant = builder.build(
      habana::HPURegistrar::get_device(devid).syn_device(),
      graph.get_graph_handle());
  synapse_helpers::tensor syn_tensor =
      absl::get<synapse_helpers::tensor>(std::move(variant));
  syn_tensor.set_pt_info(shape.vec(), calculate_strides(stride.vec()));
  PT_DYNAMIC_SHAPE_DEBUG("create_const_tensor ", syn_tensor);
  return syn_tensor;
}

std::tuple<std::vector<synapse_helpers::tensor>, std::vector<synTensor>>
create_tensors(
    const std::vector<at::Tensor>& tensors,
    synapse_helpers::graph& graph,
    bool persistent,
    bool external) {
  return create_tensors(
      tensors,
      graph,
      std::vector<bool>(tensors.size(), persistent),
      std::vector<bool>(tensors.size(), external),
      std::vector<c10::optional<c10::ScalarType>>(
          tensors.size(), c10::nullopt));
}

std::tuple<std::vector<synapse_helpers::tensor>, std::vector<synTensor>>
create_tensors(
    const std::vector<at::Tensor>& tensors,
    synapse_helpers::graph& graph,
    const std::vector<bool>& persistents,
    const std::vector<bool>& externals,
    const std::vector<c10::optional<c10::ScalarType>> dtypes) {
  const auto num_tensors = tensors.size();
  TORCH_CHECK(persistents.size() == num_tensors);
  TORCH_CHECK(externals.size() == num_tensors);
  TORCH_CHECK(dtypes.size() == num_tensors);

  // tensor_helpers are used for tenor lifetime managment
  // syn_tensors are convinient to use with synapse API
  std::vector<synapse_helpers::tensor> tensor_helpers;
  std::vector<synTensor> syn_tensors;

  tensor_helpers.reserve(num_tensors);
  syn_tensors.reserve(num_tensors);

  for (size_t i = 0; i < num_tensors; ++i) {
    tensor_helpers.push_back(create_tensor(
        tensors[i],
        graph,
        persistents[i],
        externals[i],
        dtypes[i].value_or(tensors[i].scalar_type())));
    syn_tensors.push_back(tensor_helpers[i].get());
  }

  return {std::move(tensor_helpers), std::move(syn_tensors)};
}

synapse_helpers::tensor duplicate_tensor_in_memory_section(
    const synapse_helpers::tensor& tensor,
    synapse_helpers::graph& graph,
    bool external) {
  PT_BRIDGE_TRACE;

  if (graph.is_dynamic_graph() &&
      (graph.is_optim_output_sif_enabled() == false ||
       habana::ShapeInference::GetCurrentPass() !=
           habana::ShapeInfo::InferencePass::OUTPUT_SHAPE)) {
    habana::ShapeInference::UpdateShapeInfo(graph, tensor.pt_shape());
  }

  if (graph.is_dry_run()) {
    // In case of dry run mode, just create a place holder
    return synapse_helpers::tensor::create_placeholder(
        tensor.device_id(),
        tensor.pt_shape(),
        tensor.pt_strides(),
        tensor.type(),
        tensor.is_persistent());
  }

  if (external) {
    TORCH_CHECK(
        tensor.is_persistent(), "Cannot create non persistent external tensor");
  }

  auto builder = synapse_helpers::tensor_builder(
                     tensor.shape(), tensor.stride(), tensor.type())
                     .with_memory_section(tensor.memorysection())
                     .mark_persistence(tensor.is_persistent())
                     .mark_external(external)
                     .set_offset(tensor.get_offset())
                     .with_is_shape_agnostic_on(tensor.is_shape_agnostic());
  PT_LAZY_DEBUG(
      "Setting a duplicate tensor with permutation: ", tensor.permutation());
  builder.with_permutation(tensor.permutation());
  if (tensor.has_dynamic_shape()) {
    builder.with_dynamic_shape(tensor.dynamic_shape());
  }

  if (tensor.has_quant_params()) {
    const auto quant_params = tensor.get_quant_params();
    builder.with_quant_params(quant_params.expBias, quant_params.scale);
  }

  auto maybe_tensor = builder.build(
      habana::HPURegistrar::get_device(tensor.device_id()).syn_device(),
      tensor.graph());
  synapse_helpers::tensor syn_tensor =
      absl::get<synapse_helpers::tensor>(std::move(maybe_tensor));
  syn_tensor.set_pt_info(tensor.pt_shape(), tensor.pt_strides());
  PT_DYNAMIC_SHAPE_DEBUG("duplicate_tensor_in_memory_section ", syn_tensor);
  return syn_tensor;
}

synapse_helpers::tensor duplicate_tensor_in_memory_section_with_size(
    const synapse_helpers::tensor& tensor,
    synapse_helpers::graph& graph,
    std::vector<int64_t>& sizes,
    std::vector<int64_t>& strides,
    const uint64_t offset,
    bool external,
    synapse_helpers::layouts::MemoryPermutation permutation) {
  PT_BRIDGE_TRACE;

  if (graph.is_dynamic_graph() &&
      (graph.is_optim_output_sif_enabled() == false ||
       habana::ShapeInference::GetCurrentPass() !=
           habana::ShapeInfo::InferencePass::OUTPUT_SHAPE)) {
    habana::ShapeInference::UpdateShapeInfo(graph, sizes);
  }

  // Update to contiguous strides for duplicate synapse tensor
  strides = habana_helpers::calculate_strides(sizes);
  if (graph.is_dry_run()) {
    // For dry run mode, just create a placeholder tensor
    return synapse_helpers::tensor::create_placeholder(
        tensor.device_id(),
        sizes,
        strides,
        tensor.type(),
        tensor.is_persistent());
  }

  TORCH_CHECK(
      tensor.is_persistent(),
      "Why would you like to create another tensor in the same memory section for non persistent tensor?");

  auto builder = synapse_helpers::tensor_builder(sizes, strides, tensor.type())
                     .with_memory_section(tensor.memorysection())
                     .set_offset(offset)
                     .mark_persistence(tensor.is_persistent())
                     .mark_external(external)
                     .with_is_shape_agnostic_on(tensor.is_shape_agnostic());

  if (tensor.has_dynamic_shape()) {
    std::vector<int64_t> min, max;
    auto tensor_id =
        synapse_helpers::detail::tensor_name_generator::get_tensor_id();
    std::tie(min, max) = habana::ShapeInference::GetMinMaxShape(tensor_id);
    if (min.size() && max.size() && (min != max)) {
      auto dynamic_shape = synapse_helpers::tensor::dynamic_shape_t{
          synapse_helpers::to_shape_t(min), synapse_helpers::to_shape_t(max)};
      builder.with_dynamic_shape(dynamic_shape);
    }
  }
  PT_LAZY_DEBUG("Setting a duplicate tensor with permutation: ", permutation);
  builder.with_permutation(permutation);

  if (tensor.has_quant_params()) {
    const auto quant_params = tensor.get_quant_params();
    builder.with_quant_params(quant_params.expBias, quant_params.scale);
  }

  auto maybe_tensor = builder.build(
      habana::HPURegistrar::get_device(tensor.device_id()).syn_device(),
      tensor.graph());
  synapse_helpers::tensor syn_tensor =
      absl::get<synapse_helpers::tensor>(std::move(maybe_tensor));
  syn_tensor.set_pt_info(sizes, strides);
  PT_DYNAMIC_SHAPE_DEBUG(
      "duplicate_tensor_in_memory_section_with_size ", syn_tensor);
  return syn_tensor;
}

std::vector<std::string> names(
    const std::vector<synapse_helpers::tensor>& vec) {
  std::vector<std::string> names;
  names.reserve(vec.size());

  std::transform(
      vec.begin(), vec.end(), std::back_inserter(names), [](auto& tensor) {
        return tensor.name();
      });

  return names;
}

std::vector<std::string> names(
    const std::vector<synapse_helpers::tensor_or_ref>& vec) {
  std::vector<std::string> names;
  names.reserve(vec.size());

  std::transform(
      vec.begin(),
      vec.end(),
      std::back_inserter(names),
      [](const synapse_helpers::tensor& tensor) { return tensor.name(); });

  return names;
}

std::vector<std::string> names(
    const std::deque<synapse_helpers::tensor_or_ref>& vec) {
  std::vector<std::string> names;
  names.reserve(vec.size());

  std::transform(
      vec.begin(),
      vec.end(),
      std::back_inserter(names),
      [](const synapse_helpers::tensor& tensor) { return tensor.name(); });

  return names;
}

namespace {
auto get_synapse_type_for_long() {
  if (common::IsInt64Supported()) {
    return synDataType::syn_type_int64;
  }
  return synDataType::syn_type_int32;
}
} // namespace

synDataType pytorch_to_synapse_type(const c10::ScalarType pt_type) {
  static const std::unordered_map<c10::ScalarType, synDataType> map {
    {c10::ScalarType::Byte, synDataType::syn_type_uint8},
        {c10::ScalarType::Char, synDataType::syn_type_int8},
        {c10::ScalarType::Short, synDataType::syn_type_int16},
        {c10::ScalarType::Int, synDataType::syn_type_int32},
        {c10::ScalarType::Long, get_synapse_type_for_long()},
        {c10::ScalarType::Float, synDataType::syn_type_float},
        {c10::ScalarType::Half, synDataType::syn_type_fp16},
        {c10::ScalarType::Double, synDataType::syn_type_float},
        {c10::ScalarType::Bool, synDataType::syn_type_int8},
        {c10::ScalarType::BFloat16, synDataType::syn_type_bf16},
#if HAVE_FP8_SUPPORT
        {c10::ScalarType::Float8_e5m2, synDataType::syn_type_fp8_152},
        {c10::ScalarType::Float8_e4m3fn, synDataType::syn_type_fp8_143},
#endif
  };

  auto result = map.find(pt_type);
  TORCH_CHECK(result != map.end(), "Unsupported pytorch type ", pt_type);

  return result->second;
}

c10::ScalarType synapse_to_pytorch_type(const synDataType type) {
  static const auto map = std::unordered_map<synDataType, c10::ScalarType>{
    {synDataType::syn_type_uint8, c10::ScalarType::Byte},
        {synDataType::syn_type_int8, c10::ScalarType::Char},
        {synDataType::syn_type_int16, c10::ScalarType::Short},
        {synDataType::syn_type_int32, c10::ScalarType::Int},
        {synDataType::syn_type_float, c10::ScalarType::Float},
        {synDataType::syn_type_fp16, c10::ScalarType::Half},
        {synDataType::syn_type_bf16, c10::ScalarType::BFloat16},
        {synDataType::syn_type_int64, c10::ScalarType::Long},
#if HAVE_FP8_SUPPORT
        {synDataType::syn_type_fp8_152, c10::ScalarType::Float8_e5m2},
        {synDataType::syn_type_fp8_143, c10::ScalarType::Float8_e4m3fn},
#endif
  };

  auto result = map.find(type);
  TORCH_CHECK(result != map.end(), "Unsupported synapse type ", type);

  return result->second;
}

synDataType pytorch_to_synapse_type(const c10::Scalar& s) {
  return habana_helpers::pytorch_to_synapse_type(
      habana_helpers::scalar_type(s));
}

c10::ScalarType scalar_type(const c10::Scalar& s) {
  c10::ScalarType type = c10::ScalarType::Undefined;

  if (s.isFloatingPoint()) {
    type = c10::ScalarType::Float;
  } else if (s.isIntegral(false)) {
    type = c10::ScalarType::Int;
  } else if (s.isBoolean()) {
    type = c10::ScalarType::Bool;
  } else {
    TORCH_CHECK(!s.isComplex(), "Habana doesn't support complex types");
    throw std::runtime_error("Unknown type");
  }

  return type;
}

std::tuple<synapse_helpers::layouts::MemoryPermutation, bool>
get_tensor_memory_permutation(const at::Tensor& tensor) {
  PT_BRIDGE_TRACE;
  if (!tensor.has_storage()) {
    PT_BRIDGE_DEBUG(
        "Getting permutations from storage-less tensor. Returning defaults..");
    return {
        habana::StorageExtraMeta().get_memory_permutation(),
        habana::StorageExtraMeta().get_dont_allow_permutation()};
  } else {
    auto smeta{habana::get_storage_extra_meta(tensor)};
    if (smeta) {
      return {
          smeta->get_memory_permutation(), smeta->get_dont_allow_permutation()};
    } else {
      return {
          habana::StorageExtraMeta().get_memory_permutation(),
          habana::StorageExtraMeta().get_dont_allow_permutation()};
    }
  }
}

void set_tensor_memory_permutations(
    const at::Tensor& tensor,
    const synapse_helpers::layouts::MemoryPermutation& permutation) {
  auto smeta{habana::get_storage_extra_meta(tensor)};
  if (!smeta && permutation.empty()) {
    PT_BRIDGE_DEBUG(
        "Trying to set empty permute on tensor without StorageExtraMeta, ignoring..");
    return;
  }
  HABANA_ASSERT(
      smeta,
      "Trying to set memory permutations ",
      VecToString(permutation),
      ", but no StorageExtraMeta available for tensor ",
      tensor.toString(),
      ", dims: ",
      tensor.sizes().vec());
  PT_BRIDGE_DEBUG(
      "Updating the PT storage meta address: ",
      smeta,
      " storage address : ",
      tensor.data_ptr(),
      " with permutation: ",
      VecToString(permutation),
      " old permutation was: ",
      VecToString(smeta->get_memory_permutation()));

  if (!permutation.empty() && permutation.size() != tensor.sizes().size()) {
    PT_BRIDGE_FATAL(
        "wrong permute size",
        "  permute_vec.size = ",
        permutation.size(),
        "  PT tensor shape.dims =",
        tensor.sizes().size(),
        " PT shape: ",
        VecToString(tensor.sizes().vec()));
  }

  smeta->set_memory_permutation(permutation);
}

void update_tensor_layout_and_permutation(
    const at::Tensor& pt_tensor,
    const PtTensorInfo& ti) {
  auto tmeta{habana::get_tensor_extra_meta(pt_tensor)};
  auto internal_lf = tmeta->get_tensor_layout();
  auto internal_lf_new = ti.getHbInternalLayoutFormat();
  if (internal_lf != internal_lf_new) {
    PT_BRIDGE_DEBUG(
        "For ",
        ti.get_ir_name(),
        " updating HbInternalTensorImpl layout from ",
        internal_lf,
        " to ",
        internal_lf_new);
    tmeta->set_tensor_layout(internal_lf_new);
  }
  PT_BRIDGE_DEBUG(
      "Setting synapse permutation as saved in the cache to the output tensor id: ",
      ti.get_tensor_id(),
      " permutation: ",
      VecToString(ti.getHbInternalPermute()));
  set_tensor_memory_permutations(pt_tensor, ti.getHbInternalPermute());
}

at::Tensor create_empty_tensor(const PtTensorInfo& ti) {
  auto pt_tensor = at::empty(ti.get_shape(), ti.get_topts(), ti.get_mf());
  update_tensor_layout_and_permutation(pt_tensor, ti);
  return pt_tensor;
}
} // namespace habana_helpers
