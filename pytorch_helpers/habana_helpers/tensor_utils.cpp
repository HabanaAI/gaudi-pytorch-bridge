/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <ATen/InferSize.h>
#include <perf_lib_layer_params.h>
#include <synapse_helpers/graph.h>
#include <algorithm>
#include <mutex>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_device/tensor_builder.h"
#include "habana_device/PinnedMemoryAllocator.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_lazy/hblazy/csrc/lazy_executor.h"
#include "synapse_helpers/util.h"

using namespace torch;

/*************************************************************************
 * @brief Generic helper function to cast tensors on HPU
 ************************************************************************/
at::Tensor habana_helpers::hpu_cast_tensor(
    const at::Tensor& Input,
    caffe2::TypeMeta type) {
  PT_KERNEL_BEGIN;

  std::string node_type;
  if (Input.dtype() == c10::ScalarType::Bool &&
      type == c10::ScalarType::Float) {
    node_type = "cast_i8_to_f32";
  } else if (
      Input.dtype() == c10::ScalarType::Char &&
      type == c10::ScalarType::Float) {
    node_type = "cast_i8_to_f32";
  } else if (
      Input.dtype() == c10::ScalarType::Int && type == c10::ScalarType::Float) {
    node_type = "cast_i32_to_f32";
  } else if (
      Input.dtype() == c10::ScalarType::BFloat16 &&
      type == c10::ScalarType::Float) {
    node_type = "cast_bf16_to_f32";
  } else if (
      type == c10::ScalarType::Bool &&
      Input.dtype() == c10::ScalarType::Float) {
    node_type = "cast_f32_to_i8";
  } else if (
      type == c10::ScalarType::Char &&
      Input.dtype() == c10::ScalarType::Float) {
    node_type = "cast_f32_to_i8";
  } else if (
      type == c10::ScalarType::Int && Input.dtype() == c10::ScalarType::Float) {
    node_type = "cast_f32_to_i32";
  } else if (
      type == c10::ScalarType::BFloat16 &&
      Input.dtype() == c10::ScalarType::Float) {
    node_type = "cast_f32_to_bf16";
  }

  int device_id = Input.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  CastOperator Op(device_id, node_type);
  std::vector<c10::IValue> stack = {IValue(Input),
                                    IValue(typeMetaToScalarType(type))};
  std::vector<at::Tensor> pt_inputs{Input};

  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto Output = at::empty(
        Input.sizes(),
        Input.options().dtype(type),
        Input.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs({Output});
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    // Allocate synapse inputs
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    // compile and execute the graph
    Op.Compile(graph);
  }

  PT_KERNEL_END;
  return Op.GetOutputs()[0];
}

/*************************************************************************
 * @brief This helper function casts a long tensor to int (on CPU)
 ************************************************************************/
at::Tensor habana_helpers::cast_tensor_to_integer(
    const at::Tensor& long_tensor) {
  // TODO Remove this cast on CPU when int64_t->int32 cast available on
  // HPU
  auto int_tensor = std::make_unique<at::Tensor>();
  if (long_tensor.scalar_type() == c10::ScalarType::Long) {
    *int_tensor = long_tensor.to("cpu")
                      .to(c10::ScalarType::Int)
                      .to(long_tensor.device(), c10::attr::non_blocking);
  } else {
    *int_tensor = long_tensor;
  }

  return *int_tensor;
}

at::Tensor habana_helpers::cast_tensor_to_long(const at::Tensor& int_tensor) {
  // TODO Remove this cast on CPU when int32->int64_t cast available on
  // HPU
  auto long_tensor = std::make_unique<at::Tensor>();
  if (int_tensor.scalar_type() == c10::ScalarType::Int) {
    *long_tensor = int_tensor.to("cpu")
                       .to(c10::ScalarType::Long)
                       .to(int_tensor.device(), c10::attr::non_blocking);
  } else {
    *long_tensor = int_tensor;
  }

  return *long_tensor;
}

at::Tensor habana_helpers::to_cpu(const at::Tensor& hpu_tensor) {
  if (hpu_tensor.defined()) {
    return hpu_tensor.to(at::DeviceType::CPU);
  }

  return hpu_tensor;
}

synDataType habana_helpers::pytorch_to_synapse_type(
    const c10::ScalarType pt_type) {
  static const std::unordered_map<c10::ScalarType, synDataType> map{
      {c10::ScalarType::Byte, synDataType::syn_type_uint8},
      {c10::ScalarType::Char, synDataType::syn_type_int8},
      {c10::ScalarType::Short, synDataType::syn_type_int16},
      {c10::ScalarType::Int, synDataType::syn_type_int32},
      {c10::ScalarType::Long, synDataType::syn_type_int32},
      {c10::ScalarType::Float, synDataType::syn_type_float},
      //   {c10::ScalarType::Double , synDataType::},
      {c10::ScalarType::Bool, synDataType::syn_type_int8},
      {c10::ScalarType::BFloat16, synDataType::syn_type_bf16},
  };

  auto result = map.find(pt_type);
  TORCH_CHECK(result != map.end(), "Unsupported pytorch type ", pt_type);

  return result->second;
}

synDataType pytorch_to_synapse_type(const c10::Scalar& s) {
  return habana_helpers::pytorch_to_synapse_type(
      habana_helpers::scalar_type(s));
}

c10::ScalarType habana_helpers::scalar_type(const c10::Scalar& s) {
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

at::Tensor habana_helpers::scalar_to_device_tensor(
    const at::Scalar& scalar,
    const at::Tensor& self,
    const unsigned num_dimensions) {
  auto options = self.options();
  TORCH_CHECK(
      options.device().type() == c10::DeviceType::HABANA,
      "Wrong device: ",
      options.device().type());
  auto output = at::empty(std::vector<int64_t>(num_dimensions, 1), options);

  auto self_scalar_type = self.scalar_type();
  if (self_scalar_type == c10::ScalarType::BFloat16) {
    auto val = scalar.to<at::BFloat16>();
    copy_scalar_to_device(&val, output, output.nbytes());
  } else if (self_scalar_type == c10::ScalarType::Float) {
    auto val = scalar.to<float>();
    copy_scalar_to_device(&val, output, output.nbytes());
  } else if (self_scalar_type == c10::ScalarType::Int) {
    auto val = scalar.to<int>();
    copy_scalar_to_device(&val, output, output.nbytes());
  } else {
    PT_KERNEL_FATAL("Unsupported data type used in binary op");
  }

  return output;
}

bool habana_helpers::alwaysAllocOnDevice() {
  static std::once_flag flag;
  static bool allocOnDevice;
  std::call_once(flag, [&]() {
    allocOnDevice = false;
    if (const auto envp = std::getenv("HABANA_USE_PERSISTENT_TENSOR")) {
      allocOnDevice = atoi(envp) == 1;
    }
  });
  return allocOnDevice;
}
at::Tensor habana_helpers::nonPersistentTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    const at::TensorOptions& options,
    at::optional<c10::MemoryFormat> optional_memory_format,
    at::optional<caffe2::TypeMeta> data_type) {
  auto t =
      at::detail::make_tensor<habana_helpers::StorageLessWrapperTensorImpl>(
          input, data_type);
  t.unsafeGetTensorImpl()->set_sizes_contiguous(
      (size.size() == 0) ? input.sizes() : size);

  if (optional_memory_format.has_value()) {
    t.unsafeGetTensorImpl()->empty_tensor_restride(
        optional_memory_format.value_or(MemoryFormat::Contiguous));
  } else {
    auto memory_format =
        input.options().memory_format_opt().value_or(MemoryFormat::Contiguous);
    t.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  }

  PT_SYNHELPER_DEBUG("Allocating non persistent tensor: size = ", size);
  return t;
}

at::Tensor habana_helpers::nonPersistentTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    at::IntArrayRef strides,
    const at::TensorOptions& options,
    at::optional<c10::MemoryFormat> optional_memory_format,
    at::optional<caffe2::TypeMeta> data_type) {
  auto t =
      at::detail::make_tensor<habana_helpers::StorageLessWrapperTensorImpl>(
          input, data_type);
  t.unsafeGetTensorImpl()->set_sizes_and_strides(
      (size.size() == 0) ? input.sizes() : size, strides);
  if (optional_memory_format.has_value()) {
    t.unsafeGetTensorImpl()->empty_tensor_restride(
        optional_memory_format.value_or(MemoryFormat::Contiguous));
  } else {
    auto memory_format =
        input.options().memory_format_opt().value_or(MemoryFormat::Contiguous);
    t.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  }

  PT_SYNHELPER_DEBUG("Allocating non persistent tensor: size = ", size);
  return t;
}

at::Tensor habana_helpers::createPTTensor(
    const at::Tensor& input,
    bool is_persistent) {
  at::Tensor t;

  if (is_persistent || alwaysAllocOnDevice()) {
    t = at::empty(
        input.sizes(), input.options(), input.suggest_memory_format());
  } else {
    t = habana_helpers::nonPersistentTensor(
        input, input.sizes(), input.options(), input.suggest_memory_format());
  }

  return t;
}

at::Tensor habana_helpers::createPTTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    const at::TensorOptions& options,
    bool is_persistent) {
  at::Tensor t;
  if (is_persistent || alwaysAllocOnDevice()) {
    t = at::empty(size, options, input.suggest_memory_format());
  } else {
    t = habana_helpers::nonPersistentTensor(
        input, size, options, input.suggest_memory_format());
  }

  return t;
}

at::Tensor habana_helpers::createPTTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    const at::TensorOptions& options,
    at::optional<c10::MemoryFormat> optional_memory_format,
    bool is_persistent) {
  at::Tensor t;
  if (is_persistent || alwaysAllocOnDevice()) {
    t = at::empty(
        size,
        options,
        optional_memory_format.value_or(MemoryFormat::Contiguous));
  } else {
    t = habana_helpers::nonPersistentTensor(
        input,
        size,
        options,
        optional_memory_format.value_or(MemoryFormat::Contiguous));
  }

  return t;
}

at::Tensor habana_helpers::createPTTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    at::IntArrayRef strides,
    const at::TensorOptions& options,
    at::optional<c10::MemoryFormat> optional_memory_format,
    bool is_persistent) {
  at::Tensor t;
  if (is_persistent || alwaysAllocOnDevice()) {
    t = at::empty_strided(size, strides, options);
  } else {
    t = habana_helpers::nonPersistentTensor(
        input,
        size,
        strides,
        options,
        optional_memory_format.value_or(MemoryFormat::Contiguous));
  }

  return t;
}

at::Tensor habana_helpers::createPTTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    const at::TensorOptions& options,
    at::optional<c10::MemoryFormat> optional_memory_format,
    c10::ScalarType data_type,
    bool is_persistent) {
  at::Tensor t;
  if (is_persistent || alwaysAllocOnDevice()) {
    t = at::empty(
        size,
        input.options().dtype(data_type),
        optional_memory_format.value_or(MemoryFormat::Contiguous));
  } else {
    t = habana_helpers::nonPersistentTensor(
        input,
        size,
        options,
        optional_memory_format.value_or(MemoryFormat::Contiguous),
        scalarTypeToTypeMeta(data_type));
  }

  return t;
}

/******************************************************************************
 * @brief helper function for copying data from device to host
 * @param[in] src - source tensor in device
 * @param[in] size - transfer data size in bytes
 * @param[out] dst_ptr - destination memory address in cpu
 *****************************************************************************/
void habana_helpers::copy_scalar_to_host(
    const at::Tensor& src,
    void* dst_ptr,
    uint32_t size) {
  std::atomic<bool> copyDone{false};
  bool is_pinned = at::habana::PinnedMemoryAllocator_is_pinned(src.data_ptr());
  auto syn_error =
      synapse_helpers::HPURegistrar::get_device(src.device().index())
          .copy_data_to_host(
              reinterpret_cast<synapse_helpers::device_ptr>(src.data_ptr()),
              dst_ptr,
              size,
              [&copyDone]() { copyDone = true; }, is_pinned);
  TORCH_CHECK(syn_error.status == 0, syn_error.error);

  // wait for copy completion
  while (!copyDone) {
    std::this_thread::yield();
  }
}

/******************************************************************************
 * @brief helper function for copying data from host to device
 * @param[in] src_ptr - source memory address in cpu
 * @param[in] size - transfer data size in bytes
 * @param[out] dst - destination tensor in device
 *****************************************************************************/
void habana_helpers::copy_scalar_to_device(
    void* src_ptr,
    const at::Tensor& dst,
    uint32_t size) {
  auto device_id = dst.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  if (device.IsStreamASyncEnabled()) {
    // keeps a reference to the tensor it is
    // operating on to prevent it from being deallocated while the
    // operation is still in flight.
    const at::Tensor dstRef = dst;
    auto syn_error = device.copy_data_to_device(
        src_ptr,
        reinterpret_cast<synapse_helpers::device_ptr>(dst.data_ptr()),
        size,
        [dstRef]() { return; });

  } else {
    std::atomic<bool> copyDone{false};
    auto syn_error = device.copy_data_to_device(
        src_ptr,
        reinterpret_cast<synapse_helpers::device_ptr>(dst.data_ptr()),
        size,
        [&copyDone]() { copyDone = true; });
    TORCH_CHECK(syn_error.status == 0, syn_error.error);

    // wait for copy completion
    while (!copyDone) {
      std::this_thread::yield();
    }
  }
}

synapse_helpers::tensor habana_helpers::create_tensor(
    const c10::IntArrayRef& shape,
    synGraphHandle graph,
    bool persistent,
    int devid,
    const c10::ScalarType dtype) {
  if (!std::getenv("PT_HPU_LAZY_LOWERING") && std::getenv("PT_HPU_LAZY_MODE")) {
    // Lazy mode shape inference call, just create a placeholder tensor
    return synapse_helpers::tensor::create_placeholder(devid);
  }

  auto variant =
      synapse_helpers::tensor_builder(shape, pytorch_to_synapse_type(dtype))
          .mark_persistence(persistent)
          .build(synapse_helpers::HPURegistrar::get_device(devid), graph);
  return absl::get<synapse_helpers::tensor>(std::move(variant));
}

synapse_helpers::tensor habana_helpers::create_tensor(
    const at::Tensor& tensor,
    const synGraphHandle graph,
    bool persistent,
    const c10::optional<c10::ScalarType> dtype) {
  if (!std::getenv("PT_HPU_LAZY_LOWERING") && std::getenv("PT_HPU_LAZY_MODE")) {
    // Lazy mode shape inference call, just create a placeholder tensor
    return synapse_helpers::tensor::create_placeholder(tensor.device().index());
  }
  auto variant =
      synapse_helpers::tensor_builder(
          tensor.sizes(),
          pytorch_to_synapse_type(dtype.value_or(tensor.scalar_type())))
          .mark_persistence(persistent)
          .build(
              synapse_helpers::HPURegistrar::get_device(
                  tensor.device().index()),
              graph);
  return absl::get<synapse_helpers::tensor>(std::move(variant));
}

std::tuple<std::vector<synapse_helpers::tensor>, std::vector<synTensor>>
habana_helpers::create_tensors(
    const std::vector<at::Tensor>& tensors,
    synGraphHandle graph,
    bool persistent) {
  return habana_helpers::create_tensors(
      tensors,
      graph,
      std::vector<bool>(tensors.size(), persistent),
      std::vector<c10::optional<c10::ScalarType>>(
          tensors.size(), c10::nullopt));
}

std::tuple<std::vector<synapse_helpers::tensor>, std::vector<synTensor>>
habana_helpers::create_tensors(
    const std::vector<at::Tensor>& tensors,
    synGraphHandle graph,
    const std::vector<bool> persistents,
    const std::vector<c10::optional<c10::ScalarType>> dtypes) {
  const auto num_tensors = tensors.size();
  TORCH_CHECK(persistents.size() == num_tensors);
  TORCH_CHECK(dtypes.size() == num_tensors);

  // tensor_helpers are used for tenor lifetime managment
  // syn_tensors are convinient to use with synapse API
  std::vector<synapse_helpers::tensor> tensor_helpers;
  std::vector<synTensor> syn_tensors;

  tensor_helpers.reserve(num_tensors);
  syn_tensors.reserve(num_tensors);

  for (size_t i = 0; i < num_tensors; ++i) {
    tensor_helpers.push_back(habana_helpers::create_tensor(
        tensors[i],
        graph,
        persistents[i],
        dtypes[i].value_or(tensors[i].scalar_type())));
    syn_tensors.push_back(tensor_helpers[i].get());
  }

  return {std::move(tensor_helpers), std::move(syn_tensors)};
}

synapse_helpers::tensor habana_helpers::duplicate_tensor_in_memory_section(
    const synapse_helpers::tensor& tensor) {
  if (!std::getenv("PT_HPU_LAZY_LOWERING") && std::getenv("PT_HPU_LAZY_MODE")) {
    // Lazy mode shape inference call, just create a placeholder tensor
    return synapse_helpers::tensor::create_placeholder(tensor.device_id());
  }

  TORCH_CHECK(
      tensor.is_persistent(),
      "Why would you like to create another tensor in the same memory section for non persistent tensor?");

  auto maybe_tensor =
      synapse_helpers::tensor_builder(tensor.shape(), tensor.type())
          .with_memory_section(tensor.memorysection())
          .mark_persistence(tensor.is_persistent())
          .build(
              synapse_helpers::HPURegistrar::get_device(tensor.device_id()),
              tensor.graph());
  return absl::get<synapse_helpers::tensor>(std::move(maybe_tensor));
}

std::vector<std::string> habana_helpers::names(
    const std::vector<synapse_helpers::tensor>& vec) {
  std::vector<std::string> names;
  names.reserve(vec.size());

  std::transform(
      vec.begin(), vec.end(), std::back_inserter(names), [](auto& tensor) {
        return tensor.tensor_name_;
      });

  return names;
}

std::vector<std::string> habana_helpers::names(
    const std::vector<synapse_helpers::tensor_or_ref>& vec) {
  std::vector<std::string> names;
  names.reserve(vec.size());

  std::transform(
      vec.begin(),
      vec.end(),
      std::back_inserter(names),
      [](const synapse_helpers::tensor& tensor) {
        return tensor.tensor_name_;
      });

  return names;
}

std::vector<std::string> habana_helpers::names(
    const std::deque<synapse_helpers::tensor_or_ref>& vec) {
  std::vector<std::string> names;
  names.reserve(vec.size());

  std::transform(
      vec.begin(),
      vec.end(),
      std::back_inserter(names),
      [](const synapse_helpers::tensor& tensor) {
        return tensor.tensor_name_;
      });

  return names;
}

std::string habana_helpers::name_suffix_from_type(
    const c10::ScalarType pt_type) {
  auto string_or_error = synapse_helpers::graph::name_suffix_from_type(
      pytorch_to_synapse_type(pt_type));
  if (absl::holds_alternative<synapse_helpers::synapse_error>(
          string_or_error)) {
    auto error = absl::get<synapse_helpers::synapse_error>(string_or_error);
    TORCH_HABANA_CHECK(error.status, error.error);
  }
  return absl::get<std::string>(string_or_error);
}

std::vector<void*> habana_helpers::extract_data_ptrs(
    const std::vector<const at::Tensor*>& vec) {
  std::vector<void*> ptrs;
  ptrs.reserve(vec.size());

  std::transform(
      vec.cbegin(),
      vec.cend(),
      std::back_inserter(ptrs),
      [](const auto& tensor) { return tensor->data_ptr(); });
  return ptrs;
};

std::vector<void*> habana_helpers::extract_data_ptrs(
    const std::vector<at::Tensor>& vec) {
  std::vector<void*> ptrs;
  ptrs.reserve(vec.size());

  std::transform(
      vec.cbegin(),
      vec.cend(),
      std::back_inserter(ptrs),
      [](const auto& tensor) { return tensor.data_ptr(); });
  return ptrs;
};

/******************************************************************************
 * @brief helper function for copying data from device to host
 * @param[in] src - source tensor in device
 * @param[in] size - transfer data size in bytes
 * @param[out] dst_ptr - destination memory address in cpu
 *****************************************************************************/
void habana_helpers::copy_data_to_host(
    const at::Tensor& src,
    const at::Tensor& dst,
    bool non_blocking) {
  size_t device_id = src.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  bool is_pinned = at::habana::PinnedMemoryAllocator_is_pinned(dst.data_ptr());
  if (non_blocking) {
    // keeps a reference to the tensor it is
    // operating on to prevent it from being deallocated while the
    // operation is still in flight.
    const at::Tensor srcRef = src;
    const at::Tensor dstRef = dst;
    auto syn_error = device.copy_data_to_host(
        reinterpret_cast<synapse_helpers::device_ptr>(src.data_ptr()),
        dst.data_ptr(),
        src.nbytes(),
        [srcRef, dstRef]() { return; }, is_pinned);
    TORCH_CHECK(syn_error.status == 0, syn_error.error);
  } else {
    std::atomic<bool> copyDone{false};
    auto syn_error = device.copy_data_to_host(
        reinterpret_cast<synapse_helpers::device_ptr>(src.data_ptr()),
        dst.data_ptr(),
        src.nbytes(),
        [&copyDone]() { copyDone = true; }, is_pinned);
    TORCH_CHECK(syn_error.status == 0, syn_error.error);
    // wait for copy completion
    while (!copyDone) {
      std::this_thread::yield();
    }
  }
}

/******************************************************************************
 * @brief helper function for copying data from host to device
 * @param[in] src_ptr - source memory address in cpu
 * @param[in] size - transfer data size in bytes
 * @param[out] dst - destination tensor in device
 *****************************************************************************/
void habana_helpers::copy_data_to_device(
    const at::Tensor& src,
    const at::Tensor& dst,
    bool non_blocking) {
  auto device_id = dst.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  bool is_pinned = at::habana::PinnedMemoryAllocator_is_pinned(src.data_ptr());

  if (non_blocking) {
    // keeps a reference to the tensor it is
    // operating on to prevent it from being deallocated while the
    // operation is still in flight.
    const at::Tensor srcRef = src;
    const at::Tensor dstRef = dst;
    auto syn_error = device.copy_data_to_device(
        src.data_ptr(),
        reinterpret_cast<synapse_helpers::device_ptr>(dst.data_ptr()),
        src.nbytes(),
        [srcRef, dstRef]() { return; }, is_pinned);
    TORCH_CHECK(syn_error.status == 0, syn_error.error);
  } else {
    std::atomic<bool> copyDone{false};
    auto syn_error = device.copy_data_to_device(
        src.data_ptr(),
        reinterpret_cast<synapse_helpers::device_ptr>(dst.data_ptr()),
        src.nbytes(),
        [&copyDone]() { copyDone = true; }, is_pinned);
    TORCH_CHECK(syn_error.status == 0, syn_error.error);
    // wait for copy completion
    while (!copyDone) {
      std::this_thread::yield();
    }
  }
}

/******************************************************************************
 * @brief helper function for copying data across DRAM within device
 * @param[in] src - source tensor in device
 * @param[out] dst - destination tensor in device
 *****************************************************************************/
void habana_helpers::copy_data_within_device(
    const at::Tensor& src,
    const at::Tensor& dst,
    bool non_blocking) {
  auto device_id = dst.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  if (non_blocking) {
    // keeps a reference to the tensor it is
    // operating on to prevent it from being deallocated while the
    // operation is still in flight.
    const at::Tensor srcRef = src;
    const at::Tensor dstRef = dst;
    auto syn_error = device.copy_data_within_device(
        reinterpret_cast<synapse_helpers::device_ptr>(src.data_ptr()),
        reinterpret_cast<synapse_helpers::device_ptr>(dst.data_ptr()),
        src.nbytes(),
        [srcRef, dstRef]() { return; });
    TORCH_CHECK(syn_error.status == 0, syn_error.error);
  } else {
    std::atomic<bool> copyDone{false};
    auto syn_error = device.copy_data_within_device(
        reinterpret_cast<synapse_helpers::device_ptr>(src.data_ptr()),
        reinterpret_cast<synapse_helpers::device_ptr>(dst.data_ptr()),
        src.nbytes(),
        [&copyDone]() { copyDone = true; });
    TORCH_CHECK(syn_error.status == 0, syn_error.error);
    // wait for copy completion
    while (!copyDone) {
      std::this_thread::yield();
    }
  }
}

void habana_helpers::change_tensor_strides(
    at::Tensor* pt_output,
    const at::Tensor* pt_input,
    const at::IntArrayRef* pt_new_pos) {
  auto sizes = pt_input->sizes().vec();
  auto new_pos = *pt_new_pos;
  std::vector<long int> swapped_sizes = {sizes[new_pos[0]],
                                         sizes[new_pos[1]],
                                         sizes[new_pos[2]],
                                         sizes[new_pos[3]]};
  auto strides = pt_input->strides().vec();
  std::vector<long int> swapped_strides = {strides[new_pos[0]],
                                           strides[new_pos[1]],
                                           strides[new_pos[2]],
                                           strides[new_pos[3]]};
  /* The following method of using 'alias' followed by
   * set_sizes_and_strides is necessary to "dereference" pt_outputs[i]
   * from pt_inputs[i] and create new copies of sizes and strides.
   * Using unsafeGetTensorImpl directly on pt_outputs[i] will
   * reference pt_inputs[i] itself because 'pt_output[i] = pt_input[i]'
   * is a reference copy*/
  *pt_output = at::alias(*pt_input);
  pt_output->unsafeGetTensorImpl()->set_sizes_and_strides(
      swapped_sizes, swapped_strides);
}

void habana_helpers::change_tensors_to_memory_format(
    std::vector<at::Tensor*> pt_outputs,
    std::vector<const at::Tensor*> pt_inputs,
    std::vector<const IntArrayRef*> pt_new_pos,
    c10::MemoryFormat memory_format) {
  auto count = pt_inputs.size();
  for (unsigned i = 0; i < count; i++) {
    switch (memory_format) {
      case c10::MemoryFormat::ChannelsLast: {
        auto sizes = pt_inputs[i]->sizes().vec();
        auto new_pos = *pt_new_pos[i];
        std::vector<long int> swapped_sizes = {sizes[new_pos[0]],
                                               sizes[new_pos[1]],
                                               sizes[new_pos[2]],
                                               sizes[new_pos[3]]};
        auto strides = pt_inputs[i]->strides().vec();
        std::vector<long int> swapped_strides = {strides[new_pos[0]],
                                                 strides[new_pos[1]],
                                                 strides[new_pos[2]],
                                                 strides[new_pos[3]]};
        /* The following method of using 'alias' followed by
         * set_sizes_and_strides is necessary to "dereference" pt_outputs[i]
         * from pt_inputs[i] and create new copies of sizes and strides.
         * Using unsafeGetTensorImpl directly on pt_outputs[i] will
         * reference pt_inputs[i] itself because 'pt_output[i] = pt_input[i]'
         * is a reference copy*/
        *pt_outputs[i] = at::alias(*pt_inputs[i]);
        pt_outputs[i]->unsafeGetTensorImpl()->set_sizes_and_strides(
            swapped_sizes, swapped_strides);
        break;
      }
      case c10::MemoryFormat::Contiguous: {
        // Create dimshuffled inputs and outputs to match synapse data layout
        if (pt_outputs[i] != pt_inputs[i]) {
          auto new_pos = *pt_new_pos[i];
          *pt_outputs[i] = (*pt_inputs[i]).permute(new_pos);
        }
        break;
      }
      default:
        TORCH_CHECK(
            false,
            "Unsupported memory format. Supports only ChannelsLast, Contiguous");
    }
  }
  return;
}

c10::MemoryFormat habana_helpers::get_memory_format(
    std::vector<const at::Tensor*> pt_inputs) {
  auto count = pt_inputs.size();
  TORCH_CHECK(count > 0, "Empty input tensor list given to get_memory_format");
  c10::MemoryFormat memory_format = pt_inputs[0]->suggest_memory_format();
  for (unsigned i = 0; i < count; i++) {
    if (pt_inputs[i]->suggest_memory_format() ==
        c10::MemoryFormat::ChannelsLast) {
      memory_format = c10::MemoryFormat::ChannelsLast;
      break;
    }
  }
  return memory_format;
}
