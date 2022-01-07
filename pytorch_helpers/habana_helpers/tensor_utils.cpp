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

#include "habana_bridge/kernel/hpu_shape_inference.h"

#include "habana_device/HPUCheck.h"
#include "habana_device/PinnedMemoryAllocator.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_device/tensor_builder.h"

#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"

#include "habana_kernels/habana_operator.h"
#include "habana_kernels/kernel_utils.h"

#include "habana_lazy/lazy_executor.h"
#include "synapse_helpers/env_flags.h"
#include "synapse_helpers/util.h"

using namespace torch;

std::string habana_helpers::DebugString(const at::Tensor& t, bool print_data) {
  std::stringstream O;

  if (t.has_storage()) {
    O << " @ " << (void*)t.storage().data_ptr().get() << " : " << t.data_ptr();
  } else {
    O << " STORAGE_LESS";
  }
  O << ", dim=" << t.dim() << ", shape=" << t.sizes() << ", numel=" << t.numel()
    << ", stride=" << t.strides() << ", layout=" << t.layout() << ','
    << " use_count " << t.use_count();

  if (print_data && t.has_storage() && t.is_cpu()) {
    O << ", contents:" << '\n' << t;
  }

  return O.str();
}

std::string habana_helpers::DebugString(const IVal& a) {
  if (a.isTensor()) {
    habana_helpers::DebugString(a.toTensor());
  }
  return std::string("Non tensor");
}

std::string habana_helpers::DebugString(const IValPtrShared& a) {
  return habana_helpers::DebugString(*a);
}

void habana_helpers::PrintTensor(
    const at::Tensor& t,
    std::string tname,
    bool print_data) {
  PT_TEST_DEBUG("PTI_DBG :: tensor ", tname, " : ", DebugString(t, print_data));
}

/*************************************************************************
 * @brief compute number of elements in a tensor
 ************************************************************************/
int64_t habana_helpers::tensor_numel(const at::Tensor& self) {
  auto shape_vec = self.sizes().vec();
  return std::accumulate(
      shape_vec.cbegin(), shape_vec.cend(), 1, std::multiplies<int64_t>());
}

/*************************************************************************
 * @brief Infers the size of a dim with size -1, if it exists.
 ************************************************************************/
std::vector<int64_t> habana_helpers::infer_size(
    IntArrayRef shape,
    int64_t numel) {
  // call infer_size only if 1 one of the dims is "-1" because there can be
  // cases where a dim is of size "< -1" and there infer_size throws an assert.
  // E.g. if conv2d is called with input which has dim0 of size 0, output
  // computed in at::native::_convolution has a dim with negative size, this
  // causes problem if infer_size is called from subsequent view call on output.
  auto shape_vec = shape.vec();
  auto cond = std::any_of(
      shape_vec.cbegin(), shape_vec.cend(), [](int64_t x) { return x == -1; });
  auto inferred_size = cond ? at::infer_size(shape, numel) : shape_vec;
  return inferred_size;
}

/*************************************************************************
 * @brief Generic helper function to cast tensors on HPU
 ************************************************************************/
at::Tensor habana_helpers::hpu_cast_tensor(
    const at::Tensor& Input,
    caffe2::TypeMeta type) {
  PT_KERNEL_BEGIN;

  // At times we get 0-D tensor which cannot be handled by Synapse. Convert it
  // 1-D tensor before proceeding further.
  if (Input.dim() == 0) {
    Input.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  // Determine cast node_type to use based on src & dst dtypes
  std::pair<c10::ScalarType, c10::ScalarType> type_key{
      Input.scalar_type(), at::typeMetaToScalarType(type)};
  auto iter = habana_helpers::cast_map.find(type_key);
  std::string node_type;
  if (iter != habana_helpers::cast_map.end()) {
    node_type = iter->second;
  } else {
    HABANA_ASSERT(
        0 && "Unsupported Cast operation requested in hpu_cast_tensor()");
  }

  int device_id = Input.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  CastOperator Op(device_id, node_type);
  std::vector<c10::IValue> stack = {
      IValue(Input), IValue(typeMetaToScalarType(type))};
  std::vector<at::Tensor> pt_inputs{Input};

  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto Output = at::empty(
        Input.sizes(),
        Input.options().dtype(type),
        Input.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    std::vector<at::Tensor> v{Output};
    Op.SetPTOutputs(v);
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

  if (!GET_ENV_FLAG_NEW(PT_HPU_LAZY_LOWERING) &&
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    // if not in lowering mode just return a tensor storageless wrapper as a
    // placeholder to avoid dma in case we need backend end tensor in future we
    // can replace createpttensor with empty_hpu_lazy
    *int_tensor = habana_helpers::createPTTensor(
        long_tensor,
        long_tensor.sizes(),
        long_tensor.options().dtype(c10::ScalarType::Int),
        long_tensor.suggest_memory_format(),
        long_tensor.scalar_type(),
        false);
  } else {
    if (long_tensor.scalar_type() == c10::ScalarType::Long) {
      *int_tensor = long_tensor.to("cpu")
                        .to(c10::ScalarType::Int)
                        .to(long_tensor.device(), c10::attr::non_blocking);
    } else {
      *int_tensor = long_tensor;
    }
  }

  return *int_tensor;
}

at::Tensor habana_helpers::cast_tensor_to_long(const at::Tensor& int_tensor) {
  // TODO Remove this cast on CPU when int32->int64_t cast available on
  // HPU
  auto long_tensor = std::make_unique<at::Tensor>();

  if (!GET_ENV_FLAG_NEW(PT_HPU_LAZY_LOWERING) &&
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    // if not in lowering mode just return a tensor storageless wrapper as a
    // placeholder to avoid dma
    *long_tensor = habana_helpers::createPTTensor(
        int_tensor,
        int_tensor.sizes(),
        int_tensor.options().dtype(c10::ScalarType::Long),
        int_tensor.suggest_memory_format(),
        int_tensor.scalar_type(),
        false);
  } else {
    if (int_tensor.scalar_type() == c10::ScalarType::Int) {
      *long_tensor = int_tensor.to("cpu")
                         .to(c10::ScalarType::Long)
                         .to(int_tensor.device(), c10::attr::non_blocking);
    } else {
      *long_tensor = int_tensor;
    }
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
      {c10::ScalarType::Double, synDataType::syn_type_float},
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
      options.device().type() == c10::DeviceType::HPU,
      "Wrong device: ",
      options.device().type());
  auto output = at::empty(std::vector<int64_t>(num_dimensions, 1), options);

  auto self_scalar_type = self.scalar_type();
  if (self_scalar_type == c10::ScalarType::BFloat16) {
    auto val = scalar.to<at::BFloat16>();
    copy_scalar_to_device(&val, output, output.nbytes());
  } else if (
      self_scalar_type == c10::ScalarType::Float ||
      self_scalar_type == c10::ScalarType::Double) {
    auto val = scalar.to<float>();
    copy_scalar_to_device(&val, output, output.nbytes());
  } else if (
      self_scalar_type == c10::ScalarType::Int ||
      self_scalar_type == c10::ScalarType::Long) {
    auto val = scalar.to<int>();
    copy_scalar_to_device(&val, output, output.nbytes());
  } else {
    PT_KERNEL_FATAL(
        "Unsupported data type of scalar when attempting to convert it to a tensor");
  }

  return output;
}

Tensor habana_helpers::GenerateAndCopyTensorToHPU(
    const Tensor& ref_tensor,
    const float value,
    bool is_persistent) {
  // Convert bias_corrections to tensors to avoid cache misses
  Tensor val_t = habana_helpers::createPTTensor(
      ref_tensor,
      {1},
      ref_tensor.options(),
      ref_tensor.suggest_memory_format(),
      c10::ScalarType::Float,
      is_persistent);
  auto size = val_t.numel() * val_t.element_size();
  std::vector<float> buffer(size, value);
  copy_scalar_to_device(buffer.data(), val_t, size);

  return val_t;
}

bool habana_helpers::alwaysAllocOnDevice() {
  static std::once_flag flag;
  static bool allocOnDevice;
  std::call_once(flag, [&]() {
    allocOnDevice = GET_ENV_FLAG_NEW(HABANA_USE_PERSISTENT_TENSOR);
  });
  return allocOnDevice;
}
at::Tensor habana_helpers::nonPersistentTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    const at::TensorOptions& options,
    at::optional<c10::MemoryFormat> optional_memory_format,
    at::optional<caffe2::TypeMeta> data_type) {
  static_cast<void>(options);
  auto t =
      at::detail::make_tensor<habana_helpers::StorageLessWrapperTensorImpl>(
          input, data_type);
  t.unsafeGetTensorImpl()->set_sizes_contiguous(size);

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
  static_cast<void>(options);
  auto t =
      at::detail::make_tensor<habana_helpers::StorageLessWrapperTensorImpl>(
          input, data_type);
  t.unsafeGetTensorImpl()->set_sizes_and_strides(size, strides);
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
  bool is_pinned = habana::PinnedMemoryAllocator_is_pinned(src.data_ptr());

  auto syn_error =
      synapse_helpers::HPURegistrar::get_device(src.device().index())
          .copy_data_to_host(
              reinterpret_cast<synapse_helpers::device_ptr>(src.data_ptr()),
              dst_ptr,
              reinterpret_cast<synapse_helpers::device_ptr>(
                  src.storage().data_ptr().get()),
              size,
              [&copyDone]() { copyDone = true; },
              is_pinned);
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
        reinterpret_cast<synapse_helpers::device_ptr>(
            dst.storage().data_ptr().get()),
        size,
        [dstRef]() { return; });

  } else {
    std::atomic<bool> copyDone{false};
    auto syn_error = device.copy_data_to_device(
        src_ptr,
        reinterpret_cast<synapse_helpers::device_ptr>(dst.data_ptr()),
        reinterpret_cast<synapse_helpers::device_ptr>(
            dst.storage().data_ptr().get()),
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
    const c10::IntArrayRef& stride,
    synapse_helpers::graph& graph,
    bool persistent,
    int devid,
    const c10::ScalarType dtype,
    const std::string& name) {
  uint64_t tensor_id{synapse_helpers::INVALID_SYN_TENSOR_ID};
  // In case of dynamic graph update the name shape map
  if (graph.is_dynamic_graph()) {
    tensor_id = habana::ShapeInference::UpdateShapeInfo(graph, shape.vec());
  }
  if (graph.is_dry_run()) {
    // For dry run mode, just create a placeholder tensor
    return synapse_helpers::tensor::create_placeholder(
        devid, shape.vec(), stride.vec(), name);
  }

  std::vector<int64_t> min, max;
  if (graph.is_dynamic_graph()) {
    std::tie(min, max) = habana::ShapeInference::GetMinMaxShape(tensor_id);
  }

  if (min.size() && max.size() && (min != max)) {
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
                       .with_dynamic_shape(dynamic_shape)
                       .build(
                           synapse_helpers::HPURegistrar::get_device(devid),
                           graph.get_graph_handle());
    return absl::get<synapse_helpers::tensor>(std::move(variant));
  }

  auto builder = synapse_helpers::tensor_builder(
                     shape, stride, pytorch_to_synapse_type(dtype))
                     .mark_persistence(persistent);
  if (!name.empty()) {
    builder.use_suffix(name);
  }
  auto variant = builder.build(
      synapse_helpers::HPURegistrar::get_device(devid),
      graph.get_graph_handle());
  return absl::get<synapse_helpers::tensor>(std::move(variant));
}

synapse_helpers::tensor habana_helpers::create_tensor(
    const at::Tensor& tensor,
    synapse_helpers::graph& graph,
    bool persistent,
    const c10::optional<c10::ScalarType> dtype,
    const std::string& name) {
  uint64_t tensor_id{synapse_helpers::INVALID_SYN_TENSOR_ID};
  // In case of dynamic graph update the name shape map
  if (graph.is_dynamic_graph()) {
    tensor_id =
        habana::ShapeInference::UpdateShapeInfo(graph, tensor.sizes().vec());
  }

  if (graph.is_dry_run()) {
    // For dry run mode, just create a placeholder tensor
    return synapse_helpers::tensor::create_placeholder(
        tensor.device().index(),
        tensor.sizes().vec(),
        tensor.strides().vec(),
        name,
        DATA_TENSOR,
        persistent);
  }

  std::vector<int64_t> min, max;
  if (graph.is_dynamic_graph()) {
    std::tie(min, max) = habana::ShapeInference::GetMinMaxShape(tensor_id);
  }

  if (min.size() && max.size() && (min != max)) {
    auto dynamic_shape = synapse_helpers::tensor::dynamic_shape_t{
        synapse_helpers::to_shape_t(min), synapse_helpers::to_shape_t(max)};
    // Create the max stride
    std::vector<int64_t> max_stride(max.size());
    max_stride[max.size() - 1] = 1;
    for (size_t d = max.size() - 1; d > 0; --d) {
      max_stride[d - 1] = max_stride[d] * max[d];
    }
    auto builder =
        synapse_helpers::tensor_builder(
            max,
            max_stride,
            pytorch_to_synapse_type(dtype.value_or(tensor.scalar_type())))
            .mark_persistence(persistent)
            .with_dynamic_shape(dynamic_shape);
    if (!name.empty()) {
      builder.use_suffix(name);
    }

    auto variant = builder.build(
        synapse_helpers::HPURegistrar::get_device(tensor.device().index()),
        graph.get_graph_handle());
    return absl::get<synapse_helpers::tensor>(std::move(variant));
  }

  uint64_t syn_offset = tensor.storage_offset() * tensor.itemsize();
  auto builder =
      synapse_helpers::tensor_builder(
          tensor.sizes(),
          tensor.strides(),
          pytorch_to_synapse_type(dtype.value_or(tensor.scalar_type())))
          .set_offset(syn_offset)
          .mark_persistence(persistent);
  if (!name.empty()) {
    builder.use_suffix(name);
  }

  auto variant = builder.build(
      synapse_helpers::HPURegistrar::get_device(tensor.device().index()),
      graph.get_graph_handle());
  if (absl::holds_alternative<synapse_helpers::synapse_error>(variant)) {
    auto error = absl::get<synapse_helpers::synapse_error>(variant);
    TORCH_HABANA_CHECK(error.status, error.error);
  }
  return absl::get<synapse_helpers::tensor>(std::move(variant));
}

synapse_helpers::tensor habana_helpers::create_tensor(
    const at::Tensor& tensor,
    synapse_helpers::graph& graph,
    bool persistent,
    const synDataType synType,
    const std::string& name) {
  uint64_t tensor_id{synapse_helpers::INVALID_SYN_TENSOR_ID};
  // In case of dynamic graph update the name shape map
  if (graph.is_dynamic_graph()) {
    tensor_id =
        habana::ShapeInference::UpdateShapeInfo(graph, tensor.sizes().vec());
  }

  if (graph.is_dry_run()) {
    // For dry run mode, just create a placeholder tensor
    return synapse_helpers::tensor::create_placeholder(
        tensor.device().index(),
        tensor.sizes().vec(),
        tensor.strides().vec(),
        name);
  }

  std::vector<int64_t> min, max;
  if (graph.is_dynamic_graph()) {
    std::tie(min, max) = habana::ShapeInference::GetMinMaxShape(tensor_id);
  }

  if (min.size() && max.size() && (min != max)) {
    auto dynamic_shape = synapse_helpers::tensor::dynamic_shape_t{
        synapse_helpers::to_shape_t(min), synapse_helpers::to_shape_t(max)};
    // Create the max stride
    std::vector<int64_t> max_stride(max.size());
    max_stride[max.size() - 1] = 1;
    for (size_t d = max.size() - 1; d > 0; --d) {
      max_stride[d - 1] = max_stride[d] * max[d];
    }
    auto builder = synapse_helpers::tensor_builder(max, max_stride, synType)
                       .mark_persistence(persistent)
                       .with_dynamic_shape(dynamic_shape);
    if (!name.empty()) {
      builder.use_suffix(name);
    }
    auto variant = builder.build(
        synapse_helpers::HPURegistrar::get_device(tensor.device().index()),
        graph.get_graph_handle());
    return absl::get<synapse_helpers::tensor>(std::move(variant));
  }

  auto builder =
      synapse_helpers::tensor_builder(tensor.sizes(), tensor.strides(), synType)
          .mark_persistence(persistent);
  if (!name.empty()) {
    builder.use_suffix(name);
  }
  auto variant = builder.build(
      synapse_helpers::HPURegistrar::get_device(tensor.device().index()),
      graph.get_graph_handle());
  return absl::get<synapse_helpers::tensor>(std::move(variant));
}

synapse_helpers::tensor habana_helpers::create_shape_tensor(
    const at::Tensor& tensor,
    synapse_helpers::graph& graph,
    bool persistent,
    synTensorType shape_tensor_type,
    const std::string& name) {
  uint64_t tensor_id{synapse_helpers::INVALID_SYN_TENSOR_ID};
  // In case of dynamic graph update the name shape map
  if (graph.is_dynamic_graph()) {
    tensor_id =
        habana::ShapeInference::UpdateShapeInfo(graph, tensor.sizes().vec());
  }

  if (graph.is_dry_run()) {
    // For dry run mode, just create a placeholder tensor
    return synapse_helpers::tensor::create_placeholder(
        tensor.device().index(),
        tensor.sizes().vec(),
        tensor.strides().vec(),
        name,
        shape_tensor_type);
  }

  std::vector<int64_t> min, max;
  if (graph.is_dynamic_graph()) {
    std::tie(min, max) = habana::ShapeInference::GetMinMaxShape(tensor_id);
  }

  if (min.size() && max.size() && (min != max)) {
    auto dynamic_shape = synapse_helpers::tensor::dynamic_shape_t{
        synapse_helpers::to_shape_t(min), synapse_helpers::to_shape_t(max)};
    // Create the max stride
    std::vector<int64_t> max_stride(max.size());
    max_stride[max.size() - 1] = 1;
    for (size_t d = max.size() - 1; d > 0; --d) {
      max_stride[d - 1] = max_stride[d] * max[d];
    }
    auto builder =
        synapse_helpers::tensor_builder(
            tensor.sizes(), tensor.strides(), synDataType::syn_type_uint32)
            .with_dynamic_shape(dynamic_shape);
    switch (shape_tensor_type) {
      case SHAPE_TENSOR:
        builder.mark_shape_tensor();
        break;
      case DEVICE_SHAPE_TENSOR:
        builder.mark_device_shape_tensor();
        builder.mark_persistence(persistent);
        break;
      case INPUT_DESCRIBING_SHAPE_TENSOR:
        builder.mark_input_describing_shape_tensor();
        break;
      default:
        HABANA_ASSERT(0 && "Invalid shape_tensor_type");
        break;
    }
    auto variant = builder.build(
        synapse_helpers::HPURegistrar::get_device(tensor.device().index()),
        graph.get_graph_handle());
    synapse_helpers::tensor syn_tensor =
        absl::get<synapse_helpers::tensor>(std::move(variant));
    syn_tensor.set_pt_info(tensor.sizes().vec(), tensor.strides().vec());

    return syn_tensor;
  }
  uint64_t syn_offset = tensor.storage_offset() * tensor.itemsize();
  auto builder =
      synapse_helpers::tensor_builder(
          tensor.sizes(), tensor.strides(), synDataType::syn_type_uint32)
          .set_offset(syn_offset);
  switch (shape_tensor_type) {
    case SHAPE_TENSOR:
      builder.mark_shape_tensor();
      break;
    case DEVICE_SHAPE_TENSOR:
      builder.mark_device_shape_tensor();
      builder.mark_persistence(persistent);
      break;
    case INPUT_DESCRIBING_SHAPE_TENSOR:
      builder.mark_input_describing_shape_tensor();
      break;
    default:
      HABANA_ASSERT(0 && "Invalid shape_tensor_type");
      break;
  }
  if (!name.empty()) {
    builder.use_suffix(name);
  }
  auto variant = builder.build(
      synapse_helpers::HPURegistrar::get_device(tensor.device().index()),
      graph.get_graph_handle());
  synapse_helpers::tensor syn_tensor =
      absl::get<synapse_helpers::tensor>(std::move(variant));
  syn_tensor.set_pt_info(tensor.sizes().vec(), tensor.strides().vec());

  return syn_tensor;
}

std::tuple<std::vector<synapse_helpers::tensor>, std::vector<synTensor>>
habana_helpers::create_tensors(
    const std::vector<at::Tensor>& tensors,
    synapse_helpers::graph& graph,
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
    synapse_helpers::graph& graph,
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
    const synapse_helpers::tensor& tensor,
    synapse_helpers::graph& graph) {
  if (graph.is_dynamic_graph()) {
    habana::ShapeInference::UpdateShapeInfo(graph, tensor.pt_shape());
  }

  if (graph.is_dry_run()) {
    // In case of dry run mode, just create a place holder
    return synapse_helpers::tensor::create_placeholder(
        tensor.device_id(), tensor.pt_shape(), tensor.pt_strides());
  }

  auto builder = synapse_helpers::tensor_builder(
                     tensor.shape(), tensor.stride(), tensor.type())
                     .with_memory_section(tensor.memorysection())
                     .mark_persistence(tensor.is_persistent())
                     .set_offset(tensor.get_offset());

  if (tensor.has_dynamic_shape()) {
    builder.with_dynamic_shape(tensor.dynamic_shape());
  }

  auto maybe_tensor = builder.build(
      synapse_helpers::HPURegistrar::get_device(tensor.device_id()),
      tensor.graph());
  return absl::get<synapse_helpers::tensor>(std::move(maybe_tensor));
}

synapse_helpers::tensor habana_helpers::
    duplicate_tensor_in_memory_section_with_size(
        const synapse_helpers::tensor& tensor,
        synapse_helpers::graph& graph,
        std::vector<int64_t>& sizes,
        std::vector<int64_t>& strides,
        const uint64_t offset) {
  if (graph.is_dynamic_graph()) {
    habana::ShapeInference::UpdateShapeInfo(graph, sizes);
  }

  if (graph.is_dry_run()) {
    // For dry run mode, just create a placeholder tensor
    return synapse_helpers::tensor::create_placeholder(
        tensor.device_id(), sizes, strides);
  }

  TORCH_CHECK(
      tensor.is_persistent(),
      "Why would you like to create another tensor in the same memory section for non persistent tensor?");

  auto builder = synapse_helpers::tensor_builder(sizes, strides, tensor.type())
                     .with_memory_section(tensor.memorysection())
                     .set_offset(offset)
                     .mark_persistence(tensor.is_persistent());

  if (tensor.has_dynamic_shape()) {
    if (synapse_helpers::to_shape_t(sizes) == tensor.dynamic_shape().min()) {
      builder.with_dynamic_shape(tensor.dynamic_shape());
    }
  }

  auto maybe_tensor = builder.build(
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
        return tensor.name();
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
      [](const synapse_helpers::tensor& tensor) { return tensor.name(); });

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
      [](const synapse_helpers::tensor& tensor) { return tensor.name(); });

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

std::vector<synapse_helpers::device_ptr> habana_helpers::
    extract_storage_data_ptrs(const std::vector<const at::Tensor*>& vec) {
  std::vector<synapse_helpers::device_ptr> ptrs;
  ptrs.reserve(vec.size());

  std::transform(
      vec.cbegin(),
      vec.cend(),
      std::back_inserter(ptrs),
      [](const auto& tensor) {
        return reinterpret_cast<synapse_helpers::device_ptr>(
            tensor->storage().data_ptr().get());
      });
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

std::vector<synapse_helpers::device_ptr> habana_helpers::
    extract_storage_data_ptrs(const std::vector<at::Tensor>& vec) {
  std::vector<synapse_helpers::device_ptr> ptrs;
  ptrs.reserve(vec.size());

  std::transform(
      vec.cbegin(),
      vec.cend(),
      std::back_inserter(ptrs),
      [](const auto& tensor) {
        return reinterpret_cast<synapse_helpers::device_ptr>(
            tensor.storage().data_ptr().get());
      });
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
  bool is_pinned = habana::PinnedMemoryAllocator_is_pinned(dst.data_ptr());
  if (src.nbytes() == 0) {
    return;
  }
  if (non_blocking) {
    // keeps a reference to the tensor it is
    // operating on to prevent it from being deallocated while the
    // operation is still in flight.
    const at::Tensor srcRef = src;
    const at::Tensor dstRef = dst;
    auto syn_error = device.copy_data_to_host(
        reinterpret_cast<synapse_helpers::device_ptr>(src.data_ptr()),
        dst.data_ptr(),
        reinterpret_cast<synapse_helpers::device_ptr>(
            src.storage().data_ptr().get()),
        src.nbytes(),
        [srcRef, dstRef]() { return; },
        is_pinned);
    TORCH_CHECK(syn_error.status == 0, syn_error.error);
  } else {
    std::atomic<bool> copyDone{false};
    auto syn_error = device.copy_data_to_host(
        reinterpret_cast<synapse_helpers::device_ptr>(src.data_ptr()),
        dst.data_ptr(),
        reinterpret_cast<synapse_helpers::device_ptr>(
            src.storage().data_ptr().get()),
        src.nbytes(),
        [&copyDone]() { copyDone = true; },
        is_pinned);
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
  bool is_pinned = habana::PinnedMemoryAllocator_is_pinned(src.data_ptr());

  if (src.nbytes() == 0) {
    return;
  }

  if (non_blocking) {
    // keeps a reference to the tensor it is
    // operating on to prevent it from being deallocated while the
    // operation is still in flight.
    const at::Tensor srcRef = src;
    const at::Tensor dstRef = dst;
    auto syn_error = device.copy_data_to_device(
        src.data_ptr(),
        reinterpret_cast<synapse_helpers::device_ptr>(dst.data_ptr()),
        reinterpret_cast<synapse_helpers::device_ptr>(
            dst.storage().data_ptr().get()),
        src.nbytes(),
        [srcRef, dstRef]() { return; },
        is_pinned);
    TORCH_CHECK(syn_error.status == 0, syn_error.error);
  } else {
    std::atomic<bool> copyDone{false};
    auto syn_error = device.copy_data_to_device(
        src.data_ptr(),
        reinterpret_cast<synapse_helpers::device_ptr>(dst.data_ptr()),
        reinterpret_cast<synapse_helpers::device_ptr>(
            dst.storage().data_ptr().get()),
        src.nbytes(),
        [&copyDone]() { copyDone = true; },
        is_pinned);
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
        reinterpret_cast<synapse_helpers::device_ptr>(
            src.storage().data_ptr().get()),
        reinterpret_cast<synapse_helpers::device_ptr>(
            dst.storage().data_ptr().get()),
        src.nbytes(),
        [srcRef, dstRef]() { return; });
    TORCH_CHECK(syn_error.status == 0, syn_error.error);
  } else {
    std::atomic<bool> copyDone{false};
    auto syn_error = device.copy_data_within_device(
        reinterpret_cast<synapse_helpers::device_ptr>(src.data_ptr()),
        reinterpret_cast<synapse_helpers::device_ptr>(dst.data_ptr()),
        reinterpret_cast<synapse_helpers::device_ptr>(
            src.storage().data_ptr().get()),
        reinterpret_cast<synapse_helpers::device_ptr>(
            dst.storage().data_ptr().get()),
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
  std::vector<long int> swapped_sizes = {
      sizes[new_pos[0]],
      sizes[new_pos[1]],
      sizes[new_pos[2]],
      sizes[new_pos[3]]};
  auto strides = pt_input->strides().vec();
  std::vector<long int> swapped_strides = {
      strides[new_pos[0]],
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
      case c10::MemoryFormat::ChannelsLast3d:
      case c10::MemoryFormat::ChannelsLast: {
        auto sizes = pt_inputs[i]->sizes().vec();
        auto new_pos = *pt_new_pos[i];
        auto is_3d_layout = memory_format == c10::MemoryFormat::ChannelsLast3d;
        std::vector<long int> swapped_sizes = {
            sizes[new_pos[0]],
            sizes[new_pos[1]],
            sizes[new_pos[2]],
            sizes[new_pos[3]]};
        auto strides = pt_inputs[i]->strides().vec();
        std::vector<long int> swapped_strides = {
            strides[new_pos[0]],
            strides[new_pos[1]],
            strides[new_pos[2]],
            strides[new_pos[3]]};
        if (is_3d_layout) {
          swapped_sizes.push_back(sizes[new_pos[4]]);
          swapped_strides.push_back(strides[new_pos[4]]);
        }
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
            "Unsupported memory format. Supports only ChannelsLast3d, ChannelsLast, Contiguous");
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
        c10::MemoryFormat::ChannelsLast3d) {
      memory_format = c10::MemoryFormat::ChannelsLast3d;
      break;
    }

    if (pt_inputs[i]->suggest_memory_format() ==
        c10::MemoryFormat::ChannelsLast) {
      memory_format = c10::MemoryFormat::ChannelsLast;
      break;
    }
  }
  return memory_format;
}

size_t habana_helpers::hash_combine_scalars(
    size_t hash_code,
    at::ArrayRef<torch::jit::IValue> input_refs) {
  auto num_inputs = input_refs.size();
  for (unsigned i = 0; i < num_inputs; i++) {
    if (!input_refs[i].isTensor()) {
      if (input_refs[i].isInt()) {
        int val = input_refs[i].toInt();
        std::hash<int> valhash;
        hash_code = at::hash_combine(hash_code, valhash(val));
      } else if (input_refs[i].isBool()) {
        bool val = input_refs[i].toBool();
        hash_code = at::hash_combine(hash_code, val);
      } else if (input_refs[i].isDouble()) {
        double val = input_refs[i].toDouble();
        std::hash<double> valhash;
        hash_code = at::hash_combine(hash_code, valhash(val));
      } else if (input_refs[i].isList()) {
        auto vlist = input_refs[i].toListRef();
        for (auto& v : vlist) {
          if (v.isInt()) {
            int val = v.toInt();
            std::hash<int> valhash;
            hash_code = at::hash_combine(hash_code, valhash(val));
          } else if (v.isBool()) {
            hash_code = at::hash_combine(hash_code, v.toBool());
          } else if (v.isDouble()) {
            double val = v.toDouble();
            std::hash<double> valhash;
            hash_code = at::hash_combine(hash_code, valhash(val));
          }
        }
      } else {
        PT_BRIDGE_DEBUG("Got unhandled Scalar type in hashing");
      }
    }
  }
  return hash_code;
}

void habana_helpers::recalc_strides(
    std::vector<int64_t>& self_strides,
    const std::vector<int64_t>& self_sizes) {
  int k;
  self_strides[self_strides.size() - 1] = 1;
  for (k = self_strides.size() - 2; k >= 0; k--) {
    self_strides[k] = self_strides[k + 1] * self_sizes[k + 1];
  }
  return;
}

bool habana_helpers::is_supported_type(c10::ScalarType type) {
  switch (type) {
    case c10::ScalarType::Byte:
    case c10::ScalarType::Char:
    case c10::ScalarType::Short:
    case c10::ScalarType::Int:
    case c10::ScalarType::Long:
    case c10::ScalarType::Float:
    case c10::ScalarType::Double:
    case c10::ScalarType::Bool:
    case c10::ScalarType::BFloat16:
      return true;
    default:
      return false;
  }
  return false;
}

c10::Scalar habana_helpers::_local_scalar_dense_internal(
    const at::Tensor& self) {
  Scalar r;
  // Note:
  // 1. This macro expands to more types than HPU supports,
  //   but that should not be an issue issue.
  // 2. Pytorch uses this function to check a specific emement of a tensor
  //   eg. embedding_bag validates the first value offsets to be 0 using this
  //   function
  // 3. A TORCH_CHECK is added to ensure that the size at source
  //   matches with the destination.

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "_local_scalar_dense",
      [&] {
        scalar_t val;
        TORCH_CHECK(
            elementSize(self.scalar_type()) == sizeof(val),
            " source and destination size mismatch");
        habana_helpers::copy_scalar_to_host(self, &val, sizeof(val));
        r = Scalar(val);
      });
  return r;
}

bool habana_helpers::is_shape_tensor(synTensorType shape_tensor) {
  switch (shape_tensor) {
    case SHAPE_TENSOR:
    // case OUTPUT_DESCRIBING_SHAPE_TENSOR:
    case INPUT_DESCRIBING_SHAPE_TENSOR:
    case DEVICE_SHAPE_TENSOR:
    case HOST_SHAPE_TENSOR:
      return true;
    default:
      return false;
  };
}
