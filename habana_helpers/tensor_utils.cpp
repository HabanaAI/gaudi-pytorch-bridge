/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <synapse_helpers/graph.h>
#include <algorithm>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_device/tensor_builder.h"
#include "tensor_utils.h"

at::Tensor habana_helpers::to_cpu(const at::Tensor& hpu_tensor) {
  if (hpu_tensor.defined())
    return hpu_tensor.to(at::DeviceType::CPU);
  else
    return hpu_tensor;
}

synDataType habana_helpers::pytorch_to_synapse_type(
    const c10::ScalarType pt_type) {
  static const std::unordered_map<c10::ScalarType, synDataType> map{
      {c10::ScalarType::Byte, synDataType::syn_type_uint8},
      {c10::ScalarType::Char, synDataType::syn_type_int8},
      {c10::ScalarType::Short, synDataType::syn_type_int16},
      {c10::ScalarType::Int, synDataType::syn_type_int32},
      //   {c10::ScalarType::Long , synDataType::},
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
  if (s.isFloatingPoint()) {
    return c10::ScalarType::Float;
  } else if (s.isIntegral(false)) {
    return c10::ScalarType::Int;
  } else if (s.isBoolean()) {
    return c10::ScalarType::Bool;
  } else
    TORCH_CHECK(!s.isComplex(), "Habana doesn't support complex types");
  throw std::runtime_error("Unknown type");
}

at::Tensor habana_helpers::scalar_to_device_tensor(
    const at::Scalar& scalar,
    const at::TensorOptions& options,
    const unsigned num_dimensions) {
  TORCH_CHECK(
      scalar.isFloatingPoint(),
      "scalar_to_device_tensor currently supports only float");
  TORCH_CHECK(
      options.device().type() == c10::DeviceType::HABANA,
      "Wrong device: ",
      options.device().type());
  auto output = at::empty(std::vector<int64_t>(num_dimensions, 1), options);
  auto val = scalar.to<float>();
  std::mutex mtx;
  std::condition_variable cv;
  bool copyDone = false;

  std::function<void()> cb = [&copyDone, &mtx, &cv]() {
    std::unique_lock<std::mutex> lck(mtx);
    copyDone = true;
    cv.notify_all();
  };

  synapse_helpers::HPURegistrar::get_device(options.device().index())
      .copy_data_to_device(
          &val,
          reinterpret_cast<synapse_helpers::device_ptr>(output.data_ptr()),
          output.nbytes(),
          cb);

  // wait for copy completion
  while (!copyDone) {
    std::unique_lock<std::mutex> lck(mtx);
    cv.wait(lck);
  }

  return output;
}

synapse_helpers::tensor habana_helpers::create_tensor(
    const c10::IntArrayRef& shape,
    synGraphHandle graph,
    bool persistent,
    int devid,
    const c10::ScalarType dtype) {
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
    const std::vector<const at::Tensor*> tensors,
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
    const std::vector<const at::Tensor*> tensors,
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
        *tensors[i],
        graph,
        persistents[i],
        dtypes[i].value_or(tensors[i]->scalar_type())));
    syn_tensors.push_back(tensor_helpers[i].get());
  }

  return {std::move(tensor_helpers), std::move(syn_tensors)};
}

synapse_helpers::tensor habana_helpers::duplicate_tensor_in_memory_section(
    const synapse_helpers::tensor& tensor) {
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

at::Tensor habana_helpers::contiguous_tensor(const at::Tensor& tensor) {
  if (tensor.is_contiguous())
    return tensor;

  auto device = tensor.device();
  // Note: HPU can't perform strided memcopy so I copy data to CPU,
  // override strides, shuffle data accoridngly copy and them back
  auto tensor_contiguous = tensor.to("cpu");
  tensor_contiguous.unsafeGetTensorImpl()->set_sizes_and_strides(
      tensor.sizes(), tensor.strides());
  auto tensor_contiguous2 = tensor_contiguous.contiguous();
  return tensor_contiguous2.to(device);
};

/******************************************************************************
 * @brief helper function for copying data from device to host
 * @param[in] src - source tensor in device
 * @param[in] size - transfer data size in bytes
 * @param[out] dst_ptr - destination memory address in cpu
 *****************************************************************************/
void habana_helpers::copy_data_to_host(
    const at::Tensor& src,
    void* dst_ptr,
    uint32_t size) {
  std::mutex mtx;
  std::condition_variable cv;
  bool copyDone = false;

  // callback for copy completion
  std::function<void()> cb = [&copyDone, &mtx, &cv]() {
    std::unique_lock<std::mutex> lck(mtx);
    copyDone = true;
    cv.notify_all();
  };

  auto syn_error =
      synapse_helpers::HPURegistrar::get_device(src.device().index())
          .copy_data_to_host(
              reinterpret_cast<synapse_helpers::device_ptr>(src.data_ptr()),
              dst_ptr,
              size,
              cb);
  TORCH_CHECK(syn_error.status == 0, syn_error.error);

  // wait for copy completion
  while (!copyDone) {
    std::unique_lock<std::mutex> lck(mtx);
    cv.wait(lck);
  }
}

/******************************************************************************
 * @brief helper function for copying data from host to device
 * @param[in] src_ptr - source memory address in cpu
 * @param[in] size - transfer data size in bytes
 * @param[out] dst - destination tensor in device
 *****************************************************************************/
void habana_helpers::copy_data_to_device(
    void* src_ptr,
    const at::Tensor& dst,
    uint32_t size) {
  std::mutex mtx;
  std::condition_variable cv;
  bool copyDone = false;

  // callback for copy completion
  std::function<void()> cb = [&copyDone, &mtx, &cv]() {
    std::unique_lock<std::mutex> lck(mtx);
    copyDone = true;
    cv.notify_all();
  };

  auto device_id = dst.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  auto syn_error = device.copy_data_to_device(
      src_ptr,
      reinterpret_cast<synapse_helpers::device_ptr>(dst.data_ptr()),
      size,
      cb);
  TORCH_CHECK(syn_error.status == 0, syn_error.error);

  // wait for copy completion
  while (!copyDone) {
    std::unique_lock<std::mutex> lck(mtx);
    cv.wait(lck);
  }
}

/******************************************************************************
 * @brief helper function for copying data across DRAM within device
 * @param[in] src - source tensor in device
 * @param[out] dst - destination tensor in device
 *****************************************************************************/
void habana_helpers::copy_data_within_device(
    const at::Tensor& src,
    const at::Tensor& dst) {
  std::mutex mtx;
  std::condition_variable cv;
  bool copyDone = false;

  // callback for copy completion
  std::function<void()> cb = [&copyDone, &mtx, &cv]() {
    std::unique_lock<std::mutex> lck(mtx);
    copyDone = true;
    cv.notify_all();
  };

  auto device_id = dst.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  auto syn_error = device.copy_data_within_device(
      reinterpret_cast<synapse_helpers::device_ptr>(src.data_ptr()),
      reinterpret_cast<synapse_helpers::device_ptr>(dst.data_ptr()),
      src.nbytes(),
      cb);
  TORCH_CHECK(syn_error.status == 0, syn_error.error);

  // wait for copy completion
  while (!copyDone) {
    std::unique_lock<std::mutex> lck(mtx);
    cv.wait(lck);
  }
}
