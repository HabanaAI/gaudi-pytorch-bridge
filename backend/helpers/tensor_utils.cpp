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
#include "backend/helpers/tensor_utils.h"
#include <ATen/InferSize.h>
#include <c10/core/ScalarType.h>
#include <perf_lib_layer_params.h>
#include <algorithm>
#include "backend/backend_meta.h"
#include "backend/habana_device/HPUStream.h"
#include "backend/habana_device/PinnedMemoryAllocator.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/get_n_bytes.h"
#include "backend/helpers/tensor_info.h"
#include "backend/synapse_helpers/device_helpers.h"
#include "common/utils.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/python_utils.h"
#include "pytorch_helpers/habana_helpers/logging_pt.h" // IWYU pragma: keep // NOLINT

using namespace torch;

void PtTensorInferenceData::print_map() {
  for (auto i : inference_tensor_map) {
    PT_BRIDGE_DEBUG(
        " KEY: ",
        i.first,
        "  min: ",
        i.second.first,
        " max: ",
        i.second.second);
  }
}

std::string PtTensorInferenceData::scope_to_key(std::string src) {
  src = src + ".0";
  src = src.substr(1, src.length() - 1);
  std::replace(src.begin(), src.end(), '/', '.');
  return src;
}

void PtTensorInferenceData::update_map(std::string src, std::string dst) {
  auto src_key = scope_to_key(src);
  auto dst_key = scope_to_key(dst);
  // both keys exist
  if (inference_tensor_map.find(src_key) != inference_tensor_map.end() &&
      inference_tensor_map.find(dst_key) != inference_tensor_map.end()) {
    auto stats = inference_tensor_map[src_key];
    // With this we are losing the stats for dst (ie Conv) and we assume,
    // it will never be needed
    inference_tensor_map.erase(dst_key);
    SetInferenceTensorRange(dst_key, stats.first, stats.second);
  } else {
    PT_BRIDGE_DEBUG(
        " One of these Keys do not exist: ", src_key, "  , or : ", dst_key);
  }
}

void PtTensorInferenceData::duplicate_key(std::string old, std::string now) {
  if (inference_tensor_map.find(old) != inference_tensor_map.end()) {
    auto stats = inference_tensor_map[old];
    SetInferenceTensorRange(now, stats.first, stats.second);
    PT_BRIDGE_DEBUG(" Old key: ", old, " duplicated to new key: ", now);
  } else {
    PT_BRIDGE_DEBUG(" Old Key: ", old, " does not exist.");
  }
}

std::string PtTensorInferenceData::extract_key_name(
    std::string tensor_name,
    const std::string token) {
  auto string_pos = tensor_name.find(token, 1);
  string_pos = (string_pos == std::string::npos) ? 1 : (string_pos + 1);
  auto name = tensor_name.substr(string_pos, tensor_name.length() - string_pos);
  return name;
}

void PtTensorInferenceData::update_entry(
    std::string tensor_name,
    float min,
    float max,
    bool align) {
  auto name = tensor_name;
  if (align == true) {
    name = extract_key_name(tensor_name, "/");
  }
  if (inference_tensor_map.find(name) != inference_tensor_map.end()) {
    PT_BRIDGE_DEBUG(" Update entry for Key: ", name);
    inference_tensor_map.erase(name);
  } else {
    PT_BRIDGE_DEBUG(" Add new entry for Key: ", name);
  }
  SetInferenceTensorRange(name.c_str(), min, max);
}

std::string habana_helpers::DebugString(const at::Tensor& t, bool print_data) {
  std::stringstream O;

  if (t.has_storage()) {
    O << " @ " << (void*)t.storage().data_ptr().get() << " : " << t.data_ptr();
  } else {
    O << " STORAGE_LESS";
  }
  O << ", dim=" << t.dim() << ", shape=" << t.sizes() << ", numel=" << t.numel()
    << ", stride=" << t.strides() << ", layout=" << t.layout() << ','
    << " use_count " << t.use_count() << ',' << " scalar type "
    << t.scalar_type();

  if (print_data && t.has_storage() && t.is_cpu()) {
    O << ", contents:" << '\n' << t;
  }

  return O.str();
}

std::string habana_helpers::DebugString(const IVal& a) {
  std::stringstream O;
  if (a.isTensor()) {
    O << habana_helpers::DebugString(a.toTensor());
  } else if (a.isTensorList()) {
    O << "[" << '\n';
    auto tList = a.toTensorList();
    for (auto t : tList) {
      O << habana_helpers::DebugString(t) << '\n';
    }
    O << "]";
  } else {
    O << " non tensor: " << a;
  }
  return O.str();
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

void habana_helpers::print_tensor_debug(const torch::Tensor& tensor) {
  static constexpr std::string_view marker{"********************\n"};
  PT_BRIDGE_DEBUG(
      marker,
      " Tensor Data info:\n",
      "tensor device: ",
      tensor.device(),
      ",tensor format: ",
      tensor.suggest_memory_format(),
      ", contig?:",
      tensor.is_contiguous(),
      ", tensor.strides(): ",
      tensor.strides(),
      ", tensor.sizes(): ",
      tensor.sizes(),
      "\n",
      marker);
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

/******************************************************************************
 * @brief helper function for copying data from host to device
 * @param[in] src_ptr - source memory address in cpu
 * @param[in] size - transfer data size in bytes
 * @param[out] dst - destination tensor in device
 *****************************************************************************/
void habana_helpers::copy_scalar_to_device(
    void* src_ptr,
    const at::Tensor& dst,
    uint64_t size) {
  auto device_id = dst.device().index();
  auto& device = habana::HPURegistrar::get_device(device_id);
  if (device.IsStreamASyncEnabled()) {
    // keeps a reference to the tensor it is
    // operating on to prevent it from being deallocated while the
    // operation is still in flight.
    const at::Tensor dstRef = dst;
    device.copy_data_to_device(
        src_ptr,
        reinterpret_cast<synapse_helpers::device_ptr>(dst.data_ptr()),
        reinterpret_cast<synapse_helpers::device_ptr>(
            dst.storage().data_ptr().get()),
        size,
        [dstRef]() { return; },
        c10::hpu::getCurrentHPUStream());
  } else {
    std::atomic<bool> copyDone{false};
    device.copy_data_to_device(
        src_ptr,
        reinterpret_cast<synapse_helpers::device_ptr>(dst.data_ptr()),
        reinterpret_cast<synapse_helpers::device_ptr>(
            dst.storage().data_ptr().get()),
        size,
        [&copyDone]() { copyDone = true; },
        c10::hpu::getCurrentHPUStream());

    // Release GIL if going to wait
    habana_helpers::AutoNoGIL gil_release;
    // wait for copy completion
    while (!copyDone) {
      std::this_thread::yield();
    }
  }
}

/******************************************************************************
 * @brief helper function for copying scalars tensor data from host to device
 * @param[in] tensor_list - list of tensor pairs i.e. src and dst
 *****************************************************************************/
void habana_helpers::copy_scalars_to_device(
    const std::vector<std::pair<at::Tensor, at::Tensor>>& tensors_list) {
  if (tensors_list.empty()) {
    return;
  }

  synapse_helpers::device::transfer_manifest manifest;
  std::vector<at::Tensor> src_list;
  std::vector<at::Tensor> dst_list;
  for (auto pair : tensors_list) {
    auto src = pair.first;
    auto dst = pair.second;
    TORCH_CHECK(dst.nbytes() >= src.nbytes());

    synapse_helpers::device::transfer_desc desc;
    desc.src = reinterpret_cast<synapse_helpers::device_ptr>(src.data_ptr());
    desc.bytes_to_transfer = habana_helpers::GetNBytes(src);
    desc.dst = reinterpret_cast<synapse_helpers::device_ptr>(dst.data_ptr());
    desc.dst_event_addr = reinterpret_cast<synapse_helpers::device_ptr>(
        dst.storage().data_ptr().get());
    manifest.push_back(desc);

    src_list.push_back(src);
    dst_list.push_back(dst);
  }

  auto& device = habana::HPURegistrar::get_device();
  if (device.IsStreamASyncEnabled()) {
    // src list and dst list keeps a reference to the tensors it is
    // operating on to prevent it from being deallocated while the
    // operation is still in flight.
    device.copy_data_to_device(
        manifest,
        [src_list, dst_list]() { return; },
        c10::hpu::getCurrentHPUStream());
  } else {
    std::atomic<bool> copyDone{false};
    device.copy_data_to_device(
        manifest,
        [&copyDone]() { copyDone = true; },
        c10::hpu::getCurrentHPUStream());

    // Release GIL if going to wait
    habana_helpers::AutoNoGIL gil_release;
    // wait for copy completion
    while (!copyDone) {
      std::this_thread::yield();
    }
  }
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
  return copy_data_to_host(
      src, dst, non_blocking, c10::hpu::getCurrentHPUStream());
}

void habana_helpers::copy_data_to_host(
    const at::Tensor& src,
    const at::Tensor& dst,
    bool non_blocking,
    synapse_helpers::hpuStream_t hpu_stream) {
  size_t device_id = src.device().index();
  auto& device = habana::HPURegistrar::get_device(device_id);
  bool is_pinned = habana::PinnedMemoryAllocator_is_pinned(dst.data_ptr());
  if (src.nbytes() == 0) {
    return;
  }

  auto tmeta{habana::get_tensor_extra_meta(src)};
  if (tmeta->has_valid_const_id()) {
    HABANA_ASSERT(
        tmeta->get_host_ptr() != nullptr,
        "Host pointer can not be invalid for const id: ",
        tmeta->get_const_id());
    std::memcpy(
        dst.data_ptr(), tmeta->get_host_ptr(), habana_helpers::GetNBytes(src));
    return;
  }

  auto src_data_ptr = src.data_ptr();
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0 && src.storage_offset() != 0) {
    // HPU is implicitly casting Long to Int and Double to Float without showing
    // this information to torch This results in data_ptr offset being computed
    // incorrectly. This is to adjust to this.
    switch (src.scalar_type()) {
      case c10::ScalarType::Long:
        if (!common::IsInt64Supported()) {
          src_data_ptr = reinterpret_cast<void*>(
              reinterpret_cast<uint8_t*>(src_data_ptr) -
              src.storage_offset() * sizeof(long) / 2);
        }
        break;
      case c10::ScalarType::Double:
        src_data_ptr = reinterpret_cast<void*>(
            reinterpret_cast<uint8_t*>(src_data_ptr) -
            src.storage_offset() * sizeof(double) / 2);
        break;
      default:
        break;
    }
  }

  if (non_blocking && device.IsStreamASyncEnabled()) {
    // keeps a reference to the tensor it is
    // operating on to prevent it from being deallocated while the
    // operation is still in flight.
    struct ResourceHolder {
      ResourceHolder(const at::Tensor& src, const at::Tensor& dst)
          : src_(src), dst_(dst) {}

      at::Tensor src_;
      at::Tensor dst_;

      void release_resources() {
        src_ = at::Tensor();
        dst_ = at::Tensor();
      }
    };

    auto callback = [rh = std::make_shared<ResourceHolder>(
                         src, dst)]() mutable { rh->release_resources(); };

    device.copy_data_to_host(
        reinterpret_cast<synapse_helpers::device_ptr>(src_data_ptr),
        dst.data_ptr(),
        reinterpret_cast<synapse_helpers::device_ptr>(
            src.storage().data_ptr().get()),
        habana_helpers::GetNBytes(src),
        callback,
        is_pinned,
        hpu_stream);
  } else {
    std::atomic<bool> copyDone{false};
    device.copy_data_to_host(
        reinterpret_cast<synapse_helpers::device_ptr>(src_data_ptr),
        dst.data_ptr(),
        reinterpret_cast<synapse_helpers::device_ptr>(
            src.storage().data_ptr().get()),
        habana_helpers::GetNBytes(src),
        [&copyDone]() { copyDone = true; },
        is_pinned,
        hpu_stream);

    // Release GIL if going to wait
    habana_helpers::AutoNoGIL gil_release;
    // wait for copy completion
    while (!copyDone) {
      std::this_thread::yield();
    }
  }
}

void habana_helpers::copy_data_to_device(
    const at::Tensor& src,
    const at::Tensor& dst,
    bool non_blocking) {
  return copy_data_to_device(
      src, dst, non_blocking, c10::hpu::getCurrentHPUStream());
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
    bool non_blocking,
    synapse_helpers::hpuStream_t hpu_stream,
    void* host_ptr) {
  auto device_id = dst.device().index();
  auto& device = habana::HPURegistrar::get_device(device_id);
  bool is_pinned = habana::PinnedMemoryAllocator_is_pinned(src.data_ptr());

  if (src.nbytes() == 0) {
    return;
  }

  if (non_blocking && device.IsStreamASyncEnabled()) {
    // check if non pinned memory and CPU host memory
    // already allocated in the main thread
    void* host_cpu_data = nullptr;
    if (!is_pinned) {
      if (host_ptr) {
        host_cpu_data = host_ptr;
      }
    }

    // keeps a reference to the tensor it is
    // operating on to prevent it from being deallocated while the
    // operation is still in flight.
    struct ResourceHolder {
      ResourceHolder(const at::Tensor& src, const at::Tensor& dst)
          : src_(src), dst_(dst) {}

      at::Tensor src_;
      at::Tensor dst_;

      void release_resources() {
        src_ = at::Tensor();
        dst_ = at::Tensor();
      }
    };

    auto callback = [rh = std::make_shared<ResourceHolder>(
                         src, dst)]() mutable { rh->release_resources(); };

    device.copy_data_to_device(
        src.data_ptr(),
        reinterpret_cast<synapse_helpers::device_ptr>(dst.data_ptr()),
        reinterpret_cast<synapse_helpers::device_ptr>(
            dst.storage().data_ptr().get()),
        habana_helpers::GetNBytes(src),
        callback,
        non_blocking,
        is_pinned,
        hpu_stream,
        host_cpu_data);
  } else {
    std::atomic<bool> copyDone{false};
    device.copy_data_to_device(
        src.data_ptr(),
        reinterpret_cast<synapse_helpers::device_ptr>(dst.data_ptr()),
        reinterpret_cast<synapse_helpers::device_ptr>(
            dst.storage().data_ptr().get()),
        habana_helpers::GetNBytes(src),
        [&copyDone]() { copyDone = true; },
        false,
        is_pinned,
        hpu_stream);

    // Release GIL if going to wait
    habana_helpers::AutoNoGIL gil_release;
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
  auto& device = habana::HPURegistrar::get_device(device_id);

  if (non_blocking && device.IsStreamASyncEnabled()) {
    // keeps a reference to the tensor it is
    // operating on to prevent it from being deallocated while the
    // operation is still in flight.
    struct ResourceHolder {
      ResourceHolder(const at::Tensor& src, const at::Tensor& dst)
          : src_(src), dst_(dst) {}

      at::Tensor src_;
      at::Tensor dst_;

      void release_resources() {
        src_ = at::Tensor();
        dst_ = at::Tensor();
      }
    };

    auto callback = [rh = std::make_shared<ResourceHolder>(
                         src, dst)]() mutable { rh->release_resources(); };

    device.copy_data_within_device(
        reinterpret_cast<synapse_helpers::device_ptr>(src.data_ptr()),
        reinterpret_cast<synapse_helpers::device_ptr>(dst.data_ptr()),
        reinterpret_cast<synapse_helpers::device_ptr>(
            src.storage().data_ptr().get()),
        reinterpret_cast<synapse_helpers::device_ptr>(
            dst.storage().data_ptr().get()),
        habana_helpers::GetNBytes(src),
        callback,
        c10::hpu::getCurrentHPUStream());
  } else {
    std::atomic<bool> copyDone{false};
    device.copy_data_within_device(
        reinterpret_cast<synapse_helpers::device_ptr>(src.data_ptr()),
        reinterpret_cast<synapse_helpers::device_ptr>(dst.data_ptr()),
        reinterpret_cast<synapse_helpers::device_ptr>(
            src.storage().data_ptr().get()),
        reinterpret_cast<synapse_helpers::device_ptr>(
            dst.storage().data_ptr().get()),
        habana_helpers::GetNBytes(src),
        [&copyDone]() { copyDone = true; },
        c10::hpu::getCurrentHPUStream());

    // Release GIL if going to wait
    habana_helpers::AutoNoGIL gil_release;
    // wait for copy completion
    while (!copyDone) {
      std::this_thread::yield();
    }
  }
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
  if (self_strides.size() == 0) {
    return;
  }
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
    case c10::ScalarType::Half: {
      auto device_type{habana::HPURegistrar::get_device().type()};
      if (device_type == synDeviceGaudi) {
        HABANA_ASSERT(false, "float16/half is not supported on Gaudi.");
      }
      return synapse_helpers::device_supports_fp16(device_type);
    }
    case c10::ScalarType::ComplexHalf:
    case c10::ScalarType::ComplexFloat:
    case c10::ScalarType::ComplexDouble: {
      TORCH_CHECK(false, "Complex datatype is not supported on HPU device.");
      return false;
    }
#if HAVE_FP8_SUPPORT
    case c10::ScalarType::Float8_e5m2:
    case c10::ScalarType::Float8_e4m3fn: {
      return synapse_helpers::device_supports_fp8(
          habana::HPURegistrar::get_device().type());
    }
#endif
    default:
      return false;
  }
  return false;
}

bool habana_helpers::is_shape_tensor(synTensorType shape_tensor) {
  switch (shape_tensor) {
    case SHAPE_TENSOR:
    // case OUTPUT_DESCRIBING_SHAPE_TENSOR:
    case DEVICE_SHAPE_TENSOR:
    case HOST_SHAPE_TENSOR:
    case HOST_TO_DEVICE_TENSOR:
      return true;
    default:
      return false;
  };
}

std::vector<int64_t> habana_helpers::calculate_strides(
    std::vector<int64_t> sizes) {
  // With view table based design tensor strides should always be contiguous
  const auto dim_ = sizes.size();
  std::vector<int64_t> strides(dim_);
  if (dim_ > 0) {
    const auto last_idx = dim_ - 1;
    strides[last_idx] = 1;
    for (int64_t i = last_idx - 1; i >= 0; --i) {
      strides[i] = strides[i + 1] * std::max<int64_t>(sizes[i + 1], 1);
    }
  }
  return strides;
}

std::string habana_helpers::detail::
    InternalFormatter<habana_helpers::FormatTokens>::format(
        const at::Tensor& tensor,
        habana_helpers::FormatTokens token) {
  auto tmeta{habana::get_tensor_extra_meta(tensor, true)};
  if (!tmeta) {
    return "<NO_TMETA>";
  }
  auto smeta{habana::get_storage_extra_meta(tensor)};
  switch (token) {
    case habana_helpers::FormatTokens::Permutations:
      return (smeta ? VecToString(smeta->get_memory_permutation()) : "");
    case habana_helpers::FormatTokens::Layout:
      return habana::DebugString(tmeta->get_tensor_layout());
    case habana_helpers::FormatTokens::ImplPtr:
      return absl::StrCat(
          absl::Hex(tensor.unsafeGetTensorImpl(), absl::kZeroPad8));
    case habana_helpers::FormatTokens::DataPtr:
      return absl::StrCat(absl::Hex(tensor.data_ptr(), absl::kZeroPad8));
  }
  return "";
}
