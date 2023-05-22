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
#include <ATen/InferSize.h>
#include <perf_lib_layer_params.h>
#include <algorithm>
#include <mutex>
#include "backend/backend_meta.h"
#include "backend/synapse_helpers/graph.h"

#include "backend/habana_device/HPUStream.h"
#include "backend/habana_device/PinnedMemoryAllocator.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/habana_device/tensor_builder.h"
#include "habana_helpers/logging.h"

#include "backend/helpers/graph.h"
#include "habana_helpers/pt_version_check.h"

#include "backend/helpers/tensor_utils.h"

#include "backend/create_pt_tensor.h"
#include "backend/habana_operator.h"
#include "backend/helpers/get_n_bytes.h"
#include "backend/lazy_to_backend.h"
#include "habana_kernels/kernel_utils.h"

#include "backend/synapse_helpers/device_helpers.h"
#include "backend/synapse_helpers/env_flags.h"
#include "backend/synapse_helpers/util.h"

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

Tensor habana_helpers::GenerateAndCopyTensorToHPU(
    const Tensor& ref_tensor,
    const float value,
    bool is_persistent) {
  // Convert bias_corrections to tensors to avoid cache misses
  Tensor val_t = habana::createPTTensor(
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
  if (non_blocking && device.IsStreamASyncEnabled()) {
    // keeps a reference to the tensor it is
    // operating on to prevent it from being deallocated while the
    // operation is still in flight.
    const at::Tensor srcRef = src;
    const at::Tensor dstRef = dst;
    device.copy_data_to_host(
        reinterpret_cast<synapse_helpers::device_ptr>(src.data_ptr()),
        dst.data_ptr(),
        reinterpret_cast<synapse_helpers::device_ptr>(
            src.storage().data_ptr().get()),
        habana_helpers::GetNBytes(src),
        [srcRef, dstRef]() { return; },
        is_pinned,
        hpu_stream);
  } else {
    std::atomic<bool> copyDone{false};
    device.copy_data_to_host(
        reinterpret_cast<synapse_helpers::device_ptr>(src.data_ptr()),
        dst.data_ptr(),
        reinterpret_cast<synapse_helpers::device_ptr>(
            src.storage().data_ptr().get()),
        habana_helpers::GetNBytes(src),
        [&copyDone]() { copyDone = true; },
        is_pinned,
        hpu_stream);
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
    synapse_helpers::hpuStream_t hpu_stream) {
  auto device_id = dst.device().index();
  auto& device = habana::HPURegistrar::get_device(device_id);
  bool is_pinned = habana::PinnedMemoryAllocator_is_pinned(src.data_ptr());

  if (src.nbytes() == 0) {
    return;
  }

  if (non_blocking && device.IsStreamASyncEnabled()) {
    // keeps a reference to the tensor it is
    // operating on to prevent it from being deallocated while the
    // operation is still in flight.
    const at::Tensor srcRef = src;
    const at::Tensor dstRef = dst;
    device.copy_data_to_device(
        src.data_ptr(),
        reinterpret_cast<synapse_helpers::device_ptr>(dst.data_ptr()),
        reinterpret_cast<synapse_helpers::device_ptr>(
            dst.storage().data_ptr().get()),
        habana_helpers::GetNBytes(src),
        [srcRef, dstRef]() { return; },
        non_blocking,
        is_pinned,
        hpu_stream);
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
    const at::Tensor srcRef = src;
    const at::Tensor dstRef = dst;
    device.copy_data_within_device(
        reinterpret_cast<synapse_helpers::device_ptr>(src.data_ptr()),
        reinterpret_cast<synapse_helpers::device_ptr>(dst.data_ptr()),
        reinterpret_cast<synapse_helpers::device_ptr>(
            src.storage().data_ptr().get()),
        reinterpret_cast<synapse_helpers::device_ptr>(
            dst.storage().data_ptr().get()),
        habana_helpers::GetNBytes(src),
        [srcRef, dstRef]() { return; },
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
#if HAVE_FP8R152_SUPPORT
    case c10::ScalarType::Fp8r152: {
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
