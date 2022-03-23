/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
// NOTE: file based on Resize.cuh. It uses THC

#include <ATen/ATen.h>
// TODO: In general we should remove this file.
// Cuda includes THCTensor.hpp and we are including CPU header
#include <ATen/native/Resize.h>
#include "habana_device/HPUAllocator.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/unused_macro.h"
#include "kernel_utils.h"
#define THMin(X, Y) ((X) < (Y) ? (X) : (Y))

namespace at {
namespace native {
inline  StorageImpl* THTensor_getStoragePtr(const TensorImpl* tensor) {
// Within PyTorch, the invariant is that storage_ is always
// initialized; we never have tensors that don't have any storage.
// However, for Caffe2, this is not true, because they have permitted
// tensors to be allocated without specifying what scalar type
// they should be, only to be filled when GetMutableData is called
// for the first time (providing the necessary type). It is an ERROR to
// invoke any PyTorch operations on such a half-constructed storage,
// and this check tests for that case.
TORCH_CHECK(tensor->storage(), "Cannot use PyTorch operations on a half-constructed "
"tensor. If this tensor came from Caffe2, please call GetMutableData on "
"it first; otherwise, this is a bug, please report it.");
return tensor->storage().unsafeGetStorageImpl();
}

//inline void THStorage_resizeBytes(THStorage* self, ptrdiff_t size_bytes) {
inline void THStorage_resizeBytes(c10::StorageImpl* self, ptrdiff_t size_bytes) {
  TORCH_CHECK(size_bytes >= 0, "invalid size");
  TORCH_CHECK(self->allocator() != nullptr);
  int device = habana::HPUDeviceAllocator::allocator_active_device_id;

  TORCH_CHECK(
      self->resizable(), "Trying to resize storage that is not resizable");

  if (size_bytes == 0) {
    self->set_data_ptr(
        at::DataPtr(nullptr, at::Device(at::DeviceType::HPU, device)));
    self->set_nbytes(0);
  } else {
    at::DataPtr data = self->allocator()->allocate(size_bytes);

    if (self->data_ptr()) {
      auto& Device = synapse_helpers::HPURegistrar::get_device(device);
      std::mutex mtx;
      std::condition_variable cv;
      std::atomic<bool> copyDone{false};
      std::function<void()> cb = [&copyDone, &mtx, &cv]() {
        std::unique_lock<std::mutex> lck(mtx);
        copyDone = true;
        cv.notify_all();
      };
      auto syn_error = Device.copy_data_within_device(
          reinterpret_cast<synapse_helpers::device_ptr>(self->data()),
          reinterpret_cast<synapse_helpers::device_ptr>(data.get()),
          reinterpret_cast<synapse_helpers::device_ptr>(self->data()),
          reinterpret_cast<synapse_helpers::device_ptr>(data.get()),
          THMin(self->nbytes(), (unsigned long)size_bytes),
          [&copyDone]() { copyDone = true; });
      TORCH_CHECK(syn_error.status == 0, syn_error.error);

      while (!copyDone) {
        std::this_thread::yield();
      }
    }

    // Destructively overwrite data_ptr
    self->set_data_ptr(std::move(data));
    self->set_nbytes(size_bytes);
  }
}

// These functions are called by native::resize_ as well as (legacy) THC resize.
// They are not in THC/THCTensor.cpp because the at namespace is easier
// to benchmark than THC; I can't get gbenchmark to call fns from THTensor.cpp
inline void maybe_resize_storage_hpu(TensorImpl* self, int64_t new_size) {
  // It does not make sense to try to resize a storage
  // to hold 0 elements, and this can break
  // if storage_offset is positive but
  // new_size is 0, so just bail in that case
  // (same comment is in Resize.h)
  if (new_size > 0) {
    if (!THTensor_getStoragePtr(self)) {
      AT_ERROR("Tensor: invalid null storage");
    }
    uint64_t new_size_bytes =
        (new_size + self->storage_offset()) * self->dtype().itemsize();
    if (new_size_bytes > self->storage().nbytes()) {
      THStorage_resizeBytes(THTensor_getStoragePtr(self), new_size_bytes);
    }
  }
}

inline TensorImpl* resize_impl_hpu_(
    TensorImpl* self,
    IntArrayRef size,
    c10::optional<IntArrayRef> stride,
    UNUSED bool device_guard = true) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  // TODO: maybe we should use guard here?
  // NB: We don't need to hold the device guard when calling from TH
  //   cuda::OptionalCUDAGuard guard;
  //   if (device_guard) {
  //     guard.set_index(self->storage().device().index());
  //   }

  int64_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    // NB: storage size can be different from numel.
    for (size_t dim = 0; dim < size.size(); ++dim) {
      // FIXME: Don't rely on storage_size being negative because this
      // may not be true for some edge cases.
      if (size[dim] == 0) {
        storage_size = 0;
        break;
      }
      storage_size += (size[dim] - 1) * stride.value()[dim];
    }
  } else {
    self->set_sizes_contiguous(size);
    storage_size = self->numel();
  }
  maybe_resize_storage_hpu(self, storage_size);

  return self;
}

} // namespace native
} // namespace at

// THH = TorcH Habana
// TODO: put it in proper namespace
inline void THHTensor_resizeNd(
    //THTensor* self,
    c10::TensorImpl* self,
    int nDimension,
    const int64_t* size,
    const int64_t* stride) {
  TORCH_CHECK(nDimension >= 0, "resizeNd nDimension must be non-negative");
  at::IntArrayRef sizes(size, nDimension);
  at::optional<at::IntArrayRef> strides;
  if (stride) {
    strides = at::IntArrayRef(stride, nDimension);
  }
  at::native::resize_impl_hpu_(
      self,
      sizes,
      strides,
      /*device_guard=*/false);
}

inline void THHTensor_resizeNd_nonpersistent(
    //THTensor* self,
    c10::TensorImpl* self,
    int nDimension,
    const int64_t* size,
    const int64_t* stride) {
  TORCH_CHECK(nDimension >= 0, "resizeNd nDimension must be non-negative");
  at::IntArrayRef sizes(size, nDimension);
  at::optional<at::IntArrayRef> strides;
  if (stride) {
    strides = at::IntArrayRef(stride, nDimension);
  }
  if (self->sizes() == sizes && (!stride || self->strides() == strides)) {
    return;
  }
  if (stride) {
    self->set_sizes_and_strides(sizes, *strides);
  } else {
    self->set_sizes_contiguous(sizes);
  }
}
