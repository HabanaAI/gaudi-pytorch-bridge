#pragma once
// NOTE: file based on Resize.cuh. It uses THC

#include <ATen/ATen.h>
// TODO: In general we should remove this file.
// Cuda includes THCTensor.hpp and we are including CPU header
#include <TH/THTensor.hpp>

#include "habana_device/HPUAllocator.h"

namespace at {
namespace native {
// TODO: remove static from this function
static void THHStorage_resize(THStorage* self, ptrdiff_t size) {
  TORCH_CHECK(size >= 0, "invalid size");
  TORCH_CHECK(self->allocator() != nullptr);
  int device = habana::allocator_active_device_id;

  TORCH_CHECK(
      self->resizable(), "Trying to resize storage that is not resizable");
  size_t itemsize = self->itemsize();

  if (size == 0) {
    self->set_data_ptr(
        at::DataPtr(nullptr, at::Device(at::DeviceType::HABANA, device)));
    self->set_numel(0);
  } else {
    at::DataPtr data = self->allocator()->allocate(size * itemsize);

    if (self->data_ptr()) {
      auto dma_type = synDmaDir::DRAM_TO_DRAM;

      synStreamHandle stream{};
      TORCH_HABANA_CHECK(
          synStreamCreate(&stream, device, 0),
          "Creating synapse stream failed");

      TORCH_HABANA_CHECK(
          synMemCopyAsync(
              stream,
              reinterpret_cast<uint64_t>(self->data()),
              THMin(self->numel(), size) * itemsize,
              reinterpret_cast<uint64_t>(data.get()),
              dma_type),
          "Synapse DMA: ",
          dma_type,
          "start failed");
      TORCH_HABANA_CHECK(
          synStreamSynchronize(stream), "Stream synchronization failed");

      TORCH_HABANA_CHECK(
          synStreamDestroy(stream), "Destroying synapse stream failed");
    }

    // Destructively overwrite data_ptr
    self->set_data_ptr(std::move(data));
    self->set_numel(size);
  }
}

// These functions are called by native::resize_ as well as (legacy) THC resize.
// They are not in THC/THCTensor.cpp because the at namespace is easier
// to benchmark than THC; I can't get gbenchmark to call fns from THTensor.cpp
inline void maybe_resize_storage_habana(TensorImpl* self, int64_t new_size) {
  // It does not make sense to try to resize a storage
  // to hold 0 elements, and this can break
  // if storage_offset is positive but
  // new_size is 0, so just bail in that case
  // (same comment is in Resize.h)
  if (new_size > 0) {
    if (!THTensor_getStoragePtr(self)) {
      AT_ERROR("Tensor: invalid null storage");
    }
    if (new_size + self->storage_offset() > self->storage().numel()) {
      THHStorage_resize(
          THTensor_getStoragePtr(self), new_size + self->storage_offset());
    }
  }
}

inline TensorImpl* resize_impl_habana_(
    TensorImpl* self,
    IntArrayRef size,
    c10::optional<IntArrayRef> stride,
    bool device_guard = true) {
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
  maybe_resize_storage_habana(self, storage_size);

  return self;
}

} // namespace native
} // namespace at

// THH = TorcH Habana
// TODO: put it in proper namespace
// TODO: remove static from this function
static void THHTensor_resizeNd(
    THTensor* self,
    int nDimension,
    const int64_t* size,
    const int64_t* stride) {
  TORCH_CHECK(nDimension >= 0, "resizeNd nDimension must be non-negative");
  at::IntArrayRef sizes(size, nDimension);
  at::optional<at::IntArrayRef> strides;
  if (stride) {
    strides = at::IntArrayRef(stride, nDimension);
  }
  at::native::resize_impl_habana_(
      self,
      sizes,
      strides,
      /*device_guard=*/false);
}