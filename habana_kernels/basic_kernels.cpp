#include <ATen/InferSize.h>
#include <synapse/include/synapse_api.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "habana_helpers/logging.h"
#include "resize.h"

using namespace torch;

Tensor permute_hpu(const Tensor& self, IntArrayRef dims) {
  LOG_FUNC_BEGIN;
  auto ret =
      self.to(DeviceType::CPU).permute(dims).contiguous().to(self.device());
  LOG_FUNC_END;
  return ret;
}

// cpu->hpu and hpu->cpu copy implementation
Tensor& copy_hpu_(Tensor& self, const Tensor& src, bool non_blocking) {
  LOG_FUNC_BEGIN;
  if (non_blocking)
    TORCH_WARN(
        "non_blocking flag is not supported, copy_hpu_ is always blocking");
  // TODO: (from torch code) this should be handled during dispatch, but that's
  // missing...
  Tensor& dst = self;
  TORCH_CHECK(dst.defined(), "dst is undefined");
  TORCH_CHECK(src.defined(), "src is undefined");
  TORCH_CHECK(
      dst.nbytes() == src.nbytes(), "src and dst buffers size don't match");

  int device_id = -1;

  const auto src_device = src.device().type();
  const auto dst_device = dst.device().type();
  synDmaDir dma_type;
  void* mapped_addr = nullptr;
  if (src_device == c10::DeviceType::CPU &&
      dst_device == c10::DeviceType::HABANA) {
    device_id = dst.device().index();
    dma_type = synDmaDir::HOST_TO_DRAM;
    mapped_addr = src.data_ptr();
    TORCH_HABANA_CHECK(
        synHostMap(device_id, dst.nbytes(), mapped_addr),
        "Synapse failed to map tensor");
  } else if (
      src_device == c10::DeviceType::HABANA &&
      dst_device == c10::DeviceType::CPU) {
    device_id = src.device().index();
    dma_type = synDmaDir::DRAM_TO_HOST;
    mapped_addr = dst.data_ptr();
    TORCH_HABANA_CHECK(
        synHostMap(device_id, dst.nbytes(), mapped_addr),
        "Synapse failed to map tensor");
  } else if (
      src_device == c10::DeviceType::HABANA &&
      dst_device == c10::DeviceType::HABANA) {
    device_id = dst.device().index();
    TORCH_CHECK(
        dst.device().index() == src.device().index(),
        "Tensors can't be copied between devices using copy_hpu_");
    dma_type = synDmaDir::DRAM_TO_DRAM;
  } else {
    TORCH_CHECK(
        false,
        "copy_hpu_ doesn't support ",
        src_device,
        " to ",
        dst_device,
        "copy");
  }

  synStreamHandle stream{};
  TORCH_HABANA_CHECK(
      synStreamCreate(&stream, device_id, 0), "Creating synapse stream failed");

  TORCH_HABANA_CHECK(
      synMemCopyAsync(
          stream,
          reinterpret_cast<uint64_t>(src.data_ptr()),
          src.nbytes(),
          reinterpret_cast<uint64_t>(dst.data_ptr()),
          dma_type),
      "Synapse DMA: ",
      dma_type,
      "start failed");

  TORCH_HABANA_CHECK(
      synStreamSynchronize(stream), "Stream synchronization failed");

  TORCH_HABANA_CHECK(
      synHostUnmap(device_id, mapped_addr), "Synapse failed to unmap tensor");
  TORCH_HABANA_CHECK(
      synStreamDestroy(stream), "Destroying synapse stream failed");

  LOG_FUNC_END;
  return dst;
}

Tensor& set_hpu_(
    Tensor& self,
    Storage source,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride) {
  LOG_FUNC_BEGIN;
  if (stride.data()) {
    TORCH_CHECK(size.size() == stride.size(), "inconsistent size/stride sizes");
  }

  auto scalar_type = self.scalar_type();
  auto self_ = checked_dense_tensor_unwrap(
      self, "self", 1, "_th_set_", false, DeviceType::HABANA, scalar_type);
  auto source_ = checked_storage(
      source,
      "source",
      2,
      DeviceType::HABANA,
      at::scalarTypeToTypeMeta(scalar_type));

  // Code below is based on THCTensor_setStorage
  TORCH_CHECK(
      self_->storage(),
      "Cannot use PyTorch operations on a half-constructed "
      "tensor.  If this tensor came from Caffe2, please call GetMutableData on "
      "it first; otherwise, this is a bug, please report it.");
  auto self_storage = self_->storage().unsafeGetStorageImpl();
  auto source_storage = source_.unsafeGetStorageImpl();
  if (self_storage != source_storage) {
    TORCH_CHECK(self_storage, "Invalid null storage");
    auto data_type = self_storage->dtype();
    if (self_storage) {
      c10::raw::intrusive_ptr::incref(source_storage);
      THTensor_stealAndSetStoragePtr(self_, source_storage);
    } else {
      auto THHStorage_new = [](caffe2::TypeMeta data_type) -> THStorage* {
        THStorage* storage =
            c10::make_intrusive<at::StorageImpl>(
                data_type, 0, habana::getHABANADeviceAllocator(), true)
                .release();
        return storage;
      };
      THTensor_stealAndSetStoragePtr(self_, THHStorage_new(data_type));
    }
  }

  TORCH_CHECK(storage_offset >= 0, "Invalid storage offset: ", storage_offset);
  self_->set_storage_offset(storage_offset);

  /* size and stride */
  THHTensor_resizeNd(self_, stride.size(), size.data(), stride.data());

  LOG_FUNC_END;
  return self;
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(copy_hpu_), &copy_hpu_>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a)")
                .impl_unboxedOnlyKernel<
                    decltype(at::native::as_strided_tensorimpl),
                    &at::native::as_strided_tensorimpl>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)")
                .impl_unboxedOnlyKernel<decltype(permute_hpu), &permute_hpu>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::set_.source_Storage_storage_offset( Tensor(a !) self, Storage source, int storage_offset, int[] size, int[] stride = []) ->Tensor(a !)")
                .impl_unboxedOnlyKernel<decltype(set_hpu_), &set_hpu_>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::view(Tensor(a) self, int[] size) -> Tensor(a)")
                .impl_unboxedOnlyKernel<
                    decltype(at::native::view),
                    &at::native::view>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
