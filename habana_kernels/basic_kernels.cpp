#include <ATen/InferSize.h>
#include <synapse/include/synapse_api.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "resize.h"

using namespace torch;

Tensor habana_permute(const Tensor& self, IntArrayRef dims) {
  std::cout << "habana_permute called\n";
  return self.to(DeviceType::CPU).permute(dims).contiguous().to(self.device());
}

Tensor habana_empty(
    IntArrayRef size,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format) {
  // AT_ASSERT(options.backend() == at::Backend::HABANA);
  AT_ASSERT(options.device().type() == DeviceType::HABANA);

  // TODO: how does 'is_variable' affecting us?
  // original comment:
  // is_variable should have been 'unpacked'  TODO: remove this when Variable
  // and Tensor are merged
  // AT_ASSERT(!options.is_variable());
  // TORCH_CHECK(!optional_memory_format.has_value(),"'memory_format' argument
  // is incompatible with HABANA tensor");
  TORCH_CHECK(!options.pinned_memory(), "Only dense CPU tensors can be pinned");
  // check_size_nonnegative(size); //TODO: check if tensor constructor checks
  // that

  c10::Allocator* allocator;
  if (options.pinned_memory()) {
    TORCH_CHECK(false, "habana allocator doesn't supported pinned memory");
  } else {
    allocator = at::habana::getHABANADeviceAllocator();
  }

  int64_t nelements = prod_intlist(size);
  auto dtype = options.dtype();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      dtype,
      nelements,
      allocator->allocate(nelements * dtype.itemsize()),
      allocator,
      /*resizeable=*/true);

  auto tensor = at::detail::make_tensor<TensorImpl>(
      std::move(storage_impl), at::TensorTypeId::HABANATensorId);
  // Default TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }

  auto memory_format =
      optional_memory_format.value_or(MemoryFormat::Contiguous);
  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  return tensor;
}

// cpu->hpu and hpu->cpu copy implementation
Tensor& hpu_copy_(Tensor& self, const Tensor& src, bool non_blocking) {
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
        "Tensors can't be copied between devices using hpu_copy_");
    dma_type = synDmaDir::DRAM_TO_DRAM;
  } else {
    TORCH_CHECK(
        false,
        "hpu_copy_ doesn't support ",
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

  return dst;
}

Tensor& habana_set(
    Tensor& self,
    Storage source,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride) {
  std::cout << "habana_set called\n";
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

  return self;
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(habana_empty), &habana_empty>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(hpu_copy_), &hpu_copy_>(
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
                .impl_unboxedOnlyKernel<
                    decltype(habana_permute),
                    &habana_permute>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::set_.source_Storage_storage_offset( Tensor(a !) self, Storage source, int storage_offset, int[] size, int[] stride = []) ->Tensor(a !)")
                .impl_unboxedOnlyKernel<decltype(habana_set), &habana_set>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::view(Tensor(a) self, int[] size) -> Tensor(a)")
                .impl_unboxedOnlyKernel<
                    decltype(at::native::view),
                    &at::native::view>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
