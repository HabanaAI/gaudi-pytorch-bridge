#include <torch/script.h>

using namespace torch;

Tensor set_one(Tensor image) {
  Tensor output = image;
  for(size_t i = 0; i < image.numel(); ++i){
    output[i] = 1;
  }

  return output;
}

Tensor empty_habana(IntArrayRef size, const TensorOptions& options, c10::optional<MemoryFormat> optional_memory_format) {
  AT_ASSERT(options.backend() == at::Backend::HABANA);
  AT_ASSERT(options.device().type() == DeviceType::HABANA);

  // TODO: how does 'is_variable' affecting us?
  // original comment:
  // is_variable should have been 'unpacked'  TODO: remove this when Variable and Tensor are merged
  AT_ASSERT(!options.is_variable());
  // TORCH_CHECK(!optional_memory_format.has_value(),"'memory_format' argument is incompatible with HABANA tensor");
  TORCH_CHECK(!options.pinned_memory(), "Only dense CPU tensors can be pinned");
  // check_size_nonnegative(size); //TODO: check if tensor constructor checks that

  c10::Allocator* allocator;
  if (options.pinned_memory()) {
    // allocator = at::detail::getCUDAHooks().getPinnedMemoryAllocator();
      TORCH_CHECK(false, "fail, this code will be removed");
  } else {
    allocator = at::getCPUAllocator();
  }

  int64_t nelements = prod_intlist(size);
  auto dtype = options.dtype();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
    dtype,
    nelements,
    allocator->allocate(nelements * dtype.itemsize()),
    allocator,
    /*resizeable=*/true);

  auto tensor = at::detail::make_tensor<TensorImpl>(std::move(storage_impl), at::TensorTypeId::HABANATensorId);
  // Default TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }

  auto memory_format = optional_memory_format.value_or(MemoryFormat::Contiguous);
  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  return tensor;
}

static auto registry = torch::RegisterOperators()
  .op("habana_kernels::set_one", &set_one)
  .op(torch::RegisterOperators::options()
    .schema("aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(empty_habana), &empty_habana>(TensorTypeId::HABANATensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
