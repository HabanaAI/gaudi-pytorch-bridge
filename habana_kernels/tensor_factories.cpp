#include <ATen/InferSize.h>
#include <synapse/include/synapse_api.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "habana_helpers/logging.h"
#include "resize.h"

using namespace torch;

static inline void check_size_nonnegative(IntArrayRef size) {
  for (auto x : size) {
    TORCH_CHECK(
        x >= 0,
        "Trying to create tensor with negative dimension ",
        x,
        ": ",
        size);
  }
}

namespace at {
namespace native {
Tensor empty_hpu(
    IntArrayRef size,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format) {
  LOG_FUNC_BEGIN;
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
  check_size_nonnegative(size);

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
  LOG_FUNC_END;
  return tensor;
}

Tensor empty_strided_hpu(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options) {
  LOG_FUNC_BEGIN;
  check_size_nonnegative(size);
  auto t = at::native::empty_hpu({0}, options, c10::nullopt);
  at::native::resize_impl_hpu_(t.unsafeGetTensorImpl(), size, stride);
  LOG_FUNC_END;
  return t;
}
} // namespace native
} // namespace at

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(at::native::empty_hpu),
                    &at::native::empty_hpu>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::empty_strided(int[] size, int[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(at::native::empty_strided_hpu),
                    &at::native::empty_strided_hpu>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::zero_(Tensor(a!) self) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(at::native::zero_),
                    &at::native::zero_>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
