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
#include <synapse_api.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/logging.h"
#include "resize.h"

using namespace torch;

#ifdef PT_KERNEL_END
#undef PT_KERNEL_BEGIN
#define PT_KERNEL_BEGIN (void)(0)
#undef PT_KERNEL_END
#define PT_KERNEL_END (void)(0)
#endif

// static inline void check_size_nonnegative(IntArrayRef size) {
//   for (auto x : size) {
//     TORCH_CHECK(
//         x >= 0,
//         "Trying to create tensor with negative dimension ",
//         x,
//         ": ",
//         size);
//   }
// }

Tensor empty_hpu(
    IntArrayRef size,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format) {
  PT_OTHER_OPS_BEGIN; // this macro is used because this kernel is used
                      // within Lazy kernels
  // AT_ASSERT(options.backend() == at::Backend::HABANA);
  AT_ASSERT(options.device().type() == DeviceType::HPU);

  // TODO: how does 'is_variable' affecting us?
  // original comment:
  // is_variable should have been 'unpacked'  TODO: remove this when Variable
  // and Tensor are merged
  // AT_ASSERT(!options.is_variable());
  TORCH_CHECK(!options.pinned_memory(), "Only dense CPU tensors can be pinned");
  at::check_size_nonnegative(size);

  c10::Allocator* allocator;
  if (options.pinned_memory()) {
    TORCH_CHECK(false, "habana allocator doesn't supported pinned memory");
  } else {
    allocator = habana::getHABANADeviceAllocator();
  }

  int64_t nelements = multiply_integers(size);
  auto dtype = options.dtype();
  int64_t size_bytes = nelements * dtype.itemsize();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /*resizeable=*/true);

  auto tensor = at::detail::make_tensor<TensorImpl>(
      std::move(storage_impl),
      c10::DispatchKeySet{
          at::DispatchKey::HPU, at::DispatchKey::AutogradHPU},
      dtype);
  // Default TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }

  auto memory_format = optional_memory_format.has_value()
      ? optional_memory_format.value_or(MemoryFormat::Contiguous)
      : options.memory_format_opt().value_or(MemoryFormat::Contiguous);
  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  PT_KERNEL_END;
  return tensor;
}

Tensor empty_strided_hpu(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options) {
  PT_OTHER_OPS_BEGIN; // this macro is used because this kernel is used
                      // in other kernels
  at::check_size_nonnegative(size);
  auto t = empty_hpu({0}, options, c10::nullopt);
  at::native::resize_impl_hpu_(t.unsafeGetTensorImpl(), size, stride);
  PT_KERNEL_END;
  return t;
}

Tensor clone_hpu(
    const Tensor& self,
    c10::optional<MemoryFormat> memory_format) {
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::clone(self, memory_format);
}

Tensor& zero_hpu(Tensor& self) {
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::zero_(self);
}

Tensor ones_like_hpu(
    const Tensor& self,
    const TensorOptions& options,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  static_cast<void>(options);
  static_cast<void>(optional_memory_format);
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "constant_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Note that for now we are ignoring options & optional_memory_format
  // provided. If required we can change the code to create a tensor
  // with required options & memory_format, followed by calling
  // ConstantOut OP.

  ConstantOperator Op(device_id, scalar_type);
  std::vector<c10::IValue> stack = {IValue(self), IValue(Scalar(1.0))};
  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output = habana_helpers::createPTTensor(self, true);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    Op.Compile(graph);
  }

  return Op.GetOutputs()[0];
};
