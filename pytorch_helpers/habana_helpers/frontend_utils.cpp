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

#include "habana_helpers/frontend_utils.h"
#include "backend/create_pt_tensor.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/lazy_executor.h"
#include "habana_lazy/permute_tensors.h"

/*************************************************************************
 * @brief This helper function casts a long tensor to int (on CPU)
 ************************************************************************/
at::Tensor habana_helpers::cast_tensor_to_integer(
    const at::Tensor& long_tensor) {
  // TODO Remove this cast on CPU when int64_t->int32 cast available on
  // HPU

  auto int_tensor = std::make_unique<at::Tensor>();
  if (!habana_lazy::isDeviceInLoweringMode() &&
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    // if not in lowering mode just return a tensor storageless wrapper as a
    // placeholder to avoid dma in case we need backend end tensor in future we
    // can replace createpttensor with empty_hpu_lazy
    *int_tensor = habana::createPTTensor(
        long_tensor,
        long_tensor.sizes(),
        long_tensor.options().dtype(c10::ScalarType::Int),
        long_tensor.suggest_memory_format(),
        long_tensor.scalar_type(),
        false);
  } else {
    if (long_tensor.scalar_type() == c10::ScalarType::Long) {
      *int_tensor = long_tensor.to("cpu")
                        .to(c10::ScalarType::Int)
                        .to(long_tensor.device(), c10::attr::non_blocking);
    } else {
      *int_tensor = long_tensor;
    }
  }

  return *int_tensor;
}

at::Tensor habana_helpers::downcast_to_int_if_needed(const at::Tensor& in) {
  return habana_helpers::is_downcast_to_int_needed(in.scalar_type())
      ? habana_helpers::cast_tensor_to_integer(in)
      : in;
}

at::Tensor habana_helpers::cast_tensor_to_long(const at::Tensor& int_tensor) {
  // TODO Remove this cast on CPU when int32->int64_t cast available on
  // HPU
  auto long_tensor = std::make_unique<at::Tensor>();
  if (!habana_lazy::isDeviceInLoweringMode() &&
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    // if not in lowering mode just return a tensor storageless wrapper as a
    // placeholder to avoid dma
    *long_tensor = habana::createPTTensor(
        int_tensor,
        int_tensor.sizes(),
        int_tensor.options().dtype(c10::ScalarType::Long),
        int_tensor.suggest_memory_format(),
        int_tensor.scalar_type(),
        false);
  } else {
    if (int_tensor.scalar_type() == c10::ScalarType::Int) {
      *long_tensor = int_tensor.to("cpu")
                         .to(c10::ScalarType::Long)
                         .to(int_tensor.device(), c10::attr::non_blocking);
    } else {
      *long_tensor = int_tensor;
    }
  }

  return *long_tensor;
}

/******************************************************************************
 * @brief helper function for copying data from device to host
 * @param[in] src - source tensor in device
 * @param[in] size - transfer data size in bytes
 * @param[out] dst_ptr - destination memory address in cpu
 *****************************************************************************/
void habana_helpers::copy_scalar_to_host(
    const at::Tensor& src,
    void* dst_ptr,
    uint32_t size) {
  std::atomic<bool> copyDone{false};
  bool is_pinned = habana::PinnedMemoryAllocator_is_pinned(src.data_ptr());

  auto syn_error =
      synapse_helpers::HPURegistrar::get_device(src.device().index())
          .copy_data_to_host(
              reinterpret_cast<synapse_helpers::device_ptr>(src.data_ptr()),
              dst_ptr,
              reinterpret_cast<synapse_helpers::device_ptr>(
                  src.storage().data_ptr().get()),
              size,
              [&copyDone]() { copyDone = true; },
              is_pinned,
              c10::hpu::getCurrentHPUStream());
  TORCH_CHECK(syn_error.status == 0, syn_error.error);

  // wait for copy completion
  while (!copyDone) {
    std::this_thread::yield();
  }
}
c10::Scalar habana_helpers::_local_scalar_dense_internal(
    const at::Tensor& self) {
  c10::Scalar r;
  // Note:
  // 1. This macro expands to more types than HPU supports,
  //   but that should not be an issue issue.
  // 2. Pytorch uses this function to check a specific emement of a tensor
  //   eg. embedding_bag validates the first value offsets to be 0 using this
  //   function
  // 3. A TORCH_CHECK is added to ensure that the size at source
  //   matches with the destination.

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "_local_scalar_dense",
      [&] {
        scalar_t val;
        TORCH_CHECK(
            elementSize(self.scalar_type()) == sizeof(val),
            " source and destination size mismatch");
        habana_helpers::copy_scalar_to_host(self, &val, sizeof(val));
        r = c10::Scalar(val);
      });
  return r;
}
