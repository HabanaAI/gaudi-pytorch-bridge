/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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

#include "backend/lazy_to_backend.h"
#include "backend/backend_meta.h"
#include "backend/habana_device/HPUAllocator.h"
#include "backend/helpers/create_tensor.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/lazy_executor.h"

bool lazy_to_backend::is_lazy_inference_call_context() {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    if (!habana_lazy::isDeviceInLoweringMode()) {
      // Lazy mode shape inference call, early return without execution
      return true;
    }
  }
  return false;
}

at::Tensor habana_lazy::empty_hpu_lazy(
    c10::IntArrayRef size,
    const at::TensorOptions& options,
    c10::optional<c10::MemoryFormat> optional_memory_format,
    bool create_storage,
    synTensorType tensor_type,
    c10::optional<std::reference_wrapper<const at::Tensor>> base_view,
    bool is_strided) {
  PT_LAZY_TRACE;
  c10::optional<c10::MemoryFormat> mem_format =
      optional_memory_format.has_value() ? optional_memory_format
                                         : options.memory_format_opt();
  auto original_dtype = options.dtype();
  auto type = c10::typeMetaToScalarType(original_dtype);
  auto shape_tensor = habana_helpers::is_shape_tensor(tensor_type);
  TORCH_CHECK(
      options.pinned_memory() == false,
      "habana allocator doesn't supported pinned memory");

  auto index = options.device_opt().value_or(at::kHPU).index();
  if (index != 0 && index != -1) {
    TORCH_WARN_ONCE(
        "\"hpu:X\" notation is not supported by Gaudi PyTorch "
        "intergration bridge. Please change to \"hpu\" without index");
  }

  c10::Allocator* allocator = habana::getHABANADeviceAllocator();
  HABANA_ASSERT(habana_helpers::is_supported_type(type));

  // Dont allocate 8 bytes for double/long as we are anyway going to cast at
  // CPU and then copy to device @ 4byts per element
  type = habana_helpers::is_downcast_to_int_needed(type) ? c10::ScalarType::Int
                                                         : type;
  type = type == c10::ScalarType::Double ? c10::ScalarType::Float : type;
  auto new_dtype = c10::scalarTypeToTypeMeta(type);

  if (create_storage || shape_tensor) {
    at::Tensor at_internal_tensor;
    int64_t size_bytes = 0;

    if (is_strided && base_view.has_value()) {
      const auto& base = base_view.value().get();
      at_internal_tensor = AtenInternalHbTensor(
          c10::Storage(base.storage()),
          new_dtype,
          tensor_type,
          base.sizes(),
          c10::nullopt,
          mem_format);
    } else {
      int64_t n_elements = multiply_integers(size);
      // we dont create a full storage for shape tensors but we need a backend
      // impl to get meta data
      if (shape_tensor) {
        n_elements =
            (tensor_type == DEVICE_SHAPE_TENSOR) ? SYN_MAX_TENSOR_DIM : 0;
      }
      int elem_size = new_dtype.itemsize();
      size_t storage_size_bytes = n_elements * elem_size;
      size_bytes = n_elements * original_dtype.itemsize();
      auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
          c10::StorageImpl::use_byte_size_t(),
          size_bytes,
          allocator->allocate(storage_size_bytes),
          allocator,
          /*resizeable=*/true);
      at_internal_tensor = AtenInternalHbTensor(
          std::move(storage_impl),
          new_dtype,
          tensor_type,
          size,
          c10::nullopt,
          mem_format);
    }

    // backend tensor should always be contiguous as per view table design
    std::vector<int64_t> contig_strides = at_internal_tensor.strides().vec();
    if (contig_strides.size()) {
      habana_helpers::recalc_strides(
          contig_strides, at_internal_tensor.sizes().vec());
      c10::IntArrayRef new_strides = contig_strides;
      at_internal_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
          at_internal_tensor.sizes(), new_strides);
    }

    // set metadata that its a shape tensor
    if (shape_tensor) {
      auto tmeta{habana::get_tensor_extra_meta(at_internal_tensor, true)};
      if (tmeta) {
        tmeta->set_tensor_type(tensor_type);
      }
    }

    at::Tensor at_tensor;
    bool is_in_lowering_mode = false;
    if (get_habana_lazy_executor().getExecutionMode() ==
        LazyExecutionMode::kLOWERING) {
      is_in_lowering_mode = true;
    }

    // This call could have come from a .to call and not from a lowering
    // context. In such case, create the lazt tensor.
    if (!is_in_lowering_mode) {
      HbLazyTensor hb_tensor = HbLazyTensor::CreateHbLazyTensor(
          size, 0, options.device(), c10::typeMetaToScalarType(original_dtype));

      hb_tensor.SetIsStrided(is_strided);

      // The lazy tensor will have a reference to the internal tensor
      hb_tensor.SetTensorData(at_internal_tensor);

      // Keep a pointer to the storageless tensor from the internal tensor
      auto at_internal_impl = GetHbInternalTensorImpl(at_internal_tensor);
      HABANA_ASSERT(at_internal_impl != nullptr);

      // Any lazy tensor created with storage should be marked as executed
      if (create_storage) {
        hb_tensor.getDataPtr()->execution_status = kEXECUTION_COMPLETE;
        if (size_bytes == 0) {
          PT_LAZY_DEBUG("empty_hpu_lazy: size_bytes is zero!");
          hb_tensor.created_as_zero_size_tensor = true;
        }
      }

      at_tensor = AtenFromHbLazyTensor(
          std::move(hb_tensor), tensor_type, size, c10::nullopt, mem_format);

      // As its an inplace op and we want this op to execute
      // we want to wind back status of this tensor to registered
      // so that when post order is created, we actually execute it
      // auto context =
      //    get_device_lazy_execution_context(
      //        options.device().index());
      // context->MarkTensorStatus(
      //    hb_tensor.getDataPtr(),
      //    LazyTensorExecutionStatus::kINPUT);
      // hb_tensor.IrInitAsInputNode();
    }

    // If we are not from lowering context, return the storageless one.
    if (!is_in_lowering_mode) {
      // Note: storage() api call also sets the front end storage()
      if (create_storage && at_tensor.numel()) {
        if (!is_strided) {
          TORCH_CHECK(
              (at_tensor.storage().data_ptr() &&
               (at_tensor.data_ptr() != nullptr)),
              "t_updated tensor is expected to be have storage and valid data_ptr ",
              at_tensor.storage().data_ptr(),
              " ",
              at_tensor.data_ptr());
        }
      }
      return at_tensor;
    } else {
      // else return the internal tensor with storage
      return at_internal_tensor;
    }
  } else {
    HbLazyTensor hb_tensor = HbLazyTensor::CreateHbLazyTensor(
        size, 0, options.device(), c10::typeMetaToScalarType(original_dtype));
    if (base_view.has_value()) {
      const auto& base = base_view.value().get();
      const auto& storage = base.storage();
      auto key_set = base.key_set();
      return (AtenFromHbLazyTensor(
          std::move(hb_tensor),
          storage,
          key_set,
          tensor_type,
          size,
          c10::nullopt,
          mem_format));
    } else {
      return (AtenFromHbLazyTensor(
          std::move(hb_tensor), tensor_type, size, c10::nullopt, mem_format));
    }
  }
}
