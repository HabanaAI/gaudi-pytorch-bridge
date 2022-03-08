/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "lazy_storage.h"
#include <ATen/Tensor.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Optional.h>

namespace habana_lazy {

HbLazyStorageImpl::HbLazyStorageImpl(const size_t size)
    : c10::StorageImpl(
          c10::StorageImpl::use_byte_size_t(),
          size,
          c10::DataPtr{nullptr, {c10::DeviceType::HPU, 0 /* device_id */}},
          nullptr,
          /*resizeable=*/false) {}

} // namespace habana_lazy
