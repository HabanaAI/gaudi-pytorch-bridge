/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Optional.h>
#include "hpu_lazy_tensors.h"

namespace habana_lazy {

struct TORCH_API HbLazyStorageImpl : public c10::StorageImpl {
  explicit HbLazyStorageImpl(const size_t size);

  ~HbLazyStorageImpl() override = default;
};

} // namespace habana_lazy
