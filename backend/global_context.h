/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#pragma once

#include "scalar_cache.h"

namespace habana {

namespace backend {

class GlobalContext {
 public:
  GlobalContext() = default;
  GlobalContext(const GlobalContext&) = delete;
  GlobalContext(GlobalContext&&) = delete;
  GlobalContext& operator=(const GlobalContext&) = delete;
  GlobalContext& operator=(GlobalContext&&) = delete;
  ~GlobalContext() = default;

  ScalarCache& GetScalarCache();
  void Clear();

 private:
  ScalarCache scalar_cache_;
};

} // namespace backend
} // namespace habana
