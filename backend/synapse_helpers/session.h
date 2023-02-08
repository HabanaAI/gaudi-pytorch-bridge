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
#pragma once

#include <memory>
#include <mutex>

#include "absl/types/variant.h"
#include "backend/synapse_helpers/synapse_error.h" // IWYU pragma: keep

namespace synapse_helpers {

class session {
 public:
  ~session();
  static synapse_error_v<std::shared_ptr<session>> get_or_create();
  session(const session&) = delete;
  session(session&&) = delete;
  session& operator=(const session&) = delete;
  session& operator=(session&&) = delete;

 private:
  static std::weak_ptr<session> opened_session;
  static std::mutex session_create_mutex;

  session() = default;
};

} // namespace synapse_helpers
