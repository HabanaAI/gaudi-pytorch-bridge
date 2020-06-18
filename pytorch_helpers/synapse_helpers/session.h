/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once

#include <memory>
#include <mutex>

#include "absl/types/variant.h"
#include "synapse_helpers/synapse_error.h" // IWYU pragma: keep

namespace absl {
template <typename... Ts>
class variant;
} // namespace absl

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
