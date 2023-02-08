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

#include <string>

#include <synapse_api.h>
#include "backend/synapse_helpers/session.h"
#include "backend/synapse_helpers/synapse_error.h"
#include "habana_helpers/logging.h" // IWYU pragma: keep

namespace synapse_helpers {

std::weak_ptr<session> session::opened_session;
std::mutex session::session_create_mutex;

session::~session() {
  synDestroy();
} // namespace synapse_helpers

synapse_error_v<std::shared_ptr<session>> session::get_or_create() {
  std::lock_guard<std::mutex> lock(session_create_mutex);
  std::shared_ptr<session> session_ptr = opened_session.lock();
  if (nullptr == session_ptr) {
    auto status = synInitialize();
    SYNAPSE_SUCCESS_CHECK("Session initialization failed.", status);
    // std::make_shared cannot access private ctor
    session_ptr.reset(new session());
    opened_session = session_ptr;
  }

  return session_ptr;
}

} // namespace synapse_helpers
