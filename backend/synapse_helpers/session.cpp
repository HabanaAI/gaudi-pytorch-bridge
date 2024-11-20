/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
