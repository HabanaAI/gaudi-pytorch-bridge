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
