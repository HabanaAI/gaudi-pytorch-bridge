/**
 * Copyright (c) 2025 Intel Corporation
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

#include "habana_helpers/h2d_scales.h"
#include "backend/habana_device/HPUDevice.h"
#include "backend/synapse_helpers/env_flags_impl.h"

namespace habana_helpers {

bool is_h2d_scales_enabled() {
  return habana::HPUDeviceContext::get_scale_attribute_hash_id() > 0 and
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_SCALES);
}

} // namespace habana_helpers
