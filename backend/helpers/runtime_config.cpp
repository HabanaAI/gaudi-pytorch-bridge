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

#include "backend/helpers/runtime_config.h"
#include "backend/synapse_helpers/env_flags.h"

#include <string>

namespace habana_helpers {
bool enable_inference_mode{GET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE)};

bool enable_quantization = false;
// if a proper path is set,const section serialization will be enabled.
std::string const_section_serialize_path = "";
// if true, remove all existingconst section files in given path.
bool clear_const_section_path = false;
// if true, compress the constant tensor data before serializing onto the disk
bool enable_compression = false;

// if true enables recompute based fused SDPA
bool enabled_recomputeFSDPA = true;

void EnableInferenceMode() {
  enable_inference_mode = true;
}

void DisableInferenceMode() {
  enable_inference_mode = false;
}

void EnableQuantization() {
  enable_quantization = true;
}

void DisableQuantization() {
  enable_quantization = false;
}

bool IsQuantizationEnabled() {
  return enable_quantization;
}

void EnableConstSectionSerialization(
    const char* path,
    bool clear_path,
    bool use_compression) {
  const_section_serialize_path = std::string(path);
  clear_const_section_path = clear_path;
  enable_compression = use_compression;
}

bool IsInferenceMode() {
  return enable_inference_mode;
}

std::string GetConstSectionSerializationPath() {
  return habana_helpers::const_section_serialize_path;
}

bool IsConstSectionSerialization() {
  return habana_helpers::const_section_serialize_path != "";
}

bool ShouldClearConstSectionPath() {
  return habana_helpers::clear_const_section_path;
}

bool IsCompressionEnabled() {
  return habana_helpers::enable_compression;
}

bool enable_matmul3d_2d_reshape{GET_ENV_FLAG_NEW(PT_HPU_MATMUL3D_2D_RESHAPE)};

void EnableMatmul3d2dReshape() {
  enable_matmul3d_2d_reshape = true;
}

void DisableMatmul3d2dReshape() {
  enable_matmul3d_2d_reshape = false;
}

bool IsMatmul3d2dReshapeEnabled() {
  return enable_matmul3d_2d_reshape;
}

void enableRecomputeFSDPA(bool recompute) {
  enabled_recomputeFSDPA = recompute;
}

bool isRecomputeFSDPAEnabled() {
  return enabled_recomputeFSDPA;
}
} // namespace habana_helpers
