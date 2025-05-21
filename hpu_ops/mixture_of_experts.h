/**
 * Copyright (c) 2024-2025 Intel Corporation
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

#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/op_backend.h"

namespace sh = synapse_helpers;

namespace habana {

struct MixtureOfExperts : OpBackend {
  MixtureOfExperts(
      int device_id,
      c10::ScalarType scalar_type,
      bool measurement_mode);
  void AddNode(sh::graph&, const at::Stack&) override;

 private:
  bool measurement_mode;
};
struct MixtureOfExpertsFwd : OpBackend {
  MixtureOfExpertsFwd(int device_id, c10::ScalarType scalar_type, bool recomp);
  void AddNode(sh::graph&, const at::Stack&) override;

 private:
  const bool recomp;
};

struct MixtureOfExpertsBwd : OpBackend {
  MixtureOfExpertsBwd(int device_id, c10::ScalarType scalar_type);
  void AddNode(sh::graph&, const at::Stack&) override;
};

struct MixtureOfExpertsRecompBwd : OpBackend {
  MixtureOfExpertsRecompBwd(int device_id, c10::ScalarType scalar_type);
  void AddNode(sh::graph&, const at::Stack&) override;
};

std::vector<std::vector<int64_t>> MixtureOfExpertsFwdShapes(const at::Stack&);

OutputMetaDataVector MixtureOfExpertsFp8Meta(const at::Stack& stack);
OutputMetaDataVector MixtureOfExpertsFwdMeta(const at::Stack& stack);
OutputMetaDataVector MixtureOfExpertsFwdRecompMeta(const at::Stack& stack);

OutputMetaDataVector MixtureOfExpertsBwdMeta(const at::Stack& stack);

} // namespace habana
