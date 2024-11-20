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

#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/op_backend.h"

namespace habana {

struct SDPAFwd : OpBackend {
  SDPAFwd(int device_id, c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

struct SDPABwd : OpBackend {
  SDPABwd(int device_id, c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};
struct Fp8SDPABwd : OpBackend {
  Fp8SDPABwd(int device_id, c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

struct SDPARecompFwd : OpBackend {
  SDPARecompFwd(int device_id, c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

struct Fp8SDPARecompFwd : OpBackend {
  Fp8SDPARecompFwd(int device_id, c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

struct Fp8SDPAFwd : OpBackend {
  Fp8SDPAFwd(int device_id, c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

struct SDPARecompBwd : OpBackend {
  SDPARecompBwd(int device_id, c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

OUTSHAPE_DECL(SDPAFwdOutputShape)
OUTSHAPE_DECL(Fp8SDPAFwdOutputShape)
OUTSHAPE_DECL(SDPABwdOutputShape)
OUTSHAPE_DECL(Fp8SDPABwdOutputShape)
OUTSHAPE_DECL(SDPARecompFwdOutputShape)
OUTSHAPE_DECL(Fp8SDPARecompFwdOutputShape)
OUTSHAPE_DECL(SDPARecompBwdOutputShape)

} // namespace habana
