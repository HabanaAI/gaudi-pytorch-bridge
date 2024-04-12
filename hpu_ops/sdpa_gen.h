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
