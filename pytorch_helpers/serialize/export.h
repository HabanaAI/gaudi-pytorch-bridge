/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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
#include <torch/csrc/jit/ir/ir.h>
#include <string>

namespace serialize {

using Graph = torch::jit::Graph;
using GraphPtr = std::shared_ptr<Graph>;

// Converts a JIT IR to a serializable protobuf string (i.e. pbtxt).
std::string GraphToProtoString(const GraphPtr& graph);

} // namespace serialize