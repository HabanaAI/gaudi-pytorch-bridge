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
#include "habana_lazy/hpu_lazy_cache.h"

#include <gtest/gtest.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/torch.h>

namespace habana_lazy_test {

typedef struct {
  habana_lazy::ir::NodePtrList post_order_nodes;
  std::string post_order_str;
} PostOrderTestStruct;

// Create a 3 Node vector from first level IR
// This is what is expected after a post order traversal
// of the first level IR
PostOrderTestStruct GetPostOrderNodes(bool jumbld = false);
// Create input IValues.
// tensor_shapes creates n tensors with given shapes.
// scalars creates m scalars with given value
std::vector<torch::jit::IValue> CreateInputs(
    std::vector<std::vector<int64_t>> tensor_shapes,
    std::vector<float> scalars);

std::shared_ptr<torch::jit::Graph> CreateJITGraph();

} // namespace habana_lazy_test
