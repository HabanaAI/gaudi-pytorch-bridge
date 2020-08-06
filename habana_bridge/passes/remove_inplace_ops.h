#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace habana {

using namespace torch::jit;
void RemoveInplaceOps(const std::shared_ptr<Graph>& graph);
} // namespace habana
