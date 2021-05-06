#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace habana {
void RemoveInplaceOps(const std::shared_ptr<torch::jit::Graph>& graph);
} // namespace habana
