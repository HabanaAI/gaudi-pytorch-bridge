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
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch{

namespace jit{

TORCH_API void HabanaFuseGraph(std::shared_ptr<torch::jit::Graph>& graph);

Symbol getHabanaFusedOpSymbol();

}

}//namespace torch


