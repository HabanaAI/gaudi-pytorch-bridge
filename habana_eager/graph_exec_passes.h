/*******************************************************************************
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

#include <torch/csrc/jit/ir/ir.h>

namespace habana {
namespace graph {
namespace pass {
void SanitizeGraphInput(std::shared_ptr<torch::jit::Graph> graph);
void HandleTupleOnOutput(std::shared_ptr<torch::jit::Graph> graph);
void AddAttributeAlpha(std::shared_ptr<torch::jit::Graph> graph);
void DetectWeightTensors(
    std::shared_ptr<torch::jit::Graph> graph,
    std::set<int>& graph_inputs_to_permute);
void GetOutputsOrderInGraph(
    std::shared_ptr<torch::jit::Graph> graph,
    std::vector<size_t>& outputs_order);
void ReplaceGetItemWithListUnpack(std::shared_ptr<torch::jit::Graph> graph);
bool HandleDynamicOps(
    std::shared_ptr<torch::jit::Graph> graph,
    torch::jit::Stack& stack,
    std::shared_ptr<DynamicGraphMetaData> dgraph_meta,
    std::map<int64_t, std::vector<int64_t>>* input_new_base_sizes);
void HandlePostDynamic(
    std::shared_ptr<DynamicGraphMetaData> dgraph_meta,
    std::map<int64_t, std::vector<int64_t>>& input_base_sizes_map);
void HandleDynamicInputPatching(
    torch::jit::Stack& stack,
    std::shared_ptr<DynamicGraphMetaData> dgraph_meta,
    LaunchDynamicShapes& launch_shapes,
    bool is_first_launch);
void ResolveNegativeSTSizes(
    std::shared_ptr<torch::jit::Graph> graph,
    torch::jit::Stack& stack,
    std::shared_ptr<DynamicGraphMetaData> dmeta,
    LaunchDynamicShapes& launch_shapes);
void RemoveDetachOp(std::shared_ptr<torch::jit::Graph> graph);
void HandleInputViews(
    std::shared_ptr<torch::jit::Graph> graph,
    torch::jit::Stack& example_inputs,
    std::map<int64_t, std::vector<int64_t>>& input_base_sizes_map);
} // namespace pass
} // namespace graph
} // namespace habana
