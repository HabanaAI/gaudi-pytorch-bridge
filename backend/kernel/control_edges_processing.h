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

#include <torch/jit.h>
#include <unordered_map>
#include "backend/jit_graph_cache.h"

namespace habana::control_edges {
/**
 * Determines whether given node imply control edge.
 *
 * @param node Node to check.
 *
 * @return Result of the check.
 */
bool IsControlEdgeNode(const torch::jit::Node* const node);

/**
 * Determines whether given node is one of StridedInsert or SliceInsert.
 *
 * @param node_qual_str Node name to check.
 *
 * @return Result of the check.
 */
bool IsNodeStridedInsertOrSliceInsert(const std::string_view node_qual_str);

/**
 * Optimizes the memory usage for chain of strided inserts.
 *
 * Optimization is being done by reusing the input memory for the graph output.
 * Such a case is common in allreduce.
 *
 * @param node Node to be analyzed
 * @param habana_kernel HabanaKernel pointer for analyzed node
 * @param input_stack Stack for analyzed node
 * @param syn_graph Synapse Graph representation handle
 * @param output_metadata Metadata for node outputs
 * @param[out] List of found memory reuse possibilities.
 * @param value_to_ivalue Map between tensor values and ivalues
 * @param pt_to_synapse_tensors Map between Tensors and their synapse
 * representations
 */
void ProcessStridedInsertAtOutput(
    torch::jit::Node* node,
    HabanaOperatorPtr habana_kernel,
    torch::jit::Stack& input_stack,
    std::shared_ptr<synapse_helpers::graph>& syn_graph,
    const OutputMetaDataVector& outputs_metadata,
    std::vector<std::pair<torch::jit::Value*, torch::jit::Node*>>&
        memory_reuse_pairs,
    const CValuePtrToIValuePtrMap& value_to_ivalue,
    const std::unordered_map<IValPtrShared, SharedSynTensorOrRefListPtr>&
        pt_to_synapse_tensors);

/**
 * Processes control edges in graph and alter synapse graph accordingly.
 *
 * This is being done with multiple steps:
 * 1. Analyses possibilities for I/O memory reuse and adds respective control
 * edges to allow this.
 * 2. Adds required control edges for in-place inputs taking in consideration
 * possible cycles and special handling required for custom optimizers
 *
 * Performs single-threaded processing without further data sharing.
 * Assumes caller ensure passed data is valid during processing.
 *
 * TODO: Add / change params that will allow to ensure thread safety on
 * accessing shared params.
 *
 * @param jit_ir_graph
 * @param jit_to_synapse_node_idx_map
 * @param memory_reuse_pairs
 * @param syn_graph_ptr
 *
 * @return whether the control edges have been added
 *
 */
bool ProcessControlEdges(
    torch::jit::Graph& jit_ir_graph,
    std::unordered_map<torch::jit::Node*, std::vector<synNodeId>>&
        jit_to_synapse_node_idx_map,
    std::vector<std::pair<torch::jit::Value*, torch::jit::Node*>>&
        memory_reuse_pairs,
    synapse_helpers::graph* const syn_graph_ptr);

} // namespace habana::control_edges
