/*******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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
#include "splitter.h"
namespace habana {
namespace program {

namespace {

using Node = torch::jit::Node;
using Value = torch::jit::Value;

/*
 * Auxiliary structures describes cluster being built.
 */
struct Partition {
  Partition() {
    lazy_graph_->set_cached_graph(graph_);
  }
  // Maps nodes from original graph to their reflexions in this partition
  std::unordered_map<const Node*, Node*> mapping_;
  std::shared_ptr<LazyJitGraph> lazy_graph_ = std::make_shared<LazyJitGraph>();
  std::shared_ptr<torch::jit::Graph> graph_ =
      std::make_shared<torch::jit::Graph>();
};

/*
 * Implementation of splitting algorithm.
 *
 * Algorithm visits every node and edge in topological order and creates
 * new graphs according to given decision.
 */
struct SplitterImpl {
  SplitterImpl(const LazyJitGraph& graph, const SplittingDecision& decision)
      : lazy_graph_(graph),
        graph_(*graph.get_cached_graph()),
        decision_(decision) {}

  SplittingResult Run() {
    PT_BRIDGE_WARN("Run");
    MapSpecialColors();
    for (auto* node : graph_.nodes()) {
      VisitNode(node);
    }
    VisitNode(graph_.return_node());
    PT_BRIDGE_WARN("after Run");
    return std::move(result_);
  }

  /*
   * Map node to new partition and visit input edges.
   * Since we are traversing graph in topological order, the input nodes
   * are already mapped to partitions.
   */
  void VisitNode(const Node* node) {
    node->print(std::cout, 0, {});

    auto node_color = GetColor(node);
    if (not IsSpecialColor(node_color)) {
      MapNode(node_color, node);
    }

    auto inputs = node->inputs();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      auto input = inputs[i];
      auto input_color = GetColor(input);
      VisitEdge(i, node, node_color, input, input_color);
    }
  }

  /*
   * Map edge to new partitions.
   *
   * This procedure consider cases and does dispatch to specialized
   * visitors, just to keep code clean.
   */
  void VisitEdge(
      std::size_t dst_input_index,
      const Node* dst,
      std::int64_t dst_color,
      const Value* src,
      std::int64_t src_color) {
    (void)dst_input_index;
    if (IsSpecialColor(dst_color) and IsSpecialColor(src_color)) {
      return VisitEdge_SpecialToSpecial(
          dst_input_index, dst, dst_color, src, src_color);
    }
    if (IsSpecialColor(dst_color) and not IsSpecialColor(src_color)) {
      return VisitEdge_ClusterToSpecial(
          dst_input_index, dst, dst_color, src, src_color);
    }
    if (not IsSpecialColor(dst_color) and IsSpecialColor(src_color)) {
      return VisitEdge_SpecialToCluster(
          dst_input_index, dst, dst_color, src, src_color);
    }
    if (not IsSpecialColor(dst_color) and not IsSpecialColor(src_color)) {
      if (dst_color == src_color) {
        return VisitEdge_IntraClusterToCluster(
            dst_input_index, dst, src, src_color);
      } else {
        return VisitEdge_InterClusterToCluster(
            dst_input_index, dst, dst_color, src, src_color);
      }
    }
  }

  /*
   * Handle edge between special nodes (input -> return_node?).
   */
  void VisitEdge_SpecialToSpecial(
      std::size_t dst_input_index,
      const Node* dst,
      std::int64_t dst_color,
      const Value* src,
      std::int64_t src_color) {
    (void)dst_input_index;
    (void)dst;
    (void)dst_color;
    (void)src;
    (void)src_color;
    // TODO
  }

  /*
   * Handle edge between special node to regular node (input -> node?).
   */
  void VisitEdge_SpecialToCluster(
      std::size_t dst_input_index,
      const Node* dst,
      std::int64_t dst_color,
      const Value* src,
      std::int64_t src_color) {
    (void)dst_input_index;
    (void)dst;
    (void)dst_color;
    (void)src;
    (void)src_color;
  }

  /*
   * Handle edge between regular node and special node (node -> return_node?).
   */
  void VisitEdge_ClusterToSpecial(
      std::size_t dst_input_index,
      const Node* dst,
      std::int64_t dst_color,
      const Value* src,
      std::int64_t src_color) {
    (void)dst_input_index;
    (void)dst;
    (void)dst_color;
    (void)src;
    (void)src_color;
  }

  /*
   * Handle edge between regular nodes that falls into different partitions.
   */
  void VisitEdge_InterClusterToCluster(
      std::size_t dst_input_index,
      const Node* dst,
      std::int64_t dst_color,
      const Value* src,
      std::int64_t src_color) {
    (void)dst_input_index;
    (void)dst;
    (void)dst_color;
    (void)src;
    (void)src_color;

    auto mapped_dst = MapNode(dst_color, dst);
    auto mapped_src = MapNode(src_color, src->node());
    (void)mapped_dst;
    (void)mapped_src;
  }

  /*
   * Handle edge between regular nodes that falls into same partition.
   */
  void VisitEdge_IntraClusterToCluster(
      std::size_t dst_input_index,
      const Node* dst,
      const Value* src,
      std::int64_t color) {
    (void)dst;
    (void)src;
    (void)color;

    auto mapped_dst = MapNode(color, dst);
    auto mapped_src = MapNode(color, src->node());
    auto mapped_src_output = mapped_src->output(src->offset());
    // Make sure indices are consistent
    TORCH_CHECK(mapped_dst->inputs().size() == dst_input_index);
    mapped_dst->addInput(mapped_src_output);
  }

  std::int64_t GetColor(const Node* node) {
    auto it = decision_.colors.find(node);
    if (it == decision_.colors.end())
      return -1;
    return it->second;
  }

  std::int64_t GetColor(const Value* value) {
    auto node_from_value = value->node();
    if (node_from_value)
      return GetColor(node_from_value);
    return -1;
  }

  static bool IsSpecialColor(std::int64_t i) {
    return i < 0;
  }

  /*
   * Get mapped node.
   * If mapping does not exists, then create one.
   */
  Node* MapNode(std::int64_t color, const Node* orig_node) {
    TORCH_CHECK(not IsSpecialColor(color));
    auto& partition = color2partition[color];
    auto it = partition.mapping_.find(orig_node);
    if (it == partition.mapping_.end()) {
      auto new_node = partition.graph_->create(
          orig_node->kind(), orig_node->outputs().size());
      partition.mapping_[orig_node] = new_node;
      return new_node;
    }
    return it->second;
  }

  /*
   * Extends decision by mapping for special nodes
   */
  void MapSpecialColors() {
    for (auto* input : graph_.inputs()) {
      auto input_node = input->node();
      if (input_node) {
        decision_.colors[input_node] = -1;
      }
    }
    decision_.colors[graph_.return_node()] = -1;
  }

  SplittingResult result_;
  const LazyJitGraph& lazy_graph_;
  const torch::jit::Graph& graph_;
  SplittingDecision decision_;
  std::unordered_map<std::int64_t, Partition> color2partition;
};

} // namespace

SplittingResult SplitJitIrGraph(
    const LazyJitGraph& graph,
    const SplittingDecision& decision) {
  SplitterImpl algo(graph, decision);

  return algo.Run();
}

} // namespace program
} // namespace habana