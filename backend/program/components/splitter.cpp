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
#include "splitter.h"
namespace habana {
namespace program {

namespace {

using Node = torch::jit::Node;
using Value = torch::jit::Value;

/*
 * Auxiliary structure describing cluster being built.
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

  // Cache for mapped nodes into internal inputs
  std::unordered_map<const Value*, Value*> cluster_inputs_;
  // Cache for mapped nodes into internal outputs
  std::unordered_map<const Value*, std::size_t> cluster_outputs_;
  // Cache for mapped external inputs into internal inputs
  std::unordered_map<std::size_t, Value*> forwarded_inputs_;

  void FillResult(ClusterAfterSplitting& cluster) {
    auto num_outputs = graph_->outputs().size();
    cluster.graph_ = std::move(lazy_graph_);
    cluster.outputs_.resize(num_outputs);
  }
};

/*
 * Auxiliary structure tracking dependencies
 */
struct Dependencies {
  // Port = (color, input or output index)
  using PortSet = std::set<Port>;
  std::map<Port, PortSet> data_flow;

  void Add(
      std::int64_t src_color,
      std::size_t src_output_index,
      std::int64_t dst_color,
      std::size_t dst_input_index) {
    auto output_port = Port{src_color, src_output_index};
    auto input_port = Port{dst_color, dst_input_index};
    data_flow[output_port].insert(input_port);
  }

  void FillResult(SplittingResult& result) {
    for (auto& p : data_flow) {
      auto& output = p.first;
      if (output.cluster == COLOR_PARAM) {
        FillPorts(result.inputs_, output.index, p.second);
      } else {
        FillPorts(
            result.clusters_.at(output.cluster).outputs_,
            output.index,
            p.second);
      }
    }
  }

  void FillPorts(
      std::vector<PortVector>& target,
      std::size_t index,
      const PortSet& source) {
    if (target.size() <= index) {
      target.resize(index + 1);
    }
    target[index] = PortVector(source.begin(), source.end());
  }
};

bool isPrimConstantNone(const Node* node) {
  if (node->kind() != torch::jit::prim::Constant) {
    return false;
  }

  return node->output(0)->type() == torch::NoneType::get();
}

[[maybe_unused]] bool isPrimConstantNone(const Value* value) {
  return isPrimConstantNone(value->node());
}

bool isPrimConstant(const Node* node) {
  return node->kind() == torch::jit::prim::Constant;
}

bool isPrimConstant(const Value* value) {
  return isPrimConstant(value->node());
}

/*
 * Implementation of splitting algorithm.
 *
 * Algorithm visits every node and edge in topological order and creates
 * new graphs according to given decision.
 *
 * Main idea behind algorithm is to traverse original graph and
 * reflect every node and edge in partition -- according to node
 * color.
 *
 * To make implementation simple and consistent we follow simple principle,
 * that every mapping routine (MapInput, MapNode, MapValue, ...) gets as
 * input node/value from original graph.
 *
 * Original Input/Output is called `external`.
 * Partition's Input/Output is called `internal`.
 */
struct SplitterImpl {
  SplitterImpl(const LazyJitGraph& graph, const SplittingDecision& decision)
      : lazy_graph_(graph),
        graph_(*graph.get_cached_graph()),
        decision_(decision) {}

  SplittingResult Run() {
    PT_BRIDGE_DEBUG("Run");
    MapSpecialColors();
    for (auto* node : graph_.nodes()) {
      VisitNode(node);
    }
    VisitNode(graph_.return_node());
    PT_BRIDGE_DEBUG("after Run");
    PT_BRIDGE_DEBUG("------------------------------------");
    for (auto& p : color2partition) {
      PT_BRIDGE_DEBUG("=== color=", p.first);
      // p.second.graph_->print(std::cout, false);
    }
    PT_BRIDGE_DEBUG("------------------------------------");

    // for (auto& p : dependencies_.data_flow) {
    //  auto& output = p.first;
    //  for (auto& input : p.second) {
    //    std::cout << output.toString() << " -> " << input.toString() << "\n";
    //  }
    //}

    FillResult();
    return std::move(result_);
  }

  /*
   * Map node to new partition and visit input edges.
   * Since we are traversing graph in topological order, the input nodes
   * are already mapped to partitions.
   */
  void VisitNode(const Node* node) {
    // std::cout << "VisitNode color=" << GetColor(node) << " ptr=" << node << "
    // "; node->print(std::cout, 0, {});

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
   * This procedure consider cases and dispatches control to specialized
   * visitors, just to keep code clean.
   */
  void VisitEdge(
      std::size_t dst_input_index,
      const Node* dst,
      std::int64_t dst_color,
      const Value* src,
      std::int64_t src_color) {
    // std::cout << "\tVisitEdge src_color=" << src_color
    //          << " offset=" << src->offset() << " value_ptr=" << src
    //          << " node_ptr=" << src->node() << " ";
    // src->node()->print(std::cout, 0, {});
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

    throw std::runtime_error("unreachable");
  }

  /*
   * Handle edge between special nodes (input -> return_node).
   *
   * Just tracks data flow between external input and output.
   */
  void VisitEdge_SpecialToSpecial(
      std::size_t dst_input_index,
      const Node* dst,
      std::int64_t dst_color,
      const Value* src,
      std::int64_t src_color) {
    TORCH_CHECK_EQ(src_color, COLOR_PARAM);
    TORCH_CHECK_EQ(dst_color, COLOR_RETURN);
    TORCH_CHECK_EQ(dst, graph_.return_node());
    TORCH_CHECK_EQ(src->node(), graph_.param_node());

    dependencies_.Add(src_color, src->offset(), dst_color, dst_input_index);
  }

  /*
   * Handle edge between special node to regular node, it can be:
   * - input -> node
   * - None -> node
   *
   * Maps external input into internal one and connects to current node.
   */
  void VisitEdge_SpecialToCluster(
      std::size_t dst_input_index,
      const Node* dst,
      std::int64_t dst_color,
      const Value* src,
      std::int64_t src_color) {
    Node* mapped_dst = nullptr;
    Value* mapped_src = nullptr;
    if (src_color == COLOR_PARAM) {
      TORCH_CHECK_EQ(src->node(), graph_.param_node());
      mapped_dst = MapNode(dst_color, dst);
      mapped_src = MapExternalInput(dst_color, src);
    } else if (src_color == COLOR_CONSTANT_FOR_DUPLICATION) {
      TORCH_CHECK(isPrimConstant(src));
      // Duplicate None in destination partition
      mapped_src = MapConstant(dst_color, src);
      mapped_dst = MapNode(dst_color, dst);
    } else {
      throw std::runtime_error("VisitEdge_SpecialToCluster invalid usage");
    }

    // Make sure indices are consistent
    TORCH_CHECK(mapped_dst->inputs().size() == dst_input_index);
    auto new_value = mapped_dst->addInput(mapped_src);
    new_value->setType(mapped_src->type());
  }

  /*
   * Handle edge between regular node and special node (node -> return_node).
   *
   * Connects mapped node into partition's return_node and records data flow
   * from internal output to external output.
   */
  void VisitEdge_ClusterToSpecial(
      std::size_t dst_input_index,
      const Node* dst,
      std::int64_t dst_color,
      const Value* src,
      std::int64_t src_color) {
    TORCH_CHECK_EQ(dst_color, COLOR_RETURN);
    TORCH_CHECK_EQ(dst, graph_.return_node());

    auto output_index = MapInternalOutput(src_color, src);
    dependencies_.Add(src_color, output_index, COLOR_RETURN, dst_input_index);
  }

  /*
   * Handle edge between regular nodes that falls into different partitions.
   *
   * Maps input node as partition's internal input and connects to current
   * node.
   */
  void VisitEdge_InterClusterToCluster(
      std::size_t dst_input_index,
      const Node* dst,
      std::int64_t dst_color,
      const Value* src,
      std::int64_t src_color) {
    auto mapped_src = MapInternalInputFromCluster(dst_color, src_color, src);
    auto mapped_dst = MapNode(dst_color, dst);
    TORCH_CHECK(mapped_dst->inputs().size() == dst_input_index);
    auto new_value = mapped_dst->addInput(mapped_src);
    new_value->setType(mapped_src->type());
  }

  /*
   * Handle edge between regular nodes that falls into same partition.
   *
   * Just connects mapped nodes.
   */
  void VisitEdge_IntraClusterToCluster(
      std::size_t dst_input_index,
      const Node* dst,
      const Value* src,
      std::int64_t color) {
    // Map values from original graph into partition's graph
    auto mapped_dst = MapNode(color, dst);
    auto mapped_src = MapValue(color, src);
    // Make sure indices are consistent
    TORCH_CHECK(mapped_dst->inputs().size() == dst_input_index);
    auto new_value = mapped_dst->addInput(mapped_src);
    new_value->setType(mapped_src->type());
  }

  /*
   * Get color assigned to node by strategy.
   */
  std::int64_t GetColor(const Node* node) {
    auto it = decision_.colors.find(node);
    if (it == decision_.colors.end())
      throw std::runtime_error("No color for node");
    return it->second;
  }

  /*
   * Get color assigned to value by strategy.
   */
  std::int64_t GetColor(const Value* value) {
    auto node_from_value = value->node();
    if (node_from_value)
      return GetColor(node_from_value);
    throw std::runtime_error("No color for value");
  }

  /*
   * Small helper.
   */
  static bool IsSpecialColor(std::int64_t i) {
    return i < 0;
  }

  /*
   * Map node from original graph into partition.
   */
  Node* _InternalMapNode(
      std::int64_t color,
      const Node* orig_node,
      bool append = true) {
    auto* partition = GetPartition(color);
    auto it = partition->mapping_.find(orig_node);
    if (it == partition->mapping_.end()) {
      auto new_node = partition->graph_->create(
          orig_node->kind(), orig_node->outputs().size());
      partition->mapping_[orig_node] = new_node;
      new_node->copyAttributes(*orig_node);
      new_node->copyMetadata(const_cast<Node*>(orig_node));
      if (append) {
        partition->graph_->appendNode(new_node);
      } else {
        partition->graph_->prependNode(new_node);
      }

      for (std::size_t i = 0; i < orig_node->outputs().size(); ++i) {
        auto old_output = orig_node->output(i);
        auto new_output = new_node->output(i);
        new_output->setType(old_output->type());
      }

      return new_node;
    }
    return it->second;
  }

  /*
   * Map node from original graph into partition.
   */
  Node* MapNode(std::int64_t color, const Node* orig_node) {
    TORCH_CHECK(not IsSpecialColor(color));
    return _InternalMapNode(color, orig_node);
  }

  /*
   * Map value from original graph into partition.
   */
  Value* MapValue(std::int64_t color, const Value* orig_value) {
    auto node = MapNode(color, orig_value->node());
    auto value = node->output(orig_value->offset());
    value->setDebugName("v_" + orig_value->debugName());
    return value;
  }

  /*
   * Map value from original graph into partition.
   */
  Node* MapConstant(std::int64_t color, const Node* orig_node) {
    TORCH_CHECK(isPrimConstant(orig_node));
    TORCH_CHECK(not IsSpecialColor(color));
    return _InternalMapNode(color, orig_node, false);
  }

  /*
   * Map value from original graph into partition.
   */
  Value* MapConstant(std::int64_t color, const Value* orig_value) {
    auto node = MapConstant(color, orig_value->node());
    auto value = node->output(orig_value->offset());
    value->setDebugName("v_" + orig_value->debugName());
    return value;
  }

  /*
   * Extends decision by mapping for special nodes
   */
  void MapSpecialColors() {
    for (auto* input : graph_.inputs()) {
      auto input_node = input->node();
      if (input_node) {
        decision_.colors[input_node] = COLOR_PARAM;
      }
    }
    decision_.colors[graph_.return_node()] = COLOR_RETURN;

    for (auto* node : graph_.nodes()) {
      if (isPrimConstant(node)) {
        decision_.colors[node] = COLOR_CONSTANT_FOR_DUPLICATION;
      }
    }
  }

  /*
   * Gets partition associated to given color.
   */
  Partition* GetPartition(std::int64_t color) {
    auto it = color2partition.find(color);
    if (it == color2partition.end()) {
      color2partition[color] = Partition();
      return &color2partition[color];
    }
    return &it->second;
  }

  /*
   * Maps external input to internal input inside partition.
   */
  Value* MapExternalInput(int color, const Value* input) {
    auto partition = GetPartition(color);
    auto input_index = input->offset();

    auto it = partition->forwarded_inputs_.find(input_index);
    if (it != partition->forwarded_inputs_.end()) {
      return it->second;
    }

    auto new_value =
        partition->graph_->addInput("external_input_" + input->debugName());
    partition->forwarded_inputs_[input_index] = new_value;

    new_value->setType(input->type());

    dependencies_.Add(COLOR_PARAM, input->offset(), color, new_value->offset());
    return new_value;
  }

  /*
   * Maps value into partition's output and returns output index.
   */
  std::size_t MapInternalOutput(std::int64_t color, const Value* value) {
    auto partition = GetPartition(color);
    auto it = partition->cluster_outputs_.find(value);
    if (it != partition->cluster_outputs_.end()) {
      return it->second;
    }

    auto ret_node = partition->graph_->return_node();
    auto output_index = ret_node->inputs().size();
    ret_node->addInput(MapValue(color, value));
    partition->cluster_outputs_[value] = output_index;
    return output_index;
  }

  /*
   * Maps value from other partition into internal input in another partition.
   *
   * Connects source value to internal output inside source partition if it is
   * not already connected to output.
   *
   * Records dataflow between partitions.
   */
  Value* MapInternalInputFromCluster(
      std::int64_t dst_color,
      std::int64_t src_color,
      const Value* src_value) {
    auto dst_partition = GetPartition(dst_color);
    auto it = dst_partition->cluster_inputs_.find(src_value);
    if (it != dst_partition->cluster_inputs_.end()) {
      return it->second;
    }

    auto src_output_index = MapInternalOutput(src_color, src_value);

    auto mapped_value = dst_partition->graph_->addInput(
        "internal_input_" + src_value->debugName());
    dst_partition->cluster_inputs_[src_value] = mapped_value;
    mapped_value->setType(src_value->type());

    dependencies_.Add(
        src_color, src_output_index, dst_color, mapped_value->offset());
    return mapped_value;
  }

  /*
   * Fills final data structure from auxiliary structures.
   */
  void FillResult() {
    result_.inputs_.resize(graph_.inputs().size());
    for (auto& c2p : color2partition) {
      auto& partition = c2p.second;
      auto& cluster = result_.clusters_[c2p.first];
      partition.FillResult(cluster);
    }

    dependencies_.FillResult(result_);
  }

  SplittingResult result_;
  const LazyJitGraph& lazy_graph_;
  const torch::jit::Graph& graph_;
  SplittingDecision decision_;
  std::unordered_map<std::int64_t, Partition> color2partition;
  Dependencies dependencies_;
};

struct GocCreatorImpl {
  GocCreatorImpl(SplittingResult& splitting_result)
      : splitting_result_(splitting_result) {}

  std::unique_ptr<GraphOfClusters> Run() {
    CreateClusters();
    MapInputOutputs();
    return std::move(goc_);
  }

  void CreateClusters() {
    color2cluster_[COLOR_RETURN] = GraphOfClusters::SINK;
    for (auto& p : splitting_result_.clusters_) {
      auto color = p.first;
      auto cluster = goc_->CreateCluster();
      color2cluster_[color] = cluster->id_;
      PT_BRIDGE_DEBUG(
          "Copying color=",
          color,
          " id=",
          cluster->id_,
          " ",
          p.second.graph_.get());
      cluster->lazy_graph_ = std::move(p.second.graph_);
    }
  }

  void MapInputOutputs() {
    goc_->inputs_ = MapPortVector(splitting_result_.inputs_);
    for (auto& p : color2cluster_) {
      auto color = p.first;
      if (color == COLOR_RETURN)
        continue;
      auto cluster = goc_->FindCluster(p.second);
      cluster->outputs_ =
          MapPortVector(splitting_result_.clusters_.at(color).outputs_);
    }
  }

  Port MapPort(const Port& port) {
    return Port{color2cluster_.at(port.cluster), port.index};
  }

  PortVector MapPortVector(const PortVector& port_vector) {
    PortVector result;
    result.reserve(port_vector.size());
    for (auto& p : port_vector) {
      result.emplace_back(MapPort(p));
    }
    return result;
  }

  std::vector<PortVector> MapPortVector(const std::vector<PortVector>& xs) {
    std::vector<PortVector> result;
    result.reserve(xs.size());
    for (auto& x : xs) {
      result.emplace_back(MapPortVector(x));
    }
    return result;
  }

  std::unordered_map<std::int64_t, Cluster::Id> color2cluster_;
  std::unique_ptr<GraphOfClusters> goc_ = std::make_unique<GraphOfClusters>();
  SplittingResult& splitting_result_;
};

} // namespace

SplittingResult SplitJitIrGraph(
    const LazyJitGraph& graph,
    const SplittingDecision& decision) {
  SplitterImpl algo(graph, decision);

  return algo.Run();
}

std::unique_ptr<GraphOfClusters> CreateGraphOfClustersFromSplittingResult(
    SplittingResult& result) {
  return GocCreatorImpl(result).Run();
}

std::unique_ptr<GraphOfClusters> SplitIntoGoc(
    const LazyJitGraph& graph,
    const SplittingDecision& decision) {
  auto result = SplitJitIrGraph(graph, decision);
  return CreateGraphOfClustersFromSplittingResult(result);
}

} // namespace program
} // namespace habana