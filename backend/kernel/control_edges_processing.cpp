/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
#include "backend/kernel/control_edges_processing.h"
#include "backend/jitgraph_utils.h"

namespace {
/**
 * Distinguishes between various cases for control edges.
 */
enum class ControlEdgeType {
  /**
   * Is not a control edge.
   */
  None = 0,
  /**
   * Control edge represented with hpu::control_edge_ node
   */
  Default,
  /**
   * Control edge represented with hpu::control_edge_other_ node
   */
  Other,
  /**
   * Control edge for inplace op with inplace input at first position (index 0)
   */
  InplaceInput0,
  /**
   * Control edge for inplace op with inplace input at second position (index 1)
   */
  InplaceInput1,
};

/**
 * Checks whether control edge type is one of two possible in-place.
 *
 * @param cet Control edge type to check.
 *
 * @return Result of the check.
 */
inline bool IsControlEdgeTypeInplace(const ControlEdgeType cet) {
  switch (cet) {
    case ControlEdgeType::InplaceInput0:
    case ControlEdgeType::InplaceInput1:
      return true;
    default:
      return false;
  }
}

/**
 * Detects node control edge type basing on node type or its inplace inputs.
 *
 * @param node Node to check.
 *
 * @return Detected control edge type.
 */
ControlEdgeType NodeRequiresControlEdge(const torch::jit::Node* const node) {
  using namespace std::literals;
  if ("hpu::control_edge_other_"sv == node->kind().toQualString()) {
    return ControlEdgeType::Other;
  } else if (habana::control_edges::IsControlEdgeNode(node)) {
    return ControlEdgeType::Default;
  } else if (int inputId = jitgraph_utils::inplaceInputId(node); inputId >= 0) {
    return (inputId == 0) ? ControlEdgeType::InplaceInput0
                          : ControlEdgeType::InplaceInput1;
  } else {
    return ControlEdgeType::None;
  }
}

/**
 * Checks if it is a valid blocking or blocked node.
 *
 * Specifically eliminates control edges, prim:Param and prim::Return nodes.
 *
 * @param blocking_node Node to check.
 *
 * @return Results of the check.
 */
bool IsValidBlockingOrBlockedNode(const torch::jit::Node* const blocking_node) {
  using namespace std::literals;
  // exclude control edges
  auto node_str = std::string_view{blocking_node->kind().toQualString()};
  auto c_edge = NodeRequiresControlEdge(blocking_node);
  return not(
      ((c_edge == ControlEdgeType::Default) ||
       (c_edge == ControlEdgeType::Other)) ||
      (node_str == "prim::Param"sv) || (node_str == "prim::Return"sv));
}

/**
 * Adds synapse nodes related to specific JIT node to the list.
 *
 * @param[out] syn_node_vec List to add synapse nodes to.
 * @param node JIT node to collect for.
 * @param jit_to_synapse_node_idx_map Mapping of JIT nodes to synapse nodes.
 */
void AddSynNodes(
    std::vector<synNodeId>& syn_node_vec,
    torch::jit::Node* const node,
    const std::unordered_map<torch::jit::Node*, std::vector<synNodeId>>&
        jit_to_synapse_node_idx_map) {
  auto iter = jit_to_synapse_node_idx_map.find(node);
  if (iter != jit_to_synapse_node_idx_map.end()) {
    auto syn_node_idx = iter->second;
    syn_node_vec.insert(
        std::end(syn_node_vec),
        std::begin(syn_node_idx),
        std::end(syn_node_idx));
  }
}

/**
 * Checks whether given node name corresponds to custom optimizer.
 *
 * @param node_str Node name to check.
 *
 * @return Result of the check.
 */
bool IsCustomOptimizer(const std::string_view node_str) {
  using namespace std::literals;
  // TODO: Add all the optimizers.
  static constexpr std::array<std::string_view, 7>
      custom_optimizer_nodestr_vec = {
          "hpu::fused_clip_norm"sv,
          "hpu::optimizer_sgd_momentum"sv,
          "hpu::habanaOptimizerFusedAdagrad"sv,
          "hpu::habanaOptimizerAdamW"sv,
          "hpu::optimizer_adamw"sv,
          "hpu::optimizer_lamb_phase1"sv,
          "hpu::optimizer_lamb_phase2"sv};
  auto it = std::find(
      custom_optimizer_nodestr_vec.begin(),
      custom_optimizer_nodestr_vec.end(),
      node_str);
  return (it != custom_optimizer_nodestr_vec.end());
}
} // namespace

namespace habana::control_edges {
/**
 * Check if given node name corresponds to control edge node.
 *
 * @param Node name to check.
 *
 * @return Result of the check.
 */
bool IsControlEdgeNode(const torch::jit::Node* const node) {
  const auto node_str = std::string_view{node->kind().toQualString()};
  using namespace std::literals;

  return (
      node_str == "hpu::as_strided_lazy_"sv ||
      node_str == "hpu::as_strided_lazy_cl_"sv ||
      node_str == "hpu::control_edge_other_"sv ||
      node_str == "hpu::control_edge_"sv);
}

/**
 * Analyzes affinity for the forest of directed graphs starting with given
 * nodes.
 *
 * Assumption: Forest of graphs won't change during analysis.
 * Assumption: Single user, won't need to copy nor move.
 */
class GraphAffinityAnalyzer {
 public:
  using ArrayOfNodes = at::ArrayRef<torch::jit::Value*>;

  /**
   * Construct analyzer and preprocesses graph.
   *
   * @param graph
   */
  GraphAffinityAnalyzer(const torch::jit::Graph& graph);

  GraphAffinityAnalyzer(const GraphAffinityAnalyzer&) = delete;
  GraphAffinityAnalyzer& operator=(const GraphAffinityAnalyzer&) = delete;

  /**
   * Checks if any of the blocking nodes is an ancestor to the blocked node.
   * This would create a control edge induced graph cycle and subsequently graph
   * compile failure
   *
   * @param blocked_node Node to be checked.
   * @param blocking_nodes_vec Vector of nodes to check against for cycle.
   */
  bool IsControlEdgeCycle(
      const torch::jit::Node* const blocked_node,
      const std::vector<torch::jit::Node*>& blocking_nodes_vec) const;

  /**
   * Check if there is upstream or downstream affinity between two nodes.
   *
   * @param node1 First node.
   * @param node2 Second node.
   *
   * return Result of the check.
   */
  bool IsAncestorOrDescendant(
      const torch::jit::Node* const node1,
      const torch::jit::Node* const node2) const;

 private:
  /**
   * Node order as established by DFS algorithms.
   */
  struct DfsNodeOrder {
    /**
     * Entry order.
     */
    size_t in{0};
    /**
     * Exit order.
     */
    size_t out{0};
  };

  /**
   * Node order storage for each node in the forest of graphs.
   */
  std::unordered_map<const torch::jit::Node*, DfsNodeOrder> dfs_time_map_{};

  /**
   * Current order.
   */
  size_t dfs_cnt_{0};

  // Preprocessing is done to compute in and out time time when the graph is
  // traversed using DFS. These times will be used to determine
  // ancestor-descendant relationship between any pair of nodes. This
  // relationship helps to avoid control edges induced graph cycles Specifically
  // the blocked node should NOT be an ancestor of blocking node
  void PreprocessControlEdges(const torch::jit::Graph& graph);

  /**
   * Traversing the graph using DFS algorithm.
   *
   * Beware, this results in recursive calls.
   *
   * @param node Currently traversed node.
   */
  void Dfs(const torch::jit::Node* node);

  /**
   * Checks if node 1 is an ancestor of node2.
   */
  bool IsAncestor(
      const torch::jit::Node* const node1,
      const torch::jit::Node* const node2) const;
};

GraphAffinityAnalyzer::GraphAffinityAnalyzer(const torch::jit::Graph& graph) {
  PreprocessControlEdges(graph);
}

bool GraphAffinityAnalyzer::IsControlEdgeCycle(
    const torch::jit::Node* const blocked_node,
    const std::vector<torch::jit::Node*>& blocking_nodes_vec) const {
  for (auto& blocking_node : blocking_nodes_vec) {
    // check if blocked node is an ancestor of blocking node
    TORCH_CHECK(
        dfs_time_map_.find(blocking_node) != dfs_time_map_.end(),
        blocking_node,
        blocking_node->kind().toQualString());
    TORCH_CHECK(
        dfs_time_map_.find(blocked_node) != dfs_time_map_.end(),
        blocked_node,
        blocked_node->kind().toQualString());

    if ((dfs_time_map_.at(blocking_node).in >
         dfs_time_map_.at(blocked_node).in) &&
        (dfs_time_map_.at(blocking_node).out <
         dfs_time_map_.at(blocked_node).out)) {
      PT_BRIDGE_DEBUG("control edge skipped as it introduces graph cycle")
      return true;
    }
  }

  return false;
}

bool GraphAffinityAnalyzer::IsAncestorOrDescendant(
    const torch::jit::Node* const node1,
    const torch::jit::Node* const node2) const {
  return (IsAncestor(node1, node2) || IsAncestor(node2, node1));
}

void GraphAffinityAnalyzer::PreprocessControlEdges(
    const torch::jit::Graph& graph) {
  PT_LAZY_TRACE;

  for (auto input_val : graph.inputs()) {
    // initialize the first and second values for prim::param input nodes
    dfs_time_map_.insert(
        {input_val->node(), {0, std::numeric_limits<size_t>::max()}});
    for (auto& u : input_val->uses()) {
      auto node = u.user;

      if (dfs_time_map_.find(node) == dfs_time_map_.end()) {
        Dfs(node);
      }
    }
  }

  // handle ops like full and arange that dont take input tensors
  for (auto node : graph.nodes()) {
    if (dfs_time_map_.find(node) == dfs_time_map_.end()) {
      Dfs(node);
    }
  }
}

void GraphAffinityAnalyzer::Dfs(const torch::jit::Node* node) {
  dfs_time_map_[node].in = dfs_cnt_++;

  for (auto& out : node->outputs()) {
    for (auto& u : out->uses()) {
      auto child_node = u.user;
      if (dfs_time_map_.find(child_node) == dfs_time_map_.end()) {
        Dfs(child_node);
      }
    }
  }

  dfs_time_map_[node].out = dfs_cnt_++;
}

bool GraphAffinityAnalyzer::IsAncestor(
    const torch::jit::Node* const node1,
    const torch::jit::Node* const node2) const {
  return (dfs_time_map_.at(node1).in < dfs_time_map_.at(node2).in) &&
      (dfs_time_map_.at(node1).out > dfs_time_map_.at(node2).out);
}

/**
 * Processes control edges existing in given graph.
 *
 * Assumption: Graph won't change during analysis.
 * Assumption: Single user, won't need to copy nor move.
 */
class ControlEdgesProcessor {
 public:
  /**
   * Constructor.
   *
   * @param jit_ir_graph JIT Graph to process.
   * @param jig_graph_and_metadata Optimized JIT graph and its metadata.
   * @param jit_to_synapse_node_idx_map Synapse node IDs for each JIT node.
   * @param memory_reuse_pairs List of mappings between values and nodes that
   can be used for memory reuse.
   * @param syn_graph_ptr Synapse graphs respective to JIT graph to which
   processing relates to.
   */
  ControlEdgesProcessor(
      torch::jit::Graph& jit_ir_graph,
      std::unordered_map<torch::jit::Node*, std::vector<synNodeId>>&
          jit_to_synapse_node_idx_map,
      std::vector<std::pair<torch::jit::Value*, torch::jit::Node*>>&
          memory_reuse_pairs,
      synapse_helpers::graph* const syn_graph_ptr)
      : jit_ir_graph_{jit_ir_graph},
        jit_to_synapse_node_idx_map_{jit_to_synapse_node_idx_map},
        memory_reuse_pairs_{memory_reuse_pairs},
        syn_graph_ptr_{syn_graph_ptr} {}

  ControlEdgesProcessor(const ControlEdgesProcessor&) = delete;
  ControlEdgesProcessor& operator=(const ControlEdgesProcessor&) = delete;

  /**
   * Performs very control edges processing.
   */
  bool ProcessControlEdges();

 private:
  /**
   * JIT Graph to process.
   * TODO: Make sure it is protected against concurrent accesses.
   */
  torch::jit::Graph&
      jit_ir_graph_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
  /**
   *  bool to store whether the control edges have been added or not
   */
  bool control_edge_has_been_added_ = false;

  /**
   * Synapse node IDs for each JIT node.
   */
  std::unordered_map<torch::jit::Node*, std::vector<synNodeId>>&
      jit_to_synapse_node_idx_map_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

  /**
   * List of mappings between values and nodes that can be used for memory
   * reuse.
   */
  std::vector<std::pair<torch::jit::Value*, torch::jit::Node*>>&
      memory_reuse_pairs_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

  /**
   * Synapse graphs respective to JIT graph to which processing relates to.
   */
  synapse_helpers::graph* const
      syn_graph_ptr_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

  /**
   * Optimization: Single field to save on allocations between usages. Passed
   * explicitly if used as out parameter.
   */
  std::vector<torch::jit::Node*> blocking_nodes_vec_;

  /**
   * Optimization: Single field to save on allocations between usages. Passed
   * explicitly if used as out parameter.
   */
  std::vector<synNodeId> blocking_syn_nodes_vec_;

  /**
   * Optimization: Single field to save on allocations between usages. Passed
   * explicitly if used as out parameter.
   */
  std::vector<synNodeId> blocked_syn_nodes_vec_;

  /**
   * Analyze users of given node and prepare its blocking nodes list basing on
   * control edge type.
   *
   * @param node Node to analyze.
   * @param control_type Control edge type for given node.
   * @param[out] blocking_nodes_vec List of blocking JIT nodes to append to.
   * @param[out] blocking_syn_nodes_vec List of blocking synapse nodes to append
   * to.
   */
  void PrepareBlockingNodeList(
      torch::jit::Node* node,
      ControlEdgeType control_type,
      std::vector<torch::jit::Node*>& blocking_nodes_vec,
      std::vector<synNodeId>& blocking_syn_nodes_vec);

  /**
   * Adds control edges for the cases when memory is used by different nodes of
   * the graph.
   *
   * This function expects that the information on pairs of nodes sharing memory
   * is available.
   *
   * @param affinity_analysis Affinity analysis that is to be used as processing
   * input.
   * @param[out] blocking_nodes_vec List of blocking nodes to add result to.
   */
  void ProcessControlEdgesForMemoryReuse(
      const habana::control_edges::GraphAffinityAnalyzer& affinity_analysis,
      std::vector<torch::jit::Node*>& blocking_nodes_vec);

  /**
   * Processes control edges related to custom ops.
   *
   * Custom optimizer adds large number of synapse nodes (one per each learnable
   * param in the model). If ProcessControlEdges is used naively then we end up
   * adding large number of redundant blocked nodes thereby causing higher graph
   * compile time. We optimize using the fact that custom optimizer adds syn
   * nodes in the same order as the tensor list inputs. Blocking nodes
   * corresponding to ith entry of input List construct can be paired only with
   * i-th synapse node of jit_to_synapse_node_idx_map["custom_opt"].
   *
   * @param graph_nodes List of graph nodes to process.
   */
  void ProcessCustomOptControlEdges(torch::jit::graph_node_list& graph_nodes);
};

void ControlEdgesProcessor::PrepareBlockingNodeList(
    torch::jit::Node* node,
    ControlEdgeType control_type,
    std::vector<torch::jit::Node*>& blocking_nodes_vec,
    std::vector<synNodeId>& blocking_syn_nodes_vec) {
  const int first_input =
      control_type == ControlEdgeType::InplaceInput1 ? 1 : 0;
  const int num_inputs = control_type == ControlEdgeType::Other ? 2 : 1;
  const int behind_last_input = first_input + num_inputs;

  for (int i = first_input; i < behind_last_input; i++) {
    const auto* const src_val = node->input(i);
    const auto& src_node_uses = src_val->uses();
    for (auto& u : src_node_uses) {
      auto* blocking_node = u.user;
      // Exclude current use in control_edge as well as parent node.
      if (IsValidBlockingOrBlockedNode(blocking_node)) {
        if (blocking_node == node) {
          // uses() api will include current node as well. Exclude it.
          continue;
        }
        if (jitgraph_utils::isListNode(blocking_node)) {
          auto* out_val = blocking_node->output(0);
          const auto& out_val_uses = out_val->uses();
          if (!out_val_uses.empty()) {
            blocking_node = out_val_uses[0].user;
          }
        }
        blocking_nodes_vec.emplace_back(blocking_node);
        AddSynNodes(
            blocking_syn_nodes_vec,
            blocking_node,
            jit_to_synapse_node_idx_map_);
      }
    }
  }

  /* Skip adding the parent node for inplace op as input tensor is not
  duplicated unlike a explicit control edge node. Avoiding explicit control edge
  nodes also avoids write after read issue in GC
    b.copy_(a)
    c = control_edge_(b)
    c.add_(1.0).
  Here, read of c needs to wait until b is written. If we avoid the control
  edges it comes
    b.copy_(a)
    b.add_(1.0)
  Here there is no need for control edges as  a is the parent of b
  */
  if (!IsControlEdgeTypeInplace(control_type)) {
    // Add the parent node as well.
    auto parent_node = node->input(0)->node();

    // Ff the parent node is a list node, traverse one level up.
    if (jitgraph_utils::isListNode(parent_node)) {
      parent_node = parent_node->input(0)->node();
    }

    // Traverse up until a non control edge node is reached.
    auto c_edge = NodeRequiresControlEdge(parent_node);
    while ((c_edge == ControlEdgeType::Default) ||
           (c_edge == ControlEdgeType::Other)) {
      parent_node = parent_node->input(0)->node();
      c_edge = NodeRequiresControlEdge(parent_node);
    }

    // Exclude invalid nodes like prim::Param, prim::Return.
    if (IsValidBlockingOrBlockedNode(parent_node)) {
      blocking_nodes_vec.emplace_back(parent_node);
      AddSynNodes(
          blocking_syn_nodes_vec, parent_node, jit_to_synapse_node_idx_map_);
    }
  }
}

void ControlEdgesProcessor::ProcessCustomOptControlEdges(
    torch::jit::graph_node_list& graph_nodes) {
  // Find the custom optimizer node.
  for (auto node : graph_nodes) {
    auto node_str = node->kind().toQualString();
    if (IsCustomOptimizer(node_str)) {
      auto blocked_syn_nodes_set = jit_to_synapse_node_idx_map_[node];

      using namespace std::literals;
      // Loop over all tensor list inputs.
      for (auto in_val : node->inputs()) {
        if (std::string_view{in_val->node()->kind().toQualString()} ==
            "prim::ListConstruct"sv) {
          // Check if inputs of ListConstruct is a  control edge.
          auto list_idx = 0;
          for (auto list_input_val : in_val->node()->inputs()) {
            auto list_input_node = list_input_val->node();
            auto c_edge = NodeRequiresControlEdge(list_input_node);
            if (c_edge != ControlEdgeType::None) {
              control_edge_has_been_added_ = true;
              // Prepare blocking nodes list.
              PrepareBlockingNodeList(
                  list_input_node,
                  c_edge,
                  blocking_nodes_vec_,
                  blocking_syn_nodes_vec_);

              if (blocking_syn_nodes_vec_.size()) {
                auto syn_node =
                    *std::next(blocked_syn_nodes_set.begin(), list_idx);
                blocked_syn_nodes_vec_.emplace_back(syn_node);
                syn_graph_ptr_->set_synapse_control_edges_pt(
                    blocking_syn_nodes_vec_, blocked_syn_nodes_vec_);
              }
            }

            blocking_syn_nodes_vec_.clear();
            blocked_syn_nodes_vec_.clear();
            list_idx++;
          }
        }
      }

      // Assuming there will be one custom optimizer at max in the graph.
      break;
    }
  }
}

bool ControlEdgesProcessor::ProcessControlEdges() {
  const habana::control_edges::GraphAffinityAnalyzer affinity_analysis{
      jit_ir_graph_};

  ProcessControlEdgesForMemoryReuse(affinity_analysis, blocking_nodes_vec_);

  torch::jit::graph_node_list graph_nodes = jit_ir_graph_.nodes();

  for (auto* const node : graph_nodes) {
    auto c_edge = NodeRequiresControlEdge(node);
    if (c_edge != ControlEdgeType::None) {
      control_edge_has_been_added_ = true;
      // Prepare blocking nodes list.
      PrepareBlockingNodeList(
          node, c_edge, blocking_nodes_vec_, blocking_syn_nodes_vec_);

      if (blocking_syn_nodes_vec_.size()) {
        // Prepare blocked nodes list.
        if (IsControlEdgeTypeInplace(c_edge)) {
          // If the current node is an inplace op, it becomes the blocked node.
          AddSynNodes(
              blocked_syn_nodes_vec_, node, jit_to_synapse_node_idx_map_);
        } else {
          auto dst_node_uses = node->output(0)->uses();

          for (auto& u : dst_node_uses) {
            auto blocked_node = u.user;

            auto blocked_node_str =
                std::string_view{blocked_node->kind().toQualString()};

            const auto node_str = std::string_view{node->kind().toQualString()};

            using namespace std::literals;
            // Special handling for ListConstruct.
            if (blocked_node_str == "prim::ListConstruct"sv) {
              auto blocked_node_uses = blocked_node->output(0)->uses();

              for (auto& l_u : blocked_node_uses) {
                blocked_node = l_u.user;

                if (IsValidBlockingOrBlockedNode(blocked_node)) {
                  blocked_node_str = blocked_node->kind().toQualString();

                  // Skip the custom optimizer nodes as they are handled
                  // separately.
                  if (blocked_node_str != node_str &&
                      (!IsCustomOptimizer(blocked_node_str))) {
                    if (!affinity_analysis.IsControlEdgeCycle(
                            blocked_node, blocking_nodes_vec_)) {
                      AddSynNodes(
                          blocked_syn_nodes_vec_,
                          blocked_node,
                          jit_to_synapse_node_idx_map_);
                    }
                  }
                }
              }
            } else {
              // Exclude current use in control_edge.
              if (blocked_node_str != node_str) {
                if (IsValidBlockingOrBlockedNode(blocked_node) &&
                    (!affinity_analysis.IsControlEdgeCycle(
                        blocked_node, blocking_nodes_vec_))) {
                  AddSynNodes(
                      blocked_syn_nodes_vec_,
                      blocked_node,
                      jit_to_synapse_node_idx_map_);
                }
              }
            }
          }
        }

        if (blocked_syn_nodes_vec_.size()) {
          syn_graph_ptr_->set_synapse_control_edges_pt(
              blocking_syn_nodes_vec_, blocked_syn_nodes_vec_);
        }
      }
      blocking_nodes_vec_.clear();
    }

    blocking_syn_nodes_vec_.clear();
    blocked_syn_nodes_vec_.clear();
  }

  // Process dependencies for custom optimizer.
  ProcessCustomOptControlEdges(graph_nodes);
  return control_edge_has_been_added_;
}

void ControlEdgesProcessor::ProcessControlEdgesForMemoryReuse(
    const habana::control_edges::GraphAffinityAnalyzer& affinity_analysis,
    std::vector<torch::jit::Node*>& blocking_nodes_vec) {
  for (const auto& p : memory_reuse_pairs_) {
    auto blocked_node = p.second;
    for (auto& u : p.first->uses()) {
      auto blocking_node = u.user;
      if (IsValidBlockingOrBlockedNode(blocking_node) &&
          (!affinity_analysis.IsAncestorOrDescendant(
               blocking_node, blocked_node) &&
           (blocking_node != blocked_node))) {
        // Set flag in JIT cache.
        control_edge_has_been_added_ = true;
        blocking_nodes_vec.emplace_back(blocking_node);
        AddSynNodes(
            blocking_syn_nodes_vec_,
            blocking_node,
            jit_to_synapse_node_idx_map_);
      }
    }

    if (blocking_syn_nodes_vec_.size()) {
      control_edge_has_been_added_ = true;
      AddSynNodes(
          blocked_syn_nodes_vec_, blocked_node, jit_to_synapse_node_idx_map_);
      if (blocked_syn_nodes_vec_.size()) {
        syn_graph_ptr_->set_synapse_control_edges_pt(
            blocking_syn_nodes_vec_, blocked_syn_nodes_vec_);
      }

      blocking_syn_nodes_vec_.clear();
      blocked_syn_nodes_vec_.clear();
    }
  }
}

bool IsNodeStridedInsertOrSliceInsert(const std::string_view node_qual_str) {
  // Condition below may look stange at first glance.
  //
  // hpu::strided_insert_ is new op defined in backend that works as any other
  // inplace op without necessity of special handling throughout the code.
  //
  // hpu::strided_insert is opposite. It is defined in legacy way and
  // thus requires special handling here and there.
  return (node_qual_str != "hpu::strided_insert_") &&
      ((node_qual_str.find("strided_insert") != std::string::npos) ||
       (node_qual_str.find("slice_insert") != std::string::npos));
}

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
        pt_to_synapse_tensors) {
  // Strided insert as graph output (i.e. persistence set as true).
  bool is_reuse_input = false;
  auto val_ins = node->inputs();
  auto node_qual_str = std::string_view{node->kind().toQualString()};

  // Check for unbroken chain of strided inserts from graph output to input.
  torch::jit::Node* input_node = node;
  using namespace std::literals;
  while (node_qual_str != "prim::Param"sv) {
    input_node = val_ins[0]->node();
    node_qual_str = std::string_view{input_node->kind().toQualString()};

    // Perform memory reuse if the chain has either strided/slice inserts or
    // inplace ops add control edges between consumers of inplace/ctrl edge
    // nodes and the last strided insert.
    if (!IsNodeStridedInsertOrSliceInsert(node_qual_str)) {
      if (NodeRequiresControlEdge(input_node) == ControlEdgeType::None) {
        break;
      } else {
        memory_reuse_pairs.emplace_back(
            std::make_pair(input_node->output(0), node));
      }
    }

    val_ins = input_node->inputs();
  }

  if (node_qual_str == "prim::Param"sv) {
    // Reached input with unbroken chain of strided inserts.
    is_reuse_input = true;
  }

  if (is_reuse_input == false) {
    OutputMetaDataVector md(1, outputs_metadata.at(0));
    md.at(0).persistent = true;
    habana_kernel->AllocateAndAddSynapseNode(*syn_graph, input_stack, md);
  } else {
    TORCH_CHECK(
        value_to_ivalue.count(val_ins[0]),
        "incorrect input for strided insert");
    const auto& ivalue = value_to_ivalue.at(val_ins[0]);
    TORCH_CHECK(
        pt_to_synapse_tensors.find(ivalue) != pt_to_synapse_tensors.end(),
        "incorrect ivalue for strided insert input");

    input_stack.insert(input_stack.end(), *ivalue);
    habana_kernel->ReuseMemoryAndAddSynapseNode(
        *syn_graph,
        input_stack,
        *pt_to_synapse_tensors.at(ivalue),
        outputs_metadata);

    /* Since memory is reused we need control edges between the consumers of the
    graph input (prim:param) and the strided insert at the graph output. Refer
    gtest LazyBasicKernelTest.allreducewithcontroledge. */

    // Book keep the node pair that reuses same memory.
    memory_reuse_pairs.emplace_back(std::make_pair(val_ins[0], node));
  }
}

bool ProcessControlEdges(
    torch::jit::Graph& jit_ir_graph,
    std::unordered_map<torch::jit::Node*, std::vector<synNodeId>>&
        jit_to_synapse_node_idx_map,
    std::vector<std::pair<torch::jit::Value*, torch::jit::Node*>>&
        memory_reuse_pairs,
    synapse_helpers::graph* const syn_graph_ptr) {
  return ControlEdgesProcessor{
      jit_ir_graph,
      jit_to_synapse_node_idx_map,
      memory_reuse_pairs,
      syn_graph_ptr}
      .ProcessControlEdges();
}

} // namespace habana::control_edges
