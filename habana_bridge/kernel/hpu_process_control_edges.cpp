#include "habana_bridge/kernel/hpu_habana_launch_op_pt.h"

#include <habana_device/hpu_cached_devices.h>
#include <synapse_helpers/device.h>
#include "hpu_ops/hpu_op_helper.h"

using namespace torch::jit;
using namespace habana;

bool HabanaLaunchOpPT::isControlEdge(torch::jit::Node* node) {
  bool is_controledge = false;

  auto node_str = node->kind().toQualString();

  if ((strcmp(node_str, "hpu::as_strided_lazy_") == 0) ||
      (strcmp(node_str, "hpu::as_strided_lazy_cl_") == 0) ||
      (strcmp(node_str, "hpu::control_edge_other_") == 0) ||
      (strcmp(node_str, "hpu::control_edge_") == 0)) {
    is_controledge = true;
  }

  return is_controledge;
}

bool HabanaLaunchOpPT::isInplace(torch::jit::Node* node) {
  bool is_inplace = false;

  if (!isControlEdge(node)) {
    auto node_name = node->kind().toQualString();

    size_t len = strlen(node_name);
    char endch = node_name[len - 1];

    if (endch == '_') {
      is_inplace = true;
    }
  }
  return is_inplace;
}

ControlEdgeType HabanaLaunchOpPT::nodeRequiresControlEdge(
    torch::jit::Node* node) {
  if (strcmp(node->kind().toQualString(), "hpu::control_edge_other_") == 0) {
    return ControlEdgeType::kCONTROL_EDGE_OTHER_;
  } else if (isControlEdge(node)) {
    return ControlEdgeType::kCONTROL_EDGE_;
  } else if (isInplace(node)) {
    return ControlEdgeType::kCONTROL_EDGE_INPLACE;
  } else {
    return ControlEdgeType::kCONTROL_EDGE_NONE;
  }
}

// Checks if it is a valid blocking or blocked node
// specifically eliminates control edges, prim:Param and prim::Return nodes
bool HabanaLaunchOpPT::IsValidNode(torch::jit::Node* blocking_node) {
  bool is_valid = true;

  // exclude control edges
  auto node_str = blocking_node->kind().toQualString();
  auto c_edge = nodeRequiresControlEdge(blocking_node);
  if (((c_edge == ControlEdgeType::kCONTROL_EDGE_) ||
       (c_edge == ControlEdgeType::kCONTROL_EDGE_OTHER_)) ||
      (strcmp(node_str, "prim::Param") == 0) ||
      (strcmp(node_str, "prim::Return") == 0)) {
    is_valid = false;
  }
  return is_valid;
}

void HabanaLaunchOpPT::addSynNodes(
    std::vector<synNodeId>& syn_node_vec,
    torch::jit::Node* node) {
  auto iter = jit_to_synapse_node_idx_map.find(node);
  if (iter != jit_to_synapse_node_idx_map.end()) {
    auto syn_node_idx = iter->second;
    syn_node_vec.insert(
        std::end(syn_node_vec),
        std::begin(syn_node_idx),
        std::end(syn_node_idx));
  }
}

bool HabanaLaunchOpPT::IsCustomOptimizer(std::string node_str) {
  auto it = std::find(
      custom_optimizer_nodestr_vec.begin(),
      custom_optimizer_nodestr_vec.end(),
      node_str);
  return (it != custom_optimizer_nodestr_vec.end());
}

// custom optimizer adds large number of synapse nodes (one per each learnable
// param in the model) If ProcessControlEdges is used naively then we end up
// adding large number of redundant blocked nodes thereby causing higher graph
// compile time We optimize using the fact that custom optimizer adds syn nodes
// in the same order as the tensor list inputs Blocking nodes corresponding to
// ith entry of input List construct can be paired only with ith synapse node of
// jit_to_synapse_node_idx_map["custom_opt"]
void HabanaLaunchOpPT::ProcessCustomOptControlEdges(
    torch::jit::graph_node_list graph_nodes) {
  // find the custom optimizer node
  for (auto node : graph_nodes) {
    auto node_str = node->kind().toQualString();
    if (IsCustomOptimizer(node_str)) {
      auto blocked_syn_nodes_set = jit_to_synapse_node_idx_map[node];

      // loop over all tensor list inputs
      for (auto in_val : node->inputs()) {
        blocking_syn_nodes_vec.clear();
        blocked_syn_nodes_vec.clear();

        if (strcmp(
                in_val->node()->kind().toQualString(), "prim::ListConstruct") ==
            0) {
          // check if inputs of ListConstruct is a  control edge
          auto list_idx = 0;
          for (auto list_input_val : in_val->node()->inputs()) {
            auto list_input_node = list_input_val->node();
            auto c_edge = nodeRequiresControlEdge(list_input_node);
            if (c_edge != ControlEdgeType::kCONTROL_EDGE_NONE) {
              // prepare blocking nodes list
              PrepareBlockingNodeList(list_input_node, c_edge);

              if (blocking_syn_nodes_vec.size()) {
                auto syn_node =
                    *std::next(blocked_syn_nodes_set.begin(), list_idx);
                blocked_syn_nodes_vec.emplace_back(syn_node);
                syn_graph_ptr->set_synapse_control_edges_pt(
                    blocking_syn_nodes_vec, blocked_syn_nodes_vec);
              }
            } // if (strcmp(list_input_node->kind().toQualString()...

            blocking_syn_nodes_vec.clear();
            blocked_syn_nodes_vec.clear();
            list_idx++;
          } // for (auto list_input_val: in_val->node()->inputs())
        } // if (strcmp(in_val->node()->kind().toQualString(),...
      } // for (auto in_val : node->inputs())

      // assuming there will be one custom optimizer at max in the graph
      break;
    } // if (IsCustomOptimizer(node_str))

  } // for (auto node : graph_nodes)
}

bool isListNode(torch::jit::Node* node) {
  auto node_str = node->kind().toQualString();
  bool is_list_node = false;
  if ((strcmp(node_str, "prim::ListUnpack") == 0) ||
      (strcmp(node_str, "prim::ListConstruct") == 0)) {
    is_list_node = true;
  }
  return is_list_node;
}

void HabanaLaunchOpPT::PrepareBlockingNodeList(
    Node* node,
    ControlEdgeType control_type) {
  int num_inputs =
      control_type == ControlEdgeType::kCONTROL_EDGE_OTHER_ ? 2 : 1;

  for (int i = 0; i < num_inputs; i++) {
    auto src_val = node->input(i);
    auto src_node_uses = src_val->uses();
    for (auto& u : src_node_uses) {
      auto blocking_node = u.user;

      // exclude current use in control_edge as well as parent node
      if (IsValidNode(blocking_node)) {
        // uses() api will include current node as well. Exclude it
        if (blocking_node != node) {
          blocking_nodes_vec.emplace_back(blocking_node);
          HabanaLaunchOpPT::addSynNodes(blocking_syn_nodes_vec, blocking_node);
        }
      }

    } // for (auto& u : src_node_uses)
  }

  // Add the parent node as well
  auto parent_node = node->input(0)->node();

  // if the parent node is a list node, traverse one level up
  if (isListNode(parent_node)) {
    parent_node = parent_node->input(0)->node();
  }

  // traverse up until a non control edge node is reached
  auto c_edge = nodeRequiresControlEdge(parent_node);
  while ((c_edge == ControlEdgeType::kCONTROL_EDGE_) ||
         (c_edge == ControlEdgeType::kCONTROL_EDGE_OTHER_)) {
    parent_node = parent_node->input(0)->node();
    c_edge = nodeRequiresControlEdge(parent_node);
  }

  // exclude invalid nodes like prim:Param, prim Return
  if (IsValidNode(parent_node)) {
    blocking_nodes_vec.emplace_back(parent_node);
    HabanaLaunchOpPT::addSynNodes(blocking_syn_nodes_vec, parent_node);
  }
}

void HabanaLaunchOpPT::Dfs(torch::jit::Node* node) {
  dfs_time_in_out_map[node].first = dfs_cnt++;

  for (auto& out : node->outputs()) {
    for (auto& u : out->uses()) {
      auto child_node = u.user;
      if (dfs_time_in_out_map.find(child_node) == dfs_time_in_out_map.end()) {
        Dfs(child_node);
      }
    }
  }

  dfs_time_in_out_map[node].second = dfs_cnt++;
}

// Preprocessing is done to compute in and out time time when the graph is
// traversed using DFS. These times will be used to determine
// ancester-descendant relationship between any pair of nodes. This relationship
// helps to avoid control edges induced graph cycles Specifically the blocked
// node should NOT be an ancestor of blocking node

void HabanaLaunchOpPT::PreprocessControlEdges() {
  PT_LAZY_TRACE;

  for (auto input_val : jit_ir_graph->inputs()) {
    // initialize the first and second values for prim::param input nodes
    dfs_time_in_out_map[input_val->node()].first = 0;
    dfs_time_in_out_map[input_val->node()].second = INT_MAX;
    for (auto& u : input_val->uses()) {
      auto node = u.user;

      if (dfs_time_in_out_map.find(node) == dfs_time_in_out_map.end()) {
        Dfs(node);
      }
    }
  }
}

// checks if any of the blocking nodes is an ancestor to the blocked node.
// this would create a control edge induced graph cycle and subsequently graph
// compile failure
bool HabanaLaunchOpPT::IsControlEdgeCycle(torch::jit::Node* blocked_node) {
  bool is_cycle = false;

  for (auto& blocking_node : blocking_nodes_vec) {
    // check if blocked node is an ancestor of blocking node
    HABANA_ASSERT(
        dfs_time_in_out_map.find(blocking_node) != dfs_time_in_out_map.end());
    HABANA_ASSERT(
        dfs_time_in_out_map.find(blocked_node) != dfs_time_in_out_map.end());

    if ((dfs_time_in_out_map[blocking_node].first >
         dfs_time_in_out_map[blocked_node].first) &&
        (dfs_time_in_out_map[blocking_node].second <
         dfs_time_in_out_map[blocked_node].second)) {
      is_cycle = true;
      PT_BRIDGE_DEBUG("control edge skipped as it introduces graph cycle")
      break;
    }
  }

  return is_cycle;
}

void HabanaLaunchOpPT::ProcessControlEdges() {
  PreprocessControlEdges();

  torch::jit::graph_node_list graph_nodes = jit_ir_graph->nodes();

  for (auto node : graph_nodes) {
    auto c_edge = nodeRequiresControlEdge(node);
    if (c_edge != ControlEdgeType::kCONTROL_EDGE_NONE) {
      // prepare blocking nodes list
      PrepareBlockingNodeList(node, c_edge);

      if (blocking_syn_nodes_vec.size()) {
        // prepare blocked nodes list
        if (c_edge == ControlEdgeType::kCONTROL_EDGE_INPLACE) {
          // if the current node is an inplace op, it becomes the blocked node
          HabanaLaunchOpPT::addSynNodes(blocked_syn_nodes_vec, node);
        } else {
          auto dst_node_uses = node->output(0)->uses();

          for (auto& u : dst_node_uses) {
            auto blocked_node = u.user;

            auto blocked_node_str = blocked_node->kind().toQualString();

            // special handling for listconstruct
            if (strcmp(blocked_node_str, "prim::ListConstruct") == 0) {
              auto blocked_node_uses = blocked_node->output(0)->uses();

              for (auto& l_u : blocked_node_uses) {
                blocked_node = l_u.user;

                if (IsValidNode(blocked_node)) {
                  blocked_node_str = blocked_node->kind().toQualString();

                  // skip the custom optimizer nodes as they are handled
                  // separately
                  if (strcmp(blocked_node_str, node->kind().toQualString()) &&
                      (!IsCustomOptimizer(blocked_node_str))) {
                    if (!IsControlEdgeCycle(blocked_node)) {
                      HabanaLaunchOpPT::addSynNodes(
                          blocked_syn_nodes_vec, blocked_node);
                    }
                  }
                }
              }
            } else {
              // exclude current use in control_edge
              if (strcmp(blocked_node_str, node->kind().toQualString())) {
                if (IsValidNode(blocked_node) &&
                    (!IsControlEdgeCycle(blocked_node))) {
                  HabanaLaunchOpPT::addSynNodes(
                      blocked_syn_nodes_vec, blocked_node);
                }
              }
            }
          } // for (auto& u : dst_node_uses)
        } // if (c_edge == ControlEdgeType::kCONTROL_EDGE_INPLACE)

        if (blocked_syn_nodes_vec.size()) {
          syn_graph_ptr->set_synapse_control_edges_pt(
              blocking_syn_nodes_vec, blocked_syn_nodes_vec);
        }
      }
      blocking_nodes_vec.clear();
    } // if (c_edge != ControlEdgeType::kCONTROL_EDGE_NONE)

    blocking_syn_nodes_vec.clear();
    blocked_syn_nodes_vec.clear();
  } // for (auto node : graph_nodes)

  // process dependencies for custom optimizer
  HabanaLaunchOpPT::ProcessCustomOptControlEdges(graph_nodes);
}
