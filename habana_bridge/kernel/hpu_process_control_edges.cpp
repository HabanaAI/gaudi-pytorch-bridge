#include "habana_bridge/kernel/hpu_habana_launch_op_pt.h"

using namespace torch::jit;

/*
Currently we exclude parents nodes from blocking list as they can affect
pipelining. Ideally we need to exclude all the ancestors. Revisit if current
approach results in performance issues. Alternateively check if GC can handle
do this exclusion
*/
bool HabanaLaunchOpPT::isBlockingNode(
    torch::jit::Node* blocking_node,
    torch::jit::Node* control_edge_node) {
  bool is_blocking = true;

  // exclude control edges or
  // if it is a parent node of blocked node
  // In JIT IR, the parent of blocked node would actually be parent of control
  // edge node
  auto node_str = blocking_node->kind().toQualString();
  if ((strcmp(node_str, "hpu::control_edge_") == 0) ||
      (control_edge_node->input(0)->node() == blocking_node)) {
    is_blocking = false;
  }
  return is_blocking;
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

            if (strcmp(
                    list_input_node->kind().toQualString(),
                    "hpu::control_edge_") == 0) {
              // prepare blocking nodes list
              PrepareBlockingNodeList(list_input_node);

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

void HabanaLaunchOpPT::PrepareBlockingNodeList(Node* node) {
  auto src_val = node->input(0);
  auto src_node_uses = src_val->uses();

  for (auto& u : src_node_uses) {
    auto blocking_node = u.user;

    // exclude current use in control_edge as well as parent node
    if (HabanaLaunchOpPT::isBlockingNode(blocking_node, node)) {
      HabanaLaunchOpPT::addSynNodes(blocking_syn_nodes_vec, blocking_node);
    }

  } // for (auto& u : src_node_uses)

  // Add the parent node as well
  auto parent_node = node->input(0)->node();
  HabanaLaunchOpPT::addSynNodes(blocking_syn_nodes_vec, parent_node);
}

void HabanaLaunchOpPT::ProcessControlEdges() {
  torch::jit::graph_node_list graph_nodes = subgraph_->nodes();

  for (auto node : graph_nodes) {
    if (strcmp(node->kind().toQualString(), "hpu::control_edge_") == 0) {
      // prepare blocking nodes list
      PrepareBlockingNodeList(node);

      if (blocking_syn_nodes_vec.size()) {
        // prepare blocked nodes list
        auto dst_node_uses = node->output(0)->uses();

        for (auto& u : dst_node_uses) {
          auto blocked_node = u.user;

          auto blocked_node_str = blocked_node->kind().toQualString();

          // special handling for listconstruct
          if (strcmp(blocked_node_str, "prim::ListConstruct") == 0) {
            auto blocked_node_uses = blocked_node->output(0)->uses();

            for (auto& l_u : blocked_node_uses) {
              blocked_node = l_u.user;
              blocked_node_str = blocked_node->kind().toQualString();

              // skip the custom optimizer nodes as they are handled separately
              if (strcmp(blocked_node_str, "hpu::control_edge_") &&
                  (!IsCustomOptimizer(blocked_node_str))) {
                HabanaLaunchOpPT::addSynNodes(
                    blocked_syn_nodes_vec, blocked_node);
              }
            }
          } else {
            // exclude current use in control_edge
            if (strcmp(blocked_node_str, "hpu::control_edge_")) {
              HabanaLaunchOpPT::addSynNodes(
                  blocked_syn_nodes_vec, blocked_node);
            }
          }
        } // for (auto& u : dst_node_uses)

        if (blocked_syn_nodes_vec.size()) {
          syn_graph_ptr->set_synapse_control_edges_pt(
              blocking_syn_nodes_vec, blocked_syn_nodes_vec);
        }
      }
    }

    blocking_syn_nodes_vec.clear();
    blocked_syn_nodes_vec.clear();
  } // for (auto node : graph_nodes)

  // process dependencies for custom optimizer
  HabanaLaunchOpPT::ProcessCustomOptControlEdges(graph_nodes);
}
