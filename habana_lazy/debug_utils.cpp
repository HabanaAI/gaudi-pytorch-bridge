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
#include <regex>
#include <sstream>
#include <unordered_map>

#include "absl/types/optional.h"

#include "aten_lazy_bridge.h"
#include "debug_utils.h"

namespace habana_lazy {

using NodeIdMap = std::unordered_map<ir::NodePtr, size_t>;

struct AttrTag {
  std::string name;
  std::string value;
  std::string::size_type pos;
};

std::string::size_type SkipTagSeparator(
    const std::string& node_string,
    std::string::size_type pos) {
  return node_string.compare(pos, 2, ", ") == 0 ? pos + 2 : pos;
}

absl::optional<AttrTag> ParseAttrTag(
    const std::string& node_string,
    std::string::size_type pos) {
  const std::regex tag_regex("^([a-zA-Z0-9_]+)=");
  std::smatch match;
  if (!std::regex_search(
          node_string.begin() + pos, node_string.end(), match, tag_regex)) {
    return absl::nullopt;
  }

  std::string::size_type vpos = match[1].second - node_string.begin() + 1;
  int nested_open = -1;
  int nested_close = -1;
  size_t nest_count = 1;
  AttrTag tag;
  tag.name = match[1].str();
  for (pos = vpos; pos < node_string.size(); ++pos) {
    if (nested_open < 0) {
      if (SkipTagSeparator(node_string, pos) != pos) {
        break;
      }
      switch (node_string[pos]) {
        case '(':
          nested_open = node_string[pos];
          nested_close = ')';
          break;
        case '[':
          nested_open = node_string[pos];
          nested_close = ']';
          break;
        case '{':
          nested_open = node_string[pos];
          nested_close = '}';
          break;
      }
    } else if (node_string[pos] == nested_close) {
      --nest_count;
      if (nest_count == 0) {
        nest_count = 1;
        nested_open = nested_close = -1;
      }
    } else if (node_string[pos] == nested_open) {
      ++nest_count;
    }
  }
  tag.value = node_string.substr(vpos, pos - vpos);
  tag.pos = pos;
  return tag;
}

NodeIdMap GenerateIdMap(const std::vector<ir::NodePtr>& post_order) {
  NodeIdMap id_map;
  for (auto& node : post_order) {
    id_map.emplace(node, id_map.size());
  }
  return id_map;
}

std::unordered_map<ir::NodePtr, size_t> GetRootsIds(
    const std::vector<ir::NodePtr>& roots) {
  std::unordered_map<ir::NodePtr, size_t> roots_ids;
  for (size_t i = 0; i < roots.size(); ++i) {
    roots_ids[roots[i]] = i;
  }
  return roots_ids;
}

absl::optional<size_t> GetRootNodeId(
    const ir::NodePtr& node,
    const std::unordered_map<ir::NodePtr, size_t>& roots_ids) {
  auto it = roots_ids.find(node);
  if (it == roots_ids.end()) {
    return absl::nullopt;
  }
  return it->second;
}

std::vector<AttrTag> GetNodeTags(const ir::NodePtr& node) {
  std::string node_string = node->ToString();
  std::string::size_type pos = node_string.find("\n");
  std::vector<AttrTag> tags;
  for (;;) {
    pos = SkipTagSeparator(node_string, pos + 1);
    auto tag = ParseAttrTag(node_string, pos);
    if (!tag) {
      break;
    }
    pos = tag->pos - 1;
    tags.push_back(std::move(*tag));
  }
  return tags;
}

std::string GenerateDotNodeLabel(
    const ir::NodePtr& node,
    const std::unordered_map<ir::NodePtr, size_t>& roots_ids,
    const bool use_ir_names) {
  static const size_t kMaxValueSize = 64;
  std::stringstream ss;
  if (use_ir_names) {
    auto num_outputs = node->GetNumOutputs();
    for (auto id = 0u; id < num_outputs; ++id) {
      ss << node->GetOutput(id).ToString() << "\\n";
    }
  }
  ss << node->op().toQualString() << "\\n" /*<< node->shape()*/;
  for (auto& tag : GetNodeTags(node)) {
    ss << "\\n" << tag.name << "=";
    if (tag.value.size() < kMaxValueSize) {
      ss << tag.value;
    } else {
      ss << tag.value.substr(0, kMaxValueSize) << "...";
    }
  }
  auto opt_root_id = GetRootNodeId(node, roots_ids);
  if (opt_root_id) {
    ss << "\\nROOT=" << *opt_root_id;
  }
  return ss.str();
}

std::string GenerateDotNodeSpec(
    const ir::NodePtr& node,
    const std::unordered_map<ir::NodePtr, size_t>& roots_ids,
    const bool use_ir_names) {
  std::stringstream ss;
  ss << "label=\"" << GenerateDotNodeLabel(node, roots_ids, use_ir_names)
     << "\"";
  return ss.str();
}

std::string GenerateTextNodeSpec(
    const ir::NodePtr& node,
    const NodeIdMap& id_map) {
  std::stringstream ss;
  ss << /*node->shape() << " " <<*/ node->op().toQualString() << "(";
  size_t count = 0;
  for (auto& output : node->GetInputs()) {
    if (count > 0) {
      ss << ", ";
    }
    ss << "%" << id_map.at(output.mp_node);
    if (output.mp_node->GetNumOutputs() > 1) {
      ss << "." << output.GetIndex();
    }
    ++count;
  }
  ss << ")";
  for (auto& tag : GetNodeTags(node)) {
    ss << ", " << tag.name << "=" << tag.value;
  }
  return ss.str();
}

std::string IrGraphDumpUtil::ToDot(std::vector<ir::NodePtr> nodes) {
  habana_lazy::ir::PostOrderData po_data;
  ir::Utils::ComputePostOrder(nodes, po_data);
  return PostOrderToDot(po_data.post_order, nodes, false);
}

std::string IrGraphDumpUtil::PostOrderToDot(
    const std::vector<ir::NodePtr>& post_order,
    const std::vector<ir::NodePtr>& roots,
    const bool use_ir_names) {
  std::unordered_map<ir::NodePtr, size_t> roots_ids = GetRootsIds(roots);
  NodeIdMap id_map = GenerateIdMap(post_order);
  std::stringstream ss;
  ss << "digraph G {\n";
  for (auto& node : post_order) {
    ss << "  node" << id_map.at(node) << " ["
       << GenerateDotNodeSpec(node, roots_ids, use_ir_names) << "]\n";
  }
  for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
    ir::NodePtr node = *it;
    size_t id = id_map.at(node);
    const auto& node_ips = node->GetInputs();
    for (size_t i = 0; i < node_ips.size(); ++i) {
      const auto& output = node_ips[i];
      ss << "  node" << id_map.at(output.mp_node) << " -> node" << id;
      if (node_ips.size() > 1) {
        ss << " [label=\"i=" << i;
        if (output.mp_node->GetNumOutputs() > 1) {
          ss << ",o=" << output.GetIndex();
        }
        ss << "\"]\n";
      } else {
        if (output.mp_node->GetNumOutputs() > 1) {
          ss << " [label=\"o=" << output.GetIndex() << "\"]";
        }
        ss << "\n";
      }
    }
  }
  ss << "}\n";
  return ss.str();
}

std::string IrGraphDumpUtil::ToText(std::vector<ir::NodePtr> nodes) {
  habana_lazy::ir::PostOrderData po_data;
  ir::Utils::ComputePostOrder(nodes, po_data);
  return PostOrderToText(po_data.post_order, nodes, false);
}

std::string IrGraphDumpUtil::PostOrderToText(
    const std::vector<ir::NodePtr>& post_order,
    const std::vector<ir::NodePtr>& roots,
    const bool use_ir_names,
    const bool print_ir_graph_info) {
  PT_LAZY_TRACE;
  std::unordered_map<ir::NodePtr, size_t> roots_ids = GetRootsIds(roots);
  NodeIdMap id_map = GenerateIdMap(post_order);
  std::stringstream ss;
  ss << "IR {\n";
  for (auto& node : post_order) {
    auto opt_root_id = GetRootNodeId(node, roots_ids);
    if (use_ir_names) {
      ss << "  ";
      auto num_outputs = node->GetNumOutputs();
      for (auto id = 0u; id < num_outputs; ++id) {
        ss << " %" << node->GetOutput(id).ToString();
        if (id == num_outputs - 1) {
          ss << " = ";
        } else {
          ss << ",";
        }
      }
      // Replace the \n at the end of node op name with space
      std::string node_string =
          print_ir_graph_info ? node->ToStringIrGraph() : node->ToString();
      std::string::size_type pos = node_string.find("\n");
      if (pos != std::string::npos) {
        node_string[pos] = ' ';
      }
      ss << node_string;
    } else {
      ss << "  %" << id_map.at(node) << " = "
         << GenerateTextNodeSpec(node, id_map);
    }
    if (opt_root_id) {
      ss << ", ROOT=" << *opt_root_id;
    }
    ss << "\n";
  }
  ss << "}\n";
  return ss.str();
}

} // namespace habana_lazy
