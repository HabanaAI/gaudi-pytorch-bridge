/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <regex>
#include <sstream>
#include <unordered_map>

#include "absl/types/optional.h"

#include "aten_lazy_bridge.h"
#include "debug_utils.h"
#include "tensor_comparator.hpp"

class float16;
class bfloat16;
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
      node_string[pos] = ' ';
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

#define TENSOR_COMPARE_TYPE(primitive_type) \
  success = tc.compare(                     \
      tensor_name,                          \
      (primitive_type*)hpu_data,            \
      (primitive_type*)cpu_data,            \
      cpu_res.numel(),                      \
      compare_method,                       \
      true);                                \
  break;

#define CASE_TENSOR_COMPARE_TYPE(scalar_type, primitive_type) \
  case scalar_type:                                           \
    TENSOR_COMPARE_TYPE(primitive_type)

// handling duplicate names (adding a counter suffix)
static std::string handle_duplicates(const std::string& op_type) {
  std::string tensor_name = op_type;
  static std::map<std::string, int> checked_tensors_occurences;
  auto it = checked_tensors_occurences.find(tensor_name);
  if (it != checked_tensors_occurences.end()) {
    PT_LAZY_DEBUG("Tensor name: ", tensor_name, " check number: ", it->second);
    tensor_name += "_" + std::to_string(it->second++);
  }
  checked_tensors_occurences[tensor_name] = 1;

  return tensor_name;
}

void SBSDebug::compare_tensors_cos(
    at::Tensor hpu_res,
    at::Tensor cpu_res,
    const std::string& op_type) {
  std::string tensor_name = handle_duplicates(op_type);

  static TensorComparison::TensorValidator tc;
  auto hpu_res_on_host = hpu_res.to("cpu");
  if (hpu_res_on_host.dtype() == cpu_res.dtype()) {
    auto scalarType = cpu_res.scalar_type();
    TensorComparison::ComparisonMethods compare_method;
    compare_method.set(); // all test methods

    void* hpu_data = hpu_res_on_host.data_ptr();
    void* cpu_data = cpu_res.data_ptr();
    bool success = true;
    switch (scalarType) {
      CASE_TENSOR_COMPARE_TYPE(c10::ScalarType::Byte, unsigned char)
      CASE_TENSOR_COMPARE_TYPE(c10::ScalarType::Char, signed char)
      CASE_TENSOR_COMPARE_TYPE(c10::ScalarType::Short, short)
      CASE_TENSOR_COMPARE_TYPE(c10::ScalarType::Long, long)
      CASE_TENSOR_COMPARE_TYPE(c10::ScalarType::Half, float16)
      CASE_TENSOR_COMPARE_TYPE(c10::ScalarType::Float, float)
      CASE_TENSOR_COMPARE_TYPE(c10::ScalarType::BFloat16, bfloat16)
      default:
        PT_LAZY_WARN(
            "sbs could not run on this op due to unsupported dtype. dtype: ",
            hpu_res_on_host.dtype());
        return;
    }

    if (!success) {
      PT_LAZY_WARN("Tensor Comparator failed to execute");
      return;
    }
    tc.makeReport(m_report_file_name, TensorComparison::ExportType::CSV);
  } else {
    PT_LAZY_WARN(
        "sbs could not run on this op due to different dtype. hpu dtype: ",
        hpu_res_on_host.dtype(),
        " cpu dtype:",
        cpu_res.dtype());
  }
}

void SBSDebug::CompareTensors(std::vector<HbLazyTensor>& tensors) {
  for (auto& hb_tensor : tensors) {
    at::Tensor at_tensor = AtenFromHbLazyTensor(hb_tensor);
    at_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(
        at_tensor.sizes()); // This is a work around bug fix, see [SW-66837]
    c10::optional<at::Tensor> cpu_ref = hb_tensor.GetCPUTensorData();
    if (cpu_ref == c10::nullopt) {
      PT_LAZY_DEBUG(
          "SBS: Tensor is live (comparison point), but has no CPU (SBS is not supported). Name: ",
          hb_tensor.CurrentIrValue().ToString())
    } else if (!hb_tensor
                    .GetSBSLiveTensorIndication()) // We'll avoid tensors that
                                                   // we've already checked
    {
      std::string name = hb_tensor.CurrentIrValue().ToString();
      if (name.empty()) {
        name = std::string("Op Name N/A. ID ") +
            std::to_string(hb_tensor.getTensorUniqueId());
      }
      PT_LAZY_DEBUG(
          "Comparing tensor. Name: ",
          name,
          ", ID: ",
          hb_tensor.getTensorUniqueId());

      compare_tensors_cos(at_tensor, cpu_ref.value(), name);
      hb_tensor.SetSBSLiveTensorIndication(); // Used by SBS modes 2 & 3
    }
  }
}

SBSDebug::SBSDebug() {
  PT_LAZY_DEBUG(
      "SBS: Tensor compare report will be saved to: ", m_report_file_name);
}

} // namespace habana_lazy
