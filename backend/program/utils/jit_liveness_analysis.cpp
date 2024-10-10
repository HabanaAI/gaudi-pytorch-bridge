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

#include "jit_liveness_analysis.h"
#include <set>
#include <sstream>

namespace habana {
namespace program {
namespace utils {

namespace {

/*
 * Implementation of liveness analysis.
 *
 * Since we work with JIT IR graph that do not contain any control constructs,
 * the implementation of analysis simplifies to single pass algorithm.
 *
 * For every node N we have:
 *      uses(N) - set of values used by this node (de facto inputs)
 *      defs(N) - set of values defined by this node (de facto outputs)
 *
 * Algorithm:
 *      - algorithm maintains set of live values, initial value is
 * uses(RETURN_NODE)
 *      - traverse schedule backwards and for every visited node N...
 *          - record number of live bytes in bytes_live_after
 *          - eliminate defs(N) from live set
 *          - include uses(N) in live set
 *          - record number of live bytes in bytes_live_before
 */
class LivenessAnalysisImpl {
 public:
  LivenessAnalysisImpl(
      const LazyJitGraph& lazy_graph,
      const std::vector<const torch::jit::Node*>& schedule)
      : graph_(*lazy_graph.get_cached_graph()), schedule_(schedule) {
    result.bytes_alive_before.resize(schedule_.size());
    result.bytes_alive_after.resize(schedule_.size());
  }

  LivenessResult Run() {
    Analyse();
    return std::move(result);
  }

 private:
  std::set<const torch::jit::Value*> ComputeUses(const torch::jit::Node* node) {
    std::set<const torch::jit::Value*> xs;
    xs.insert(node->inputs().begin(), node->inputs().end());
    return xs;
  }

  std::set<const torch::jit::Value*> ComputeDefs(const torch::jit::Node* node) {
    std::set<const torch::jit::Value*> xs;
    xs.insert(node->outputs().begin(), node->outputs().end());
    return xs;
  }

  void Analyse() {
    auto live = ComputeUses(graph_.return_node());
    for (std::size_t _i = 0; _i < schedule_.size(); ++_i) {
      std::size_t i = schedule_.size() - _i - 1;
      auto node = schedule_[i];

      auto uses = ComputeUses(node);
      auto defs = ComputeDefs(node);

      result.bytes_alive_after[i] = ComputeBytes(live);
      for (auto def : defs)
        live.erase(def);
      live.insert(uses.begin(), uses.end());
      result.bytes_alive_before[i] = ComputeBytes(live);
    }
  }

  // TODO: does pytorch export similar function?
  std::size_t ComputeBytes(const torch::jit::TypePtr& _type) {
    if (auto type = _type->cast<torch::jit::TensorType>()) {
      auto numberOfElementsOpt = type->numel();
      auto scalarTypeOpt = type->scalarType();
      if (not numberOfElementsOpt or not scalarTypeOpt) {
        return 0;
      }
      auto elementSize = torch::elementSize(*scalarTypeOpt);
      return *numberOfElementsOpt * elementSize;
    }

    if (auto type = _type->cast<torch::jit::ListType>()) {
      std::size_t bytes = 0;
      for (auto& elemType : type->containedTypes()) {
        bytes += ComputeBytes(elemType);
      }
      return bytes;
    }

    if (auto type = _type->cast<torch::jit::TupleType>()) {
      std::size_t bytes = 0;
      for (auto& elemType : type->containedTypes()) {
        bytes += ComputeBytes(elemType);
      }
      return bytes;
    }

    // TODO: Should we handle more types?

    return 0;
  }

  std::size_t ComputeBytes(const torch::jit::Value* x) {
    return ComputeBytes(x->type());
  }

  std::size_t ComputeBytes(const std::set<const torch::jit::Value*>& xs) {
    std::size_t bytes = 0;
    for (auto x : xs) {
      bytes += ComputeBytes(x);
    }
    return bytes;
  }

  LivenessResult result;
  const torch::jit::Graph& graph_;
  const std::vector<const torch::jit::Node*>& schedule_;
};

} // namespace

LivenessResult AnalyseLiveness(
    const LazyJitGraph& lazy_graph,
    const std::vector<const torch::jit::Node*>& schedule) {
  return LivenessAnalysisImpl(lazy_graph, schedule).Run();
}

std::string LivenessResult::ToDebugString(
    const std::vector<const torch::jit::Node*>& schedule) const {
  const std::vector<const torch::jit::Node*>* annotations = nullptr;
  if (schedule.size() == bytes_alive_before.size()) {
    annotations = &schedule;
  }

  auto getAnnotation = [&](std::size_t index) {
    if (not annotations)
      return std::string();
    std::stringstream ss;
    auto node = (*annotations)[index];
    ss << " " << node->kind().toDisplayString();
    /*
    for (auto output : node->outputs()) {
      ss << " " << output->debugName();
    }
    */
    return ss.str();
  };

  auto getHumanBytes = [](std::size_t _bytes) {
    std::vector<std::string> suffixes = {"B", "KiB", "MiB", "GiB"};
    double bytes = _bytes;
    std::size_t i = 0;
    while (i < suffixes.size() and bytes >= 1024) {
      bytes /= 1024;
      i += 1;
    }

    char buff[128];
    snprintf(buff, 128, "% 6.3f %s", bytes, suffixes[i].c_str());
    return std::string(buff);
  };

  std::string ret;
  ret += "Liveness analysis:\n";
  for (std::size_t i = 0; i < bytes_alive_before.size(); ++i) {
    ret += std::to_string(i);
    ret += ". ";
    ret += getHumanBytes(bytes_alive_before[i]);
    ret += "\t->\t";
    ret += getHumanBytes(bytes_alive_after[i]);
    ret += "\t\t\t";
    ret += getAnnotation(i);
    ret += "\n";
  }
  return ret;
}

} // namespace utils
} // namespace program
} // namespace habana
