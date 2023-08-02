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
#include "executor.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "cache.h"
#include "habana_helpers/logging.h"
#include "splitter.h"
#include "strategy.h"

namespace habana {
namespace program {

Executor::Executor(ClusteredProgramSPtr program, Environment& env)
    : program_(std::move(program)), env_(env) {}

void Executor::Run(torch::jit::Stack& stack) {
  PT_BRIDGE_DEBUG("Running executor");
  // std::cout << "Running executor\n";
  std::unique_lock _lck(program_->mutex());

  PrepareFrames();
  ForwardProgramInputs(stack);

  auto goc = program_->GetGraphOfClusters();
  for (auto cluster_id : program_->GetSchedule()) {
    auto cluster = goc->FindCluster(cluster_id);
    RunCluster(*cluster);
  }
  ForwardProgramOutputs(stack);
  PT_BRIDGE_DEBUG("All executed");
  // std::cout << "All executed\n";
}

void Executor::RunCluster(Cluster& cluster) {
  PT_BRIDGE_DEBUG("Running cluster ", &cluster, " ", cluster.id_);
  // std::cout << "Running cluster " << cluster.id_ << "\n";

  auto& frame = frames_.at(cluster.id_);

  // Temporary work-around, TODO: clean this
  if (cluster.lazy_graph_exec_ == nullptr) {
    cluster.lazy_graph_exec_ = cluster.lazy_graph_;
    std::string cname =
        env_.op_name + "__cluster_" + std::to_string(cluster.id_);
    auto cluster_key = 0;
    cluster_key =
        at::hash_combine(env_.graph_hash, std::hash<std::string>()(cname));
    cluster_key = at::hash_combine(cluster_key, env_.graph_index);
    cluster.lazy_graph_exec_->set_cached_graph_key(cluster_key);
    cluster.lazy_graph_exec_->SetOpName(cname);
  }

  cluster.lazy_graph_exec_->SetHPUStream(env_.stream);
  habana::HabanaLaunchOpPT habana_launch_op_(cluster.lazy_graph_exec_);
  habana_launch_op_.run(frame.stack_);

  PropagateClusterOutputs(frame, cluster);
}

void Executor::PrepareFrames() {
  auto goc = program_->GetGraphOfClusters();
  // Allocate frames
  frames_[GraphOfClusters::SINK] = Frame();
  for (auto& p : goc->nodes_) {
    frames_[p.first] = Frame();
  }
}

void Executor::PropagateClusterOutputs(Frame& frame, Cluster& cluster) {
  for (std::size_t output_index = 0; output_index < cluster.outputs_.size();
       ++output_index) {
    for (auto& p : cluster.outputs_[output_index]) {
      auto& input_frame = frames_.at(p.cluster);
      input_frame.Assign(p.index, frame.stack_.at(output_index));
    }
  }
  frames_.erase(cluster.id_);
}

void Executor::ForwardProgramInputs(torch::jit::Stack& stack) {
  auto goc = program_->GetGraphOfClusters();
  auto num_inputs = goc->inputs_.size();
  for (std::size_t i = 0; i < num_inputs; ++i) {
    auto& targets = goc->inputs_[i];
    for (auto& target : targets) {
      frames_.at(target.cluster).Assign(target.index, stack.at(i));
    }
  }

  stack.resize(stack.size() - num_inputs);
}

void Executor::ForwardProgramOutputs(torch::jit::Stack& stack) {
  auto& output_frame = frames_.at(GraphOfClusters::SINK);
  for (auto& t : output_frame.stack_) {
    stack.emplace_back(std::move(t));
  }
  frames_.erase(GraphOfClusters::SINK);
}

void Executor::Frame::Assign(std::size_t index, at::IValue& value) {
  if (stack_.size() <= index) {
    stack_.resize(index + 1);
  }
  stack_.at(index) = value;
}

} // namespace program
} // namespace habana
