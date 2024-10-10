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

#include "create_executor.h"
#include "backend/synapse_helpers/env_flags.h"
#include "components/cache.h"
#include "components/splitter.h"
#include "components/strategy.h"

namespace habana {
namespace program {

namespace {

void EnforceSplit(ClusteredProgram& program, Cluster* main_cluster) {
  auto goc = program.GetGraphOfClusters();
  auto split_strategy = GetSplittingStrategy();
  auto split_decision = split_strategy(*main_cluster->lazy_graph_);

  if (not split_decision.Validate(*main_cluster->lazy_graph_)) {
    PT_BRIDGE_FATAL("Strategy gave incorrect decision!");
    abort();
  }

  auto result = SplitJitIrGraph(*main_cluster->lazy_graph_, split_decision);

  PT_BRIDGE_DEBUG("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");

  auto subgoc = CreateGraphOfClustersFromSplittingResult(result);
  PT_BRIDGE_DEBUG("Subgoc: ", subgoc->ToDebugString());
  PT_BRIDGE_DEBUG(subgoc->ToDebugSummary());

  PT_BRIDGE_DEBUG("Expanding cluster ", main_cluster->id_);
  goc->ExpandCluster(main_cluster->id_, std::move(subgoc));

  auto sched_strategy = GetSchedulingStrategy();
  auto sched_decision = sched_strategy(*goc);
  program.SetSchedule(std::move(sched_decision.scheduling));

  PT_BRIDGE_DEBUG("Modified program: ", program.ToDebugString());
}

/*
 * Creates initial program for given graph from lazy tensor layer.
 * Initial program contains just one cluster, the initial graph.
 */
ClusteredProgramSPtr CreateInitialProgram(
    const std::shared_ptr<LazyJitGraph>& lazy_jit_graph) {
  auto goc = std::make_unique<GraphOfClusters>();
  auto main_cluster = goc->CreateCluster();
  main_cluster->lazy_graph_ = lazy_jit_graph;

  // Connect all top-level inputs to inputs of our single cluster
  auto param_node = lazy_jit_graph->get_cached_graph()->param_node();
  for (std::size_t i = 0; i < param_node->outputs().size(); ++i) {
    goc->inputs_.emplace_back();
    goc->inputs_.back().emplace_back(main_cluster->id_, i);
  }

  // Connect all single cluster outputs to top-level outputs
  auto return_node = lazy_jit_graph->get_cached_graph()->return_node();
  for (std::size_t i = 0; i < return_node->inputs().size(); ++i) {
    main_cluster->outputs_.emplace_back();
    main_cluster->outputs_.back().emplace_back(GraphOfClusters::SINK, i);
  }

  auto program = std::make_shared<ClusteredProgram>(std::move(goc));
  program->SetSchedule({main_cluster->id_});

  PT_BRIDGE_DEBUG("Initial program: ", program->ToDebugString());

  if (GET_ENV_FLAG_NEW(PT_HPU_CLUSTERED_PROGRAM_ENFORCE)) {
    EnforceSplit(*program, main_cluster);
  }
  return program;
}

} // namespace

std::unique_ptr<Executor> CreateExecutor(
    std::size_t graph_hash,
    const std::shared_ptr<LazyJitGraph>& lazy_jit_graph) {
  PT_BRIDGE_DEBUG(
      "looking up for program hash=",
      graph_hash,
      " ptr=",
      lazy_jit_graph.get());

  auto& programCache = Cache::GetInstance();

  auto program = programCache.Lookup(graph_hash);
  if (program == nullptr) {
    PT_BRIDGE_DEBUG("creating new program");
    program = CreateInitialProgram(lazy_jit_graph);
    programCache.Insert(graph_hash, program);
  }

  auto env = Environment{
      lazy_jit_graph->GetHPUStream(),
      lazy_jit_graph->GetOpName(),
      graph_hash,
      lazy_jit_graph->get_cached_graph_key(),
      lazy_jit_graph->GetGraphIndex()};
  auto result = std::make_unique<Executor>(std::move(program), env);
  return result;
}

} // namespace program
} // namespace habana
