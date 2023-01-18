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
#include "executor.h"
#include "cache.h"
#include "habana_helpers/logging.h"
#include "splitter.h"
#include "strategy.h"

namespace habana {
namespace program {

Executor::Executor(ClusteredProgramSPtr program)
    : program_(std::move(program)) {}

void Executor::Run(torch::jit::Stack& stack) {
  PT_BRIDGE_WARN("Running executor");
  (void)stack;

  auto goc = program_->GetGraphOfClusters();
  for (auto cluster_id : program_->GetSchedule()) {
    auto cluster = goc->FindCluster(cluster_id);
    RunCluster(cluster);
  }
  PT_BRIDGE_WARN("All executed");
}

void Executor::RunCluster(Cluster* cluster) {
  PT_BRIDGE_WARN("Running cluster ", cluster, " ", cluster->id_);
}

namespace {

/*
 * Creates initial program for given graph from lazy tensor layer.
 * Initial program contains just one cluster, the initial graph.
 */
ClusteredProgramSPtr CreateInitialProgram(
    const std::shared_ptr<LazyJitGraph>& lazy_jit_graph) {
  (void)lazy_jit_graph;
  auto goc = std::make_unique<GraphOfClusters>();
  auto main_cluster = goc->CreateCluster();
  main_cluster->lazy_graph_ = lazy_jit_graph;
  auto program = std::make_shared<ClusteredProgram>(std::move(goc));
  program->SetSchedule({main_cluster->id_});

  // For testing
  PT_BRIDGE_WARN("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
  lazy_jit_graph->get_cached_graph()->print(std::cout, false);
  PT_BRIDGE_WARN("-----------------------------------------");
  auto strategy = GetSplittingStrategy();
  auto decision = strategy(*lazy_jit_graph);
  SplitJitIrGraph(*lazy_jit_graph, decision);
  PT_BRIDGE_WARN("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");

  return program;
}

} // namespace

std::unique_ptr<Executor> CreateExecutor(
    const std::shared_ptr<LazyJitGraph>& lazy_jit_graph) {
  auto key = lazy_jit_graph->get_cached_graph_key();
  PT_BRIDGE_WARN(
      "looking up for program key=", key, " ptr=", lazy_jit_graph.get());

  auto& programCache = Cache::GetInstance();

  auto program = programCache.Lookup(key);
  if (program == nullptr) {
    PT_BRIDGE_WARN("creating new program");
    program = CreateInitialProgram(lazy_jit_graph);
    programCache.Insert(key, program);
  }

  auto result = std::make_unique<Executor>(std::move(program));
  return result;
}

} // namespace program
} // namespace habana