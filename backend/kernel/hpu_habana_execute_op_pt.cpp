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

#include "backend/kernel/hpu_habana_execute_op_pt.h"
#include "backend/helpers/eager_pipeline.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "backend/synapse_helpers/device_context.h"

namespace habana {

namespace HabanaLaunchOpPipeline {
void ExecuteSynapseTask(std::unique_ptr<habana::HabanaLaunchOpPT>&& launch_op) {
  auto execute_queue_length =
      habana_helpers::Singleton_CompileThreadPool::getInstance()
          .get_number_of_active_tasks_in_queue();
  LOP::emit_event_fast(
      true,
      "EagerExecuteTask()",
      (int32_t)LOP::PipelineStageID::PIPELIE_STAGE_EXECUTE_ID,
      execute_queue_length);
  launch_op->ExecuteSynapse();
  auto& device = habana::HPURegistrar::get_device().syn_device();
  LOP::emit_event_fast(
      false,
      "EagerExecuteTask()",
      (int32_t)LOP::PipelineStageID::PIPELIE_STAGE_EXECUTE_ID,
      execute_queue_length,
      launch_op->get_graph_key(),
      launch_op->get_jit_graph_cache_hit_count(),
      device.get_active_recipe_counter().get_count());
}
} // namespace HabanaLaunchOpPipeline

namespace {
void SynapseGraphDestroyTask(synGraphHandle graphHandle) {
  PT_SYNHELPER_DEBUG("Graph destroy.");
  if (graphHandle != nullptr) {
    synGraphDestroy(graphHandle);
  }
}
} // namespace

void HabanaLaunchOpPT::ExecuteSynapse() {
  PT_BRIDGE_BEGIN;
  if (execution_control_.graph_key_with_perm_.has_value()) {
    ExecuteSynapseCache(execution_control_.graph_key_with_perm_.value());
    return;
  }

  ExecuteSynapseGraph();

  if (get_enable_shape_agnostic_caching_() &&
      get_is_shape_agnostic_supported()) {
    auto graphHandle = syn_graph_ptr_->get_graph_handle();
    if (graphHandle != nullptr) {
      syn_graph_ptr_->set_is_valid(false);
      habana_helpers::Singleton_GarbageCollectionThreadPool::getInstance()
          .Enqueue(SynapseGraphDestroyTask, std::move(graphHandle));
    }
  }

  ClearStatics();
  PT_BRIDGE_END;
}
} // namespace habana