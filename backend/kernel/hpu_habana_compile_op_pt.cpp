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

#include "backend/kernel/hpu_habana_compile_op_pt.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/kernel/hpu_habana_execute_op_pt.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "backend/synapse_helpers/device_context.h"

namespace habana {

namespace HabanaLaunchOpPipeline {
void CompileSynapseTask(std::unique_ptr<habana::HabanaLaunchOpPT>&& launch_op) {
  auto compile_queue_length =
      HPUDeviceContext::compile_thread().get_active_task_count();
  LOP::emit_event_fast(
      true,
      "EagerCompileTask()",
      (int32_t)LOP::PipelineStageID::PIPELIE_STAGE_COMPILE_ID,
      compile_queue_length);
  bool sync_with_execute_stage = !launch_op->get_enable_4stage_pipeline();

  launch_op->CompileSynapse();

  auto graph_key_for_event = launch_op->get_graph_key();
  auto jit_cache_hit_count_for_event =
      launch_op->get_jit_graph_cache_hit_count();

  HPUDeviceContext::execute_thread().enqueue(
      HabanaLaunchOpPipeline::ExecuteSynapseTask, std::move(launch_op));

  if (sync_with_execute_stage)
    HPUDeviceContext::execute_thread().waitWorkComplete();

  LOP::emit_event_fast(
      false,
      "EagerCompileTask()",
      (int32_t)LOP::PipelineStageID::PIPELIE_STAGE_COMPILE_ID,
      compile_queue_length,
      graph_key_for_event,
      jit_cache_hit_count_for_event);
}
}; // namespace HabanaLaunchOpPipeline

std::shared_ptr<RecipeValueSpec> HabanaLaunchOpPT::
    CompileSynapseGraphAndPatchTable() {
  PT_BRIDGE_BEGIN;

  auto recipe = CompileSynapseGraph();
  auto rvs = std::make_shared<RecipeValueSpec>(jit_ir_graph_);

  ConstructPatchingTableAndAtenOutputs(*rvs, recipe);
  UpdateSynapsePermutations(*rvs, *recipe);
  PT_BRIDGE_DEBUG(*rvs);

  recipe_launcher_ = std::make_unique<RecipeLauncher>(*rvs, recipe);

  StoreCompiledInformation(rvs);

  PT_BRIDGE_END;
  return rvs;
}

void HabanaLaunchOpPT::CompileSynapse() {
  PT_BRIDGE_BEGIN;

  if (execution_control_.no_compile_) {
    return;
  }

  if (execution_control_.is_shape_agnostic_cache_hit_)
    recipe_launcher_->SetRecipe(CompileSynapseGraph());
  else
    CompileSynapseGraphAndPatchTable();

  PT_BRIDGE_END;
}
} // namespace habana