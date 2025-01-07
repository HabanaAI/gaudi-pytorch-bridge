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
  const auto& op_name = launch_op->get_jit_graph_and_meta_data()->GetOpName();
  auto& device = habana::HPUDeviceContext::get_device();
  uint64_t device_queue_length = device.get_active_recipe_counter().get_count();
  LOP::ScopeEvent scope_event(
      "EagerCompileTask()",
      op_name,
      (int32_t)LOP::PipelineStageID::PIPELIE_STAGE_COMPILE_ID,
      launch_op->get_graph_key(),
      launch_op->get_jit_graph_cache_hit_count(),
      compile_queue_length,
      device_queue_length);
  bool sync_with_execute_stage = !launch_op->get_enable_4stage_pipeline();

  launch_op->CompileSynapse();

  HPUDeviceContext::execute_thread().enqueue(
      HabanaLaunchOpPipeline::ExecuteSynapseTask, std::move(launch_op));

  if (sync_with_execute_stage)
    HPUDeviceContext::execute_thread().waitWorkComplete();
}
}; // namespace HabanaLaunchOpPipeline

std::shared_ptr<RecipeValueSpec> HabanaLaunchOpPT::
    CompileSynapseGraphAndPatchTable() {
  PT_BRIDGE_BEGIN;

  auto recipe = CompileSynapseGraph();
  auto rvs = std::make_shared<RecipeValueSpec>(jit_ir_graph_);

  rvs->curr_symval_hash_ = curr_symval_hash_;

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