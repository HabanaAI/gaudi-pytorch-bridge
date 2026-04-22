/**
 * Copyright (c) 2021-2025 Intel Corporation
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
#include <absl/functional/any_invocable.h>
#include "backend/helpers/tensor_utils.h"
#include "backend/kernel/hpu_habana_execute_op_pt.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "backend/synapse_helpers/device_context.h"
#include "hpu_habana_launch_op_pt.h"

namespace habana {

namespace HabanaLaunchOpPipeline {
void CompileSynapseTaskWrapper(
    habana::HabanaLaunchOpPT& launch_op,
    absl::AnyInvocable<void(habana::HabanaLaunchOpPT&)>&& func) {
  auto& device = habana::HPUDeviceContext::get_device();
  uint64_t device_queue_length = device.get_active_recipe_counter().get_count();
  LOP::ScopeEvent scope_event(
      "EagerCompileTask()",
      launch_op.get_jit_graph_and_meta_data()->GetOpOrGraphName(),
      LOP::PipelineStageID::PIPELINE_STAGE_COMPILE_ID,
      launch_op.get_graph_key(),
      launch_op.get_jit_graph_cache_hit_count(),
      HPUDeviceContext::compile_thread_pool().get_active_task_count(),
      device_queue_length);

  if (func) {
    func(launch_op);
  }
}
}; // namespace HabanaLaunchOpPipeline

void HabanaLaunchOpPT::CompileSynapseGraphAndPatchTable(
    std::shared_ptr<RecipeValueSpec>& rvs) {
  PT_BRIDGE_BEGIN;
  auto recipe = CompileSynapseGraph();
  CreatePatchTable(rvs /**[in,out]*/, recipe);

  recipe_launcher_ = std::make_shared<RecipeLauncher>(*rvs, recipe);
  StoreCompiledInformation(rvs);

  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::CreatePatchTable(
    std::shared_ptr<RecipeValueSpec>& rvs,
    const std::shared_ptr<synapse_helpers::graph::recipe_handle>& recipe) {
  PT_BRIDGE_BEGIN;

  ConstructPatchingTableAndAtenOutputs(*rvs, recipe);
  UpdateSynapsePermutations(*rvs, recipe);
  PT_BRIDGE_DEBUG(*rvs);

  PT_BRIDGE_END;
}

std::shared_ptr<RecipeValueSpec> HabanaLaunchOpPT::CreateRVSAndPatchTable(
    const std::shared_ptr<synapse_helpers::graph::recipe_handle>& recipe) {
  auto rvs =
      std::make_shared<RecipeValueSpec>(jit_ir_graph_, curr_symval_hash_);

  CreatePatchTable(rvs /**[in,out]*/, recipe);

  return rvs;
}

} // namespace habana
