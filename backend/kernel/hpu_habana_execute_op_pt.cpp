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

#include "backend/kernel/hpu_habana_execute_op_pt.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "backend/synapse_helpers/device_context.h"

namespace habana {

namespace HabanaLaunchOpPipeline {
void ExecuteSynapseTaskWrapper(
    habana::HabanaLaunchOpPT& launch_op,
    absl::AnyInvocable<void(habana::HabanaLaunchOpPT&)>&& func) {
  auto& device = habana::HPUDeviceContext::get_device();
  uint64_t device_queue_length = device.get_active_recipe_counter().get_count();
  LOP::ScopeEvent scope_event(
      "EagerExecuteTask()",
      launch_op.get_jit_graph_and_meta_data()->GetOpName(),
      LOP::PipelineStageID::PIPELINE_STAGE_EXECUTE_ID,
      launch_op.get_graph_key(),
      launch_op.get_jit_graph_cache_hit_count(),
      HPUDeviceContext::execute_thread().get_active_task_count(),
      device_queue_length);

  if (func)
    func(launch_op);
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

void HabanaLaunchOpPT::RemoveDuplicateGraph() {
  auto graphHandle = syn_graph_ptr_->get_graph_handle();
  if (graphHandle != nullptr) {
    syn_graph_ptr_->set_is_valid(false);
    HPUDeviceContext::garbage_collection_thread().enqueue(
        SynapseGraphDestroyTask, std::move(graphHandle));
  }
}
} // namespace habana
