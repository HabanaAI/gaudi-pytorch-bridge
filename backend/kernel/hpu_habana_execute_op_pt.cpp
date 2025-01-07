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
void ExecuteSynapseTask(std::unique_ptr<habana::HabanaLaunchOpPT>&& launch_op) {
  auto execute_queue_length =
      HPUDeviceContext::execute_thread().get_active_task_count();
  const auto& op_name = launch_op->get_jit_graph_and_meta_data()->GetOpName();
  auto& device = habana::HPUDeviceContext::get_device();
  uint64_t device_queue_length = device.get_active_recipe_counter().get_count();
  LOP::ScopeEvent scope_event(
      "EagerExecuteTask()",
      op_name,
      (int32_t)LOP::PipelineStageID::PIPELIE_STAGE_EXECUTE_ID,
      launch_op->get_graph_key(),
      launch_op->get_jit_graph_cache_hit_count(),
      execute_queue_length,
      device_queue_length);

  launch_op->ExecuteSynapse();
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
      HPUDeviceContext::garbage_collection_thread().enqueue(
          SynapseGraphDestroyTask, std::move(graphHandle));
    }
  }

  ClearStatics();
  PT_BRIDGE_END;
}
} // namespace habana