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
#include "backend/helpers/eager_pipeline.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/kernel/hpu_habana_execute_op_pt.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"

namespace habana {

namespace HabanaLaunchOpPipeline {
void CompileSynapseTask(std::unique_ptr<habana::HabanaLaunchOpPT>&& launch_op) {
  bool sync_with_execute_stage = !launch_op->get_enable_4stage_pipeline();

  launch_op->CompileSynapse();

  habana_helpers::Singleton_ExecThreadPool::getInstance().Enqueue(
      HabanaLaunchOpPipeline::ExecuteSynapseTask, std::move(launch_op));

  if (sync_with_execute_stage)
    habana_helpers::Singleton_ExecThreadPool::getInstance().JoinPendingThread();
}
}; // namespace HabanaLaunchOpPipeline

void HabanaLaunchOpPT::CompileSynapse() {
  PT_BRIDGE_BEGIN;

  if (execution_control_.no_compile_) {
    return;
  }

  CompileSynapseGraph();

  if (get_enable_shape_agnostic_caching_() &&
      get_is_shape_agnostic_supported()) {
    if (execution_control_.is_shape_agnostic_cache_miss_) {
      StoreShapeAgnosticGraph();
      ConstructPatchingTableAndAtenOutputs();
      UpdateSynapsePermutations();
      StoreCompiledInformation();
      get_jit_graph_and_meta_data()->set_shape_agnostic_recipe(
          get_cur_rvalpsh());
    }
  } else {
    aten_outputs_ptr_sh_ = std::make_unique<VecOfIValPtrSh>();
    ConstructPatchingTableAndAtenOutputs();
    UpdateSynapsePermutations();
    StoreCompiledInformation();
  }
  PT_BRIDGE_END;
}
} // namespace habana