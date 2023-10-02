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

void habana::HabanaCompile::CompileSynapse(
    synapse_helpers::hpuStream_t hpu_stream,
    bool is_shape_agnostic_cache_miss,
    std::shared_ptr<HabanaLaunchOpPT> hb_launch_op,
    size_t graph_key_with_perm,
    bool do_nothing_compile,
    bool do_nothing_execute,
    bool dry_run) {
  PT_BRIDGE_BEGIN;
  auto enqueue_execute_synapse =
      [&](synapse_helpers::hpuStream_t hpu_stream,
          std::shared_ptr<HabanaLaunchOpPT> hb_launch_op,
          bool is_shape_agnostic_cache_miss,
          bool do_nothing_execute,
          bool dry_run) {
        std::shared_ptr<HabanaExecute> habanaexecutor =
            std::make_shared<HabanaExecute>();
        habana_helpers::Singleton_ExecThreadPool::getInstance()
            .ScheduleWorkAndUpdateThreadHandle(
                habanaexecutor->ExecuteSynapse,
                std::move(hpu_stream),
                is_shape_agnostic_cache_miss,
                std::move(hb_launch_op),
                do_nothing_execute,
                dry_run);
      };

  auto is_enable_4stage_pipeline = hb_launch_op->get_enable_4stage_pipeline();

  if (do_nothing_compile && !do_nothing_execute) {
    // TODO : Move to HabanaExecute class and use single lambda to enqueue both
    // ExecuteSynapse as well as ExecuteSynapseCache
    std::shared_ptr<HabanaExecute> habanaexecutor =
        std::make_shared<HabanaExecute>();
    habana_helpers::Singleton_ExecThreadPool::getInstance()
        .ScheduleWorkAndUpdateThreadHandle(
            hb_launch_op->ExecuteSynapseCacheTask,
            graph_key_with_perm,
            std::move(hb_launch_op),
            dry_run);
    // TODO: Merge with is_pipeline_supported_ status flag
    if (!is_enable_4stage_pipeline) {
      habana_helpers::Singleton_ExecThreadPool::getInstance()
          .JoinPendingThread();
    }
    return;
  } else if (do_nothing_compile && do_nothing_execute) {
    enqueue_execute_synapse(
        hpu_stream, hb_launch_op, true, do_nothing_execute, dry_run);

    if (!is_enable_4stage_pipeline) {
      habana_helpers::Singleton_ExecThreadPool::getInstance()
          .JoinPendingThread();
    }
    return;
  }

  if (hb_launch_op->get_enable_shape_agnostic_caching_() &&
      hb_launch_op->get_jit_graph_and_meta_data()
          ->get_is_shape_agnostic_supported()) {
    if (is_shape_agnostic_cache_miss) {
      hb_launch_op->CompileSynapseGraph();
      hb_launch_op->StoreShapeAgnosticGraph();
      hb_launch_op->ConstructPatchingTableAndAtenOutputs();
      hb_launch_op->UpdateSynapsePermutations();
      hb_launch_op->StoreCompiledInformation(hpu_stream);
      enqueue_execute_synapse(
          hpu_stream, hb_launch_op, true, do_nothing_execute, dry_run);
      if (!is_enable_4stage_pipeline) {
        habana_helpers::Singleton_ExecThreadPool::getInstance()
            .JoinPendingThread();
      }

    } else {
      hb_launch_op->CompileSynapseGraph(false);
      enqueue_execute_synapse(
          hpu_stream, hb_launch_op, false, do_nothing_execute, dry_run);
      if (!is_enable_4stage_pipeline) {
        habana_helpers::Singleton_ExecThreadPool::getInstance()
            .JoinPendingThread();
      }
    }
  } else {
    hb_launch_op->CompileSynapseGraph();
    hb_launch_op->ConstructPatchingTableAndAtenOutputs();
    hb_launch_op->UpdateSynapsePermutations();
    hb_launch_op->StoreCompiledInformation(hpu_stream);
    enqueue_execute_synapse(
        hpu_stream, hb_launch_op, false, do_nothing_execute, dry_run);
    if (!is_enable_4stage_pipeline) {
      habana_helpers::Singleton_ExecThreadPool::getInstance()
          .JoinPendingThread();
    }
  }
  PT_BRIDGE_END;
}
} // namespace habana