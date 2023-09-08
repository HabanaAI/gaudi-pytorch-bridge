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

namespace habana {

void habana::HabanaExecute::ExecuteSynapse(
    synapse_helpers::hpuStream_t hpu_stream,
    bool is_shape_agnostic_cache_miss,
    std::shared_ptr<HabanaLaunchOpPT> hb_launch_op,
    bool do_nothing_execute,
    bool dry_run) {
  PT_BRIDGE_BEGIN;
  if (do_nothing_execute) {
    PT_BRIDGE_END;
    return;
  }
  if (hb_launch_op->get_enable_shape_agnostic_caching_() &&
      hb_launch_op->get_jit_graph_and_meta_data()
          ->get_is_shape_agnostic_supported()) {
    if (is_shape_agnostic_cache_miss) {
      hb_launch_op->get_jit_graph_and_meta_data()->set_shape_agnostic_recipe(
          hb_launch_op->get_cur_rvalpsh());
      hb_launch_op->ExecuteSynapseGraph(hpu_stream);

      synGraphDestroy(
          hb_launch_op->syn_graph_ptr_->get_duplicate_graph_handle());
      PT_EAGER_DEBUG("[SHAPE AGNOSTIC] shape agnostic cache miss (end)");
    } else {
      RecipeValueSpec& rv = *hb_launch_op->get_cur_rvalpsh();
      hb_launch_op->ExecuteSynapseGraph(hpu_stream);

      synGraphDestroy(
          hb_launch_op->syn_graph_ptr_->get_duplicate_graph_handle());
      PT_EAGER_DEBUG("[SHAPE AGNOSTIC] shape agnostic cache hit (end)");
    }
  } else {
    hb_launch_op->ExecuteSynapseGraph(hpu_stream);
  }
  hb_launch_op->ClearStatics();
  PT_BRIDGE_END;
}
} // namespace habana