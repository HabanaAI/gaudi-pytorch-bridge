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
    bool is_shape_agnostic_cache_miss,
    std::shared_ptr<HabanaLaunchOpPT> hb_launch_op) {
  PT_BRIDGE_BEGIN;
  if (hb_launch_op->get_enable_shape_agnostic_caching_() &&
      hb_launch_op->get_is_shape_agnostic_supported()) {
    if (!is_shape_agnostic_cache_miss) {
      RecipeValueSpec& rv = *hb_launch_op->get_cur_rvalpsh();
      // SAG cache hit case - to avoid race condition with compile thread
      rv.recipe = hb_launch_op->get_hpu_op_recipe();
      rv.workspace_size = hb_launch_op->get_hpu_op_workspace_size();
      // SAG cache hit case - to avoid race condition with lowering thread
      rv.ntensorbytes = hb_launch_op->get_hpu_op_ntensorbytes();
    }
    hb_launch_op->ExecuteSynapseGraph();

    synGraphDestroy(hb_launch_op->syn_graph_ptr_->get_duplicate_graph_handle());
  } else {
    hb_launch_op->ExecuteSynapseGraph();
  }
  hb_launch_op->ClearStatics();
  PT_BRIDGE_END;
}
} // namespace habana