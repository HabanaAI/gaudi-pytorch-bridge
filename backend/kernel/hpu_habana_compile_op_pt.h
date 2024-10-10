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
#pragma once

#include "backend/helpers/tensor_utils.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"

namespace habana::HabanaLaunchOpPipeline {
void CompileSynapseTask(std::unique_ptr<habana::HabanaLaunchOpPT>&& launch_op);
}
