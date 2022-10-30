/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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

// This file will act as shim layer for reduce Op
//  which was impacted due to version upgrades.
#include <torch/csrc/api/include/torch/version.h>
#include <torch/library.h>
#undef UNUSED // Collision between pytorch_helpers/synapse_helpers/graph.h and
              // c10d::ReduceOp enum from c10d/Types.hpp
#if ((TORCH_VERSION_MAJOR == 1) && (TORCH_VERSION_MINOR < 13))
#include <c10d/Types.hpp>
using RedOpType = c10d::ReduceOp;
#else
#include <torch/csrc/distributed/c10d/Types.hpp>
using RedOpType = c10d::ReduceOp::RedOpType;
#endif
