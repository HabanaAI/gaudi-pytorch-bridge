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
#include "../components/strategy.h"

namespace habana {
namespace program {
namespace utils {

/*
 * Utility to fix graph coloring to take into account some JIT IR constraints
 * induced by our lowering mechanism.
 *
 * This utility will analyse given `decision` and `graph` and recolor parts of
 * graphs. Recoloring is obtained by merging already existing colors. For
 * example, if we have edge A[color1] -> B[color2] and utility decide that A and
 * B have to be in the same partition then everything marked by one color will
 * be recolored to be marked with other color.
 *
 * If `fix_cycles` is to true, then procedure will internall call `FixCycles` at
 * end.
 */
void FixColors(
    SplittingDecision& decision,
    const torch::jit::Graph& graph,
    bool fix_cycles = true);

} // namespace utils
} // namespace program
} // namespace habana
