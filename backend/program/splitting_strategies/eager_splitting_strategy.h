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

/*
 * Eager-mode like splitting strategy.
 *
 * Puts every op in separate cluster.
 */
SplittingDecision EagerSplittingStrategy(const LazyJitGraph& graph);

} // namespace program
} // namespace habana