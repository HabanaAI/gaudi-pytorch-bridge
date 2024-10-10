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

#include <cstdint>
#include <string>

namespace habana_helpers {

void EnableInferenceMode();
void DisableInferenceMode();
bool IsInferenceMode();

void EnableQuantization();
void DisableQuantization();
bool IsQuantizationEnabled();

void EnableConstSectionSerialization(
    const char* path,
    bool clear_path,
    bool use_compression = false);
bool IsConstSectionSerialization();
std::string GetConstSectionSerializationPath();
bool ShouldClearConstSectionPath();
bool IsCompressionEnabled();

void EnableMatmul3d2dReshape();
void DisableMatmul3d2dReshape();
bool IsMatmul3d2dReshapeEnabled();
void enableRecomputeFSDPA(bool);
bool isRecomputeFSDPAEnabled();
} // namespace habana_helpers
