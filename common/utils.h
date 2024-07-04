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

// This file is intended to expose functions shared between Lazy and Eager,
// which contain separate implementations. Those implementation should go
// to lazy/ and eager/ sub-folders respectively.

#pragma once

namespace at {
class Tensor;
}

namespace common {
// Function to get underlying storage pointer.
// Lazy relies on lazy infra, and Eager can access directly storage of a Tensor.
void* GetDataPtrFromTensor(const at::Tensor& tensor);

bool IsStepMarkerSupported();

bool IsInt64Supported();

enum class LibraryType {
  LAZY, // libhabana_pytorch_plugin.so
  EAGER, // libhabana_pytorch2_plugin.so
};

// This function retuns specific LibraryType type based on loaded library.
LibraryType getLoadedLibraryType();

// StreamAllocator can be enabled only on eager
bool IsRecordStreamEnabled();
bool IsRecordStreamNoHolderEnabled();

} // namespace common
