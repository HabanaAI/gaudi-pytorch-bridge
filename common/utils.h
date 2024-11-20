/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
