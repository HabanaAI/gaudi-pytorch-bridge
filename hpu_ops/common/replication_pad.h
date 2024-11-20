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
#pragma once
#include "generated/backend/replication_pad1d.h"
#include "generated/backend/replication_pad1d_backward.h"
#include "generated/backend/replication_pad2d.h"
#include "generated/backend/replication_pad2d_backward.h"
#include "generated/backend/replication_pad3d.h"
#include "generated/backend/replication_pad3d_backward.h"

namespace habana {
enum PadType : int8_t { pad1D = 0, pad2D, pad3D };
sizes_vec ComputePadOutputShape(const at::Stack& stack, PadType padType);
std::shared_ptr<void> FillPadFwdBwdParams(
    const at::Stack& stack,
    PadType padType,
    size_t& size,
    bool backward);

} // namespace habana