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
#include <string_view>

// namespace habana_lazy
namespace habana_lazy {

void log_dev_mem_stats(
    std::string_view msg,
    std::string_view name = "",
    uint64_t size = 0);

const std::pair<uint64_t, uint32_t> get_future_memory();

} // namespace habana_lazy
