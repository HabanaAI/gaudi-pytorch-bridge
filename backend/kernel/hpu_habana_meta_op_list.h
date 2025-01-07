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

#include <iostream>
#include <string>
#include <unordered_set>

namespace habana {

class HabanaMetaOpList {
 private:
  static const std::unordered_set<std::string_view>& meta_ops() {
    static const std::unordered_set<std::string_view> meta_ops_list = {
        // Add aten string here for ops to support
        // e.g  :: "aten::view"
        "aten::size",
        "prim::dtype",
    };
    return meta_ops_list;
  }

 public:
  static bool isHabanaMetaOp(const std::string_view op_name) {
    return (meta_ops().find(op_name) != meta_ops().end());
  }
};

} // namespace habana
