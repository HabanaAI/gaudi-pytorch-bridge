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
#include "custom_op_outshape.h"

namespace habana {

CustomOpOutShapeFunRegistrar* CustomOpOutShapeFunRegistrar::pInstance = nullptr;
std::mutex CustomOpOutShapeFunRegistrar::mutexGetInstance;
std::mutex CustomOpOutShapeFunRegistrar::mutexRegister;

CustomOpOutShapeFunRegistrar& CustomOpOutShapeFunRegistrar::GetInstance() {
  std::lock_guard<std::mutex> lock(mutexGetInstance);
  if (!pInstance) {
    pInstance = new CustomOpOutShapeFunRegistrar();
  }
  return *pInstance;
}

#define ITEM(NAME, ARGS, ...)                               \
  void CustomOpOutShapeFunRegistrar::Register(              \
      const std::string_view opname, FunType_##NAME f) {    \
    std::lock_guard<std::mutex> lock(mutexRegister);        \
    map_##NAME.emplace(opname, f);                          \
  }                                                         \
                                                            \
  sym_sizes_vec CustomOpOutShapeFunRegistrar::CalcOutShape( \
      const std::string_view opname, __VA_ARGS__) const {   \
    return map_##NAME.at(opname) ARGS;                      \
  }
SUPPORTED_PROTO_LIST
#undef ITEM

} // namespace habana