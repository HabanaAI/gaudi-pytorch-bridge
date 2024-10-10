/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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