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
#pragma once

#include <mutex>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace at {
class Tensor;
}

namespace habana {

using sizes_vec = std::vector<std::vector<int64_t>>;

#define SUPPORTED_PROTO_LIST                 \
  ITEM(                                      \
      params_int,                            \
      (inputs, params),                      \
      const std::vector<at::Tensor>& inputs, \
      const std::vector<int>& params)

class CustomOpOutShapeFunRegistrar {
 public:
  CustomOpOutShapeFunRegistrar(CustomOpOutShapeFunRegistrar& other) = delete;
  void operator=(const CustomOpOutShapeFunRegistrar&) = delete;

  static CustomOpOutShapeFunRegistrar& GetInstance();

#define ITEM(NAME, ARGS, ...)                                   \
  typedef sizes_vec (*FunType_##NAME)(__VA_ARGS__);             \
  void Register(const std::string_view opname, FunType_##NAME); \
  sizes_vec CalcOutShape(const std::string_view opname, __VA_ARGS__) const;
  SUPPORTED_PROTO_LIST
#undef ITEM

 private:
  CustomOpOutShapeFunRegistrar() = default;

  static CustomOpOutShapeFunRegistrar* pInstance;
  static std::mutex mutexGetInstance;
  static std::mutex mutexRegister;

#define ITEM(NAME, ARGS, ...) \
  std::unordered_map<std::string_view, FunType_##NAME> map_##NAME;
  SUPPORTED_PROTO_LIST
#undef ITEM
};

#define REGISTER_CUSTOM_OP_OUTSHAPE_FUN(NAME, FUN) \
  RegisterCustomOpOutShapeFun out_shape_registrar_##NAME(#NAME, FUN)

struct RegisterCustomOpOutShapeFun {
  template <class Fun>
  RegisterCustomOpOutShapeFun(const std::string_view opname, Fun f) {
    CustomOpOutShapeFunRegistrar::GetInstance().Register(opname, f);
  }
};

} // namespace habana