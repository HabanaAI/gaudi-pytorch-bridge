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

#include <c10/core/SymInt.h>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace at {
class Tensor;
}

namespace habana {

using sym_sizes_vec = std::vector<std::vector<c10::SymInt>>;

// All the below prototypes have to be added to PYBIND11 bindings:
// python_packages/habana_frameworks/torch/hpu/csrc/bindings.cpp
#define SUPPORTED_PROTO_LIST                                       \
  ITEM(no_params, (inputs), const std::vector<at::Tensor>& inputs) \
  ITEM(                                                            \
      params_int,                                                  \
      (inputs, params),                                            \
      const std::vector<at::Tensor>& inputs,                       \
      const std::vector<int64_t>& params)                          \
  ITEM(                                                            \
      params_float,                                                \
      (inputs, params),                                            \
      const std::vector<at::Tensor>& inputs,                       \
      const std::vector<float>& params)

class CustomOpOutShapeFunRegistrar {
 public:
  CustomOpOutShapeFunRegistrar(CustomOpOutShapeFunRegistrar& other) = delete;
  void operator=(const CustomOpOutShapeFunRegistrar&) = delete;

  static CustomOpOutShapeFunRegistrar& GetInstance();

#define ITEM(NAME, ARGS, ...)                                   \
  typedef sym_sizes_vec (*FunType_##NAME)(__VA_ARGS__);         \
  void Register(const std::string_view opname, FunType_##NAME); \
  sym_sizes_vec CalcOutShape(const std::string_view opname, __VA_ARGS__) const;
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