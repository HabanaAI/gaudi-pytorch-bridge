/**
 * Copyright (c) 2025 Intel Corporation
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

#include <any>
#include <cstddef>
#include <memory>

namespace habana {

namespace detail {
template <class T>
void* AnyCastToVoid(const std::any* operand) noexcept {
  return const_cast<T*>(std::any_cast<T>(operand));
}

void* SpCastToVoid(const std::any* operand) noexcept;
} // namespace detail

class FillParamsT {
 public:
  template <class T>
  static FillParamsT create() {
    FillParamsT ret;
    ret.params_ = std::make_any<T>();
    ret.ptrFun_ = &detail::AnyCastToVoid<T>;
    ret.size_ = sizeof(T);
    return ret;
  }

  static FillParamsT createForCustomOp(
      const std::shared_ptr<void>& params,
      size_t size) {
    FillParamsT ret;
    ret.params_ = params;
    ret.ptrFun_ = &detail::SpCastToVoid;
    ret.size_ = size;
    return ret;
  }

  template <class T>
  const T* paramsPtr() const {
    return std::any_cast<T>(&params_);
  }

  template <class T>
  T* paramsPtr() {
    return std::any_cast<T>(&params_);
  }

  void* ptr() const {
    return ptrFun_ ? ptrFun_(&params_) : nullptr;
  }

  size_t size() const {
    return size_;
  }

 private:
  std::any params_;
  decltype(detail::AnyCastToVoid<void>)* ptrFun_{nullptr};
  size_t size_{0};
};

} // namespace habana
