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

#include <ATen/TypeDefault.h>

namespace common {

// This is not standard iterator with begin/end.
// It has different api to optimize compilation for case it is one-level list.
//
// Recommended usage:
// ListOfListsCustomIterator<T> customIt(list);
// if (!customIt.empty())
//   do {
//     auto tensors = customIt.get_next_item();
//   } while (customIt.has_more_items());
//
// When there is one level list only all conditions are constexprs and
// they are completely compiled out together with outside loop.
//
// For list-of-list case outside loop behaves like regular begin-end
// iteration.
template <class T>
class ListOfListsCustomIterator;

template <>
class ListOfListsCustomIterator<at::TensorList> {
 public:
  ListOfListsCustomIterator(at::TensorList listIn) : list(listIn) {}

  at::TensorList get_next_item() const {
    return list;
  }

  constexpr bool has_more_items() const {
    return false;
  }

  constexpr bool empty() const {
    return false;
  }

 private:
  at::TensorList list;
};

template <>
class ListOfListsCustomIterator<c10::ArrayRef<at::TensorList>> {
 public:
  ListOfListsCustomIterator(c10::ArrayRef<at::TensorList> listIn)
      : it(listIn.begin()), it_end(listIn.end()) {}

  at::TensorList get_next_item() {
    auto retval = *it;
    ++it;
    return retval;
  }

  bool has_more_items() const {
    return it != it_end;
  }

  bool empty() const {
    return !has_more_items();
  }

 private:
  c10::ArrayRef<at::TensorList>::iterator it;
  c10::ArrayRef<at::TensorList>::iterator it_end;
};

} // namespace common
