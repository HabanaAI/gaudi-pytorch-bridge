/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
