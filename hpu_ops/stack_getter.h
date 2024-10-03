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

#include "hpu_ops/op_backend.h"

namespace habana {

struct TensorsPair {
  const at::Tensor& pt_t;
  synTensor syn_t;
  int syn_idx = -1; // In case it's needed to call SynInput instead of
                    // op->syn_in, we hold also syn_idx
};

class StackGetter {
 public:
  StackGetter(OpBackend* opIn, const at::Stack& stackIn, const char* labelIn)
      : op(opIn), stack(stackIn), label(labelIn) {}

  template <class T>
  auto getNextInput() {
    return getNextInputInternal((T*){});
  }

 private:
  size_t CheckGetAndIncrStackPos() {
    TORCH_CHECK(
        stackPos < stack.size(),
        label,
        " expected at least ",
        stackPos + 1,
        " args on stack but got ",
        stack.size())
    return stackPos++;
  }

  size_t GetAndIncrSynPos() {
    return synPos++;
  }

  void MoveSynPos(size_t offset) {
    synPos += offset;
  }

  OpBackend* op;
  const at::Stack& stack;

  size_t stackPos = 0;
  size_t synPos = 0;
  const char* label;

  c10::IValue getNextInputInternal(c10::IValue*) {
    auto pos = CheckGetAndIncrStackPos();
    if (stack[pos].isTensor()) {
      MoveSynPos(1);
    } else if (stack[pos].isTensorList()) {
      MoveSynPos(stack[pos].toTensorList().size());
    }
    return stack[pos];
  }

  TensorsPair getNextInputInternal(TensorsPair*) {
    auto pos = CheckGetAndIncrStackPos();
    TORCH_CHECK(
        stack[pos].isTensor(),
        "Input ",
        pos,
        " type expected to be ",
        "tensor");
    int syn_pos = GetAndIncrSynPos();
    return {stack[pos].toTensor(), op->syn_in(syn_pos), syn_pos};
  }

  c10::optional<TensorsPair> getNextInputInternal(c10::optional<TensorsPair>*) {
    auto pos = CheckGetAndIncrStackPos();
    TORCH_CHECK(
        stack[pos].isNone() || stack[pos].isTensor(),
        "Input ",
        pos,
        " type expected to be ",
        "none or tensor");
    if (stack[pos].isTensor()) {
      int syn_pos = GetAndIncrSynPos();
      return TensorsPair{stack[pos].toTensor(), op->syn_in(syn_pos), syn_pos};
    } else {
      return c10::optional<TensorsPair>{};
    }
  }

  c10::optional<std::vector<TensorsPair>> getNextInputInternal(
      c10::optional<std::vector<TensorsPair>>*) {
    auto pos = CheckGetAndIncrStackPos();
    c10::optional<c10::List<at::Tensor>> tensorList =
        stack[pos].toOptional<c10::List<at::Tensor>>();
    if (tensorList.has_value()) {
      c10::optional<std::vector<TensorsPair>> result =
          std::vector<TensorsPair>();
      for (auto&& v : tensorList.value()) {
        result.value().push_back({v, op->syn_in(GetAndIncrSynPos())});
      }
      return result;
    } else {
      return c10::optional<std::vector<TensorsPair>>{};
    }
  }

  std::vector<TensorsPair> getNextInputInternal(std::vector<TensorsPair>*) {
    auto pos = CheckGetAndIncrStackPos();
    TORCH_CHECK(
        stack[pos].isTensorList(),
        "Input ",
        pos,
        " type expected to be ",
        "tensor list");
    auto list = stack[pos].toTensorList();
    std::vector<TensorsPair> result;
    for (auto&& v : list) {
      result.push_back({v, op->syn_in(GetAndIncrSynPos())});
    }
    return result;
  }

  c10::optional<c10::ScalarType> getNextInputInternal(
      c10::optional<c10::ScalarType>*) {
    auto pos = CheckGetAndIncrStackPos();
    TORCH_CHECK(
        stack[pos].isNone() || stack[pos].isInt(),
        "Input ",
        pos,
        " type expected to be ",
        "none or ScalarType");
    return stack[pos].toOptional<at::ScalarType>();
  }

  std::variant<TensorsPair, c10::IValue> getNextInputInternal(
      std::variant<TensorsPair, c10::IValue>*) {
    auto pos = CheckGetAndIncrStackPos();
    if (stack[pos].isTensor()) {
      int syn_pos = GetAndIncrSynPos();
      return TensorsPair{stack[pos].toTensor(), op->syn_in(syn_pos)};
    } else {
      return stack[pos];
    }
  }

#define GET_NEXT_INPUT_INTERNAL(T, isFn, toFn, Tstr)                      \
  T getNextInputInternal(T*) {                                            \
    auto pos = CheckGetAndIncrStackPos();                                 \
    TORCH_CHECK(                                                          \
        stack[pos].isFn(), "Input ", pos, " type expected to be ", Tstr); \
    return stack[pos].toFn();                                             \
  }

  GET_NEXT_INPUT_INTERNAL(bool, isBool, toBool, "bool")
  GET_NEXT_INPUT_INTERNAL(double, isDouble, toDouble, "double")
  GET_NEXT_INPUT_INTERNAL(int, isInt, toInt, "int")
  GET_NEXT_INPUT_INTERNAL(c10::List<bool>, isBoolList, toBoolList, "bool array")
  GET_NEXT_INPUT_INTERNAL(c10::ScalarType, isInt, toScalarType, "ScalarType")
  GET_NEXT_INPUT_INTERNAL(
      std::vector<int64_t>,
      isIntList,
      toIntList().vec,
      "int list")
  GET_NEXT_INPUT_INTERNAL(c10::string_view, isString, toStringView, "string")
#undef GET_NEXT_INPUT_INTERNAL
};

} // namespace habana
