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

#include "hpu_ops/op_backend.h"

namespace habana {

struct TensorsPair {
  const at::Tensor& pt_t;
  synTensor syn_t;
  int syn_idx = -1; // In case it's needed to call SynInput instead of
                    // op->syn_in, we hold also syn_idx
};

template <class... Ts>
struct VariantWrapper {
  using Type = std::variant<Ts...>;

  VariantWrapper(const Type& src) : v(src) {}
  VariantWrapper(Type&& src) : v(std::move(src)) {}

#define ISTO(IS, TO, T)                  \
  bool IS() const {                      \
    return std::holds_alternative<T>(v); \
  }                                      \
  T TO()&& {                             \
    return std::get<T>(v);               \
  }                                      \
  T& TO()& {                             \
    return std::get<T>(v);               \
  }                                      \
  const T& TO() const& {                 \
    return std::get<T>(v);               \
  }

  ISTO(isIValue, toIValue, c10::IValue)
  ISTO(isTensorsPair, toTensorsPair, TensorsPair)
  ISTO(isDouble, toDouble, double)
  ISTO(isDoubleList, toDoubleVector, std::vector<double>)

#undef ISTO

  Type v;
};

class StackGetter {
 public:
  StackGetter(OpBackend* opIn, const at::Stack& stackIn, const char* labelIn)
      : op(opIn), stack(stackIn), label(labelIn) {}

  template <class T>
  auto getNextInput() {
    using namespace std::literals;
    return getNextInputInternal(""sv, (T*){});
  }

 private:
  void CheckStackPos() {
    TORCH_CHECK(
        stackPos < stack.size(),
        label,
        " expected at least ",
        stackPos + 1,
        " args on stack but got ",
        stack.size());
  }

  size_t GetAndIncrStackPos() {
    return stackPos++;
  }

  size_t GetAndIncrSynPos() {
    return synPos++;
  }

  size_t CheckAndGetStackPos() {
    CheckStackPos();
    return stackPos;
  }

  size_t CheckGetAndIncrStackPos() {
    CheckStackPos();
    return GetAndIncrStackPos();
  }

  OpBackend* op;
  const at::Stack& stack;

  size_t stackPos = 0;
  size_t synPos = 0;
  const char* label;

  TensorsPair getTensorsPair(const c10::IValue& ivalue) {
    int syn_pos = GetAndIncrSynPos();
    return {ivalue.toTensor(), op->syn_in(syn_pos), syn_pos};
  };

  // It is suitable only for IValue's not having syn_in() associated.
  // Otherwise consider std::variant.
  // Hybrid std::variant<types_with_syn_in..., IValue> is also possible.
  const c10::IValue& getNextInputInternal(std::string_view, c10::IValue*) {
    return stack[CheckGetAndIncrStackPos()];
  }

  TensorsPair getNextInputInternal(
      std::string_view orNoneStrOpt,
      TensorsPair*) {
    auto pos = CheckGetAndIncrStackPos();
    TORCH_CHECK(
        stack[pos].isTensor(),
        "Input ",
        pos,
        " type expected to be ",
        orNoneStrOpt,
        "tensor");
    return getTensorsPair(stack[pos]);
  }

  std::vector<TensorsPair> getNextInputInternal(
      std::string_view orNoneStrOpt,
      std::vector<TensorsPair>*) {
    auto pos = CheckGetAndIncrStackPos();
    TORCH_CHECK(
        stack[pos].isTensorList(),
        "Input ",
        pos,
        " type expected to be ",
        orNoneStrOpt,
        "tensor list");
    auto list = stack[pos].toTensorList();
    std::vector<TensorsPair> result;
    for (auto&& v : list) {
      int syn_pos = GetAndIncrSynPos();
      result.push_back({v, op->syn_in(syn_pos), syn_pos});
    }
    return result;
  }

#define MATCH_INPUT_INTERNAL_TO_TYPE_GENERIC(T, RT, isExpr, toExpr, Tstr) \
  bool valueMatchesType(const c10::IValue& ivalue, T*) {                  \
    return isExpr;                                                        \
  }                                                                       \
  RT valueToType(const c10::IValue& ivalue, T*) {                         \
    return toExpr;                                                        \
  }                                                                       \
  std::string_view typeToStr(T*) {                                        \
    return Tstr;                                                          \
  }

#define MATCH_INPUT_INTERNAL_TO_TYPE(T, isExpr, toExpr, Tstr) \
  MATCH_INPUT_INTERNAL_TO_TYPE_GENERIC(T, T, isExpr, toExpr, Tstr)

  MATCH_INPUT_INTERNAL_TO_TYPE(
      TensorsPair,
      ivalue.isTensor(),
      getTensorsPair(ivalue),
      "tensor")

  MATCH_INPUT_INTERNAL_TO_TYPE(
      std::monostate,
      ivalue.isNone(),
      (static_cast<void>(ivalue), std::monostate{}),
      "none")

  MATCH_INPUT_INTERNAL_TO_TYPE_GENERIC(
      c10::IValue,
      const c10::IValue&,
      (static_cast<void>(ivalue), true),
      ivalue,
      "ivalue")

#define GET_NEXT_INPUT_INTERNAL(T, isFn, toFn, Tstr)          \
  T getNextInputInternal(std::string_view orNoneStrOpt, T*) { \
    auto pos = CheckGetAndIncrStackPos();                     \
    TORCH_CHECK(                                              \
        stack[pos].isFn(),                                    \
        "Input ",                                             \
        pos,                                                  \
        " type expected to be ",                              \
        orNoneStrOpt,                                         \
        Tstr);                                                \
    return stack[pos].toFn();                                 \
  }                                                           \
                                                              \
  MATCH_INPUT_INTERNAL_TO_TYPE(T, ivalue.isFn(), ivalue.toFn(), Tstr)

  GET_NEXT_INPUT_INTERNAL(bool, isBool, toBool, "bool")
  GET_NEXT_INPUT_INTERNAL(int, isInt, toInt, "int")
  GET_NEXT_INPUT_INTERNAL(double, isDouble, toDouble, "double")
  GET_NEXT_INPUT_INTERNAL(c10::ScalarType, isInt, toScalarType, "ScalarType")
  GET_NEXT_INPUT_INTERNAL(c10::List<bool>, isBoolList, toBoolList, "bool array")
  GET_NEXT_INPUT_INTERNAL(
      std::vector<int64_t>,
      isIntList,
      toIntVector,
      "int list")
  GET_NEXT_INPUT_INTERNAL(
      std::vector<double>,
      isDoubleList,
      toDoubleVector,
      "double list")
  GET_NEXT_INPUT_INTERNAL(c10::string_view, isString, toStringView, "string")
#undef GET_NEXT_INPUT_INTERNAL

#undef MATCH_INPUT_INTERNAL_TO_TYPE
#undef MATCH_INPUT_INTERNAL_TO_TYPE_GENERIC

  template <class T>
  std::optional<T> getNextInputInternal(std::string_view, std::optional<T>*) {
    auto pos = CheckAndGetStackPos();
    if (stack[pos].isNone()) {
      GetAndIncrStackPos();
      return {};
    } else {
      return getNextInputInternal("none or ", (T*){});
    }
  }

  template <class T, class... Ts>
  std::string typesListToStr() {
    std::string s = std::string(typeToStr((T*){}));
    if constexpr (sizeof...(Ts)) {
      s += ", ";
      s += typesListToStr<Ts...>();
    }
    return s;
  }

  template <class... Ts>
  std::string typeToStr(std::variant<Ts...>*) {
    return typesListToStr<Ts...>();
  }

  template <class T, class... Ts, class RT>
  RT matchInputToTypeList(const c10::IValue& ivalue, RT* pRT) {
    static_assert(
        !std::is_same_v<T, c10::IValue> || (sizeof...(Ts) == 0),
        "IValue must be the last type on the types list");

    if (valueMatchesType(ivalue, (T*){}))
      return valueToType(ivalue, (T*){});

    if constexpr (sizeof...(Ts))
      return matchInputToTypeList<Ts...>(ivalue, pRT);

    std::string typesListStr = typeToStr(pRT);
    throw typesListStr;
  }

  template <class... Ts>
  VariantWrapper<Ts...> getNextInputInternal(
      std::string_view orNoneStrOpt,
      std::variant<Ts...>* pVarT) {
    auto pos = CheckGetAndIncrStackPos();
    try {
      return matchInputToTypeList<Ts...>(stack[pos], pVarT);
    } catch (const std::string& typesListStr) {
      TORCH_CHECK(
          false,
          "Input ",
          pos,
          " type expected to be ",
          orNoneStrOpt,
          "one of: ",
          typesListStr);
      throw std::bad_variant_access{};
    }
  }
};

} // namespace habana
