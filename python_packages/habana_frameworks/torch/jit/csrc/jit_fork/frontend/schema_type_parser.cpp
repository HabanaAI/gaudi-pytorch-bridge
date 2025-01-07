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
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "jit_fork/frontend/schema_type_parser.h"

#include <ATen/core/alias_info.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/symbol.h>
#include <ATen/core/type_factory.h>
#include <c10/util/string_utils.h>
#include <torch/custom_class.h>

#include <algorithm>
#include <string>

#include "jit_fork/frontend/lexer.h"
#include "jit_fork/frontend/parse_string_literal.h"

using c10::AliasInfo;
using c10::AnyClassType;
using c10::AnyEnumType;
using c10::AnyType;
using c10::AwaitType;
using c10::BoolType;
using c10::CapsuleType;
using c10::ComplexType;
using c10::DeviceObjType;
using c10::DictType;
using c10::FloatType;
using c10::FutureType;
using c10::GeneratorType;
using c10::IntType;
using c10::LayoutType;
using c10::ListType;
using c10::MemoryFormatType;
using c10::NoneType;
using c10::NumberType;
using c10::QSchemeType;
using c10::QuantizerType;
using c10::RRefType;
using c10::ScalarTypeType;
using c10::StorageType;
using c10::StreamObjType;
using c10::StringType;
using c10::Symbol;
using c10::SymBoolType;
using c10::SymFloatType;
using c10::SymIntType;
using c10::TensorType;
using c10::TupleType;
using c10::UnionType;
using c10::VarType;

namespace habana_torch::jit {
TypeWrapper SchemaTypeParser::parseBaseType() {
  static std::unordered_map<std::string, TypePtr> type_map = {
      {"Generator", c10::TypeFactory::get<GeneratorType>()},
      {"Dimname", c10::TypeFactory::get<StringType>()},
      {"ScalarType", c10::TypeFactory::get<ScalarTypeType>()},
      {"Layout", c10::TypeFactory::get<LayoutType>()},
      {"MemoryFormat", c10::TypeFactory::get<MemoryFormatType>()},
      {"Storage", c10::TypeFactory::get<StorageType>()},
      {"QScheme", c10::TypeFactory::get<QSchemeType>()},
      {"Quantizer", c10::TypeFactory::get<QuantizerType>()},
      {"ConstQuantizerPtr",
       c10::TypeFactory::get<IntType>()}, // TODO This type should be removed
                                          // from the schema parser, it should
                                          // use the custom class mechanism
                                          // instead. @jerryzh
      {"Device", c10::TypeFactory::get<DeviceObjType>()},
      {"DeviceIndex", c10::TypeFactory::get<IntType>()},
      {"Stream", c10::TypeFactory::get<StreamObjType>()},
      {"Scalar", c10::TypeFactory::get<NumberType>()},
      {"str", c10::TypeFactory::get<StringType>()},
      {"float", c10::TypeFactory::get<FloatType>()},
      {"SymFloat", c10::TypeFactory::get<SymFloatType>()},
      {"complex", c10::TypeFactory::get<ComplexType>()},
      {"int", c10::TypeFactory::get<IntType>()},
      {"SymInt", c10::TypeFactory::get<SymIntType>()},
      {"bool", c10::TypeFactory::get<BoolType>()},
      {"SymBool", c10::TypeFactory::get<SymBoolType>()},
      {"None", c10::TypeFactory::get<NoneType>()},
      {"NoneType", c10::TypeFactory::get<NoneType>()},
      {"Capsule", c10::TypeFactory::get<CapsuleType>()},
      {"Any", c10::TypeFactory::get<AnyType>()},
      {"AnyClassType", c10::TypeFactory::get<AnyClassType>()},
      {"AnyEnumType", c10::TypeFactory::get<AnyEnumType>()},
  };
  auto tok = L.cur();
  if (!L.nextIf(TK_NONE) && !L.nextIf(TK_NONE_TYPE)) {
    L.expect(TK_IDENT);
  }
  std::string text = tok.text();

  auto it = type_map.find(text);
  if (it == type_map.end()) {
    if (!text.empty() && islower(text[0])) {
      // lower case identifiers that are not otherwise valid types
      // are treated as type variables
      return c10::TypeFactory::createNamed<VarType>(text);
    }
    throw ErrorReport(tok.range) << "unknown type specifier";
  }

  const c10::TypeKind type_kind = it->second->kind();
  const bool is_symbolic = type_kind == c10::TypeKind::SymIntType ||
      type_kind == c10::TypeKind::SymFloatType ||
      type_kind == c10::TypeKind::SymBoolType;
  const bool has_symbolic_info = L.cur().kind == '(';
  if (is_symbolic && has_symbolic_info) {
    // Three scenarios:
    // 1. ',' - Value as an argument and it is not last argument.
    // 2. '=' - Assignment operator
    // 3. '):' - Value as an argument and it is last argument.
    const auto simple_patterns = std::vector<int>({',', '='});
    const auto advanced_patterns =
        std::vector<std::pair<int, int>>{std::pair<int, int>(')', ':')};
    std::string symbol_or_expr = parseUntil(simple_patterns, advanced_patterns);
    if (symbol_or_expr[0] != '(' ||
        symbol_or_expr[symbol_or_expr.length() - 1] != ')') {
      throw ErrorReport(tok.range)
          << "Symbol or expression was not enclosed correctly by parenthesis";
    }

    const auto left_par_num =
        std::count(symbol_or_expr.begin(), symbol_or_expr.end(), '(');
    const auto right_par_num =
        std::count(symbol_or_expr.begin(), symbol_or_expr.end(), ')');
    if (left_par_num == right_par_num) {
      symbol_or_expr = symbol_or_expr.substr(1, symbol_or_expr.length() - 2);
    } else {
      throw ErrorReport(tok.range)
          << "Symbol or expression has mismatch in number of parenthesis";
    }
    return TypeWrapper(it->second, symbol_or_expr);
  }

  return TypeWrapper(it->second);
}

// Examples:
// Tensor(a) // Tensor is in set a
// Tensor(a!) // it is also written to
// Tensor!  // shorthand for Tensor(fresh_identifier!)
// Tensor(a! -> a|b) // Tensor is in set a, written to,
//                      and after the write is in set a AND b.
c10::optional<AliasInfo> SchemaTypeParser::parseAliasAnnotation() {
  AliasInfo alias_info;
  if (L.nextIf('(')) {
    // optional 'alias set annotation'
    parseList(TK_NOTHING, '|', TK_NOTHING, [&] {
      if (L.nextIf('*')) {
        alias_info.addBeforeSet(AliasInfo::wildcardSet());

        // If we found a wildcard, ignore all subsequent annotations
      } else if (!alias_info.isWildcardBefore()) {
        alias_info.addBeforeSet(
            Symbol::fromQualString("alias::" + L.expect(TK_IDENT).text()));
      }
    });
    if (L.nextIf('!')) {
      alias_info.setIsWrite(true);
    }
    if (L.nextIf(TK_ARROW)) {
      // optional 'alias set annotation'
      parseList(TK_NOTHING, '|', TK_NOTHING, [&] {
        if (L.nextIf('*')) {
          alias_info.addAfterSet(AliasInfo::wildcardSet());

          // If we found a wildcard, ignore all subsequent annotations
        } else if (!alias_info.isWildcardAfter()) {
          alias_info.addAfterSet(
              Symbol::fromQualString("alias::" + L.expect(TK_IDENT).text()));
        }
      });
    } else {
      // We didn't encounter an ->, so assume the "after set" is identical
      // to the "before set"
      HABANA_ASSERT(alias_info.afterSets().empty());
      for (const auto& set : alias_info.beforeSets()) {
        alias_info.addAfterSet(set);
      }
    }
    L.expect(')');
  } else if (L.nextIf('!')) {
    alias_info.addBeforeSet(
        Symbol::fromQualString("alias::$" + std::to_string(next_id++)));
    alias_info.setIsWrite(true);
  } else {
    return c10::nullopt;
  }

  return alias_info;
}

c10::optional<at::ScalarType> SchemaTypeParser::parseTensorDType(
    const std::string& dtype) {
#define DEFINE_SCALAR_TYPE(_1, n) {#n, at::ScalarType::n},

  static std::unordered_map<std::string, at::ScalarType> type_map = {
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_SCALAR_TYPE)};

  auto type = type_map.find(dtype);
  if (type != type_map.end()) {
    return type->second;
  }
  return c10::nullopt;
}

c10::optional<c10::Device> SchemaTypeParser::tryToParseDeviceType() {
  L.expect('=');
  const std::string& dev = L.expect(TK_IDENT).text();

  if (dev == "cpu") {
    return c10::Device(at::kCPU);
  }

  if (dev == "xpu") {
    c10::DeviceIndex device_idx = -1;
    if (L.cur().kind == ':') {
      L.expect(':');
      const std::string& num = L.expect(TK_NUMBER).text();
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      std::string::size_type num_len;
      try {
        device_idx = std::stoi(num, &num_len);
      } catch (const std::invalid_argument& e) {
        throw ErrorReport(L.cur().range)
            << "Device index cannot be converted to integer";
      } catch (const std::out_of_range& e) {
        throw ErrorReport(L.cur().range) << "Device index is too long";
      }
    }
    return c10::Device(at::kXPU, device_idx);
  }

  throw ErrorReport(L.cur().range)
      << "cannot parse device type '" << dev << "'\n";
}

c10::optional<bool> SchemaTypeParser::tryToParseRequiresGrad() {
  L.expect('=');
  const std::string& num = L.expect(TK_NUMBER).text();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::string::size_type num_len;

  try {
    return (bool)std::stoi(num, &num_len);
  } catch (const std::invalid_argument& e) {
    throw ErrorReport(L.cur().range)
        << "Field requires_grad cannot be converted to integer";
  } catch (const std::out_of_range& e) {
    throw ErrorReport(L.cur().range) << "Field requires_grad is too long";
  }
}

TypeWrapper SchemaTypeParser::parseRefinedTensor() {
  // Parse a type with either no ranks, known ranks with sizes, ranks with
  // unknown sizes, a mix of ranks with known and unknown sizes, or ranks with
  // known sizes and strides. The type might also have requires_grad and/or
  // device option. Examples of types we're handling here:
  //   Long(10, 8, 6, strides=[48, 6, 1], requires_grad=0, device=cuda:1)
  //   Float(10, *, 20, device=cuda:1)
  //   Float(requires_grad=1)
  TypePtr ptr;
  c10::optional<c10::Device> device;
  c10::optional<bool> requires_grad;
  SymbolicShape shape;
  SymbolicStrides strides;

  auto maybe_dtype = parseTensorDType(L.expect(TK_IDENT).text());
  HABANA_ASSERT(maybe_dtype);
  at::ScalarType dtype = *maybe_dtype;
  parseList('(', ',', ')', [&] {
    const std::string& field = L.expect(TK_IDENT).text();
    if (field == "device") {
      auto parsed_device = tryToParseDeviceType();
      if (parsed_device.has_value()) {
        if (device.has_value()) {
          throw ErrorReport(L.cur().range) << "'device' is specified twice";
        }
        device = parsed_device;
      }
      return;
    }
    if (field == "requires_grad") {
      auto parsed_requires_grad = tryToParseRequiresGrad();
      if (parsed_requires_grad.has_value()) {
        if (requires_grad.has_value()) {
          throw ErrorReport(L.cur().range)
              << "'requires_grad' is specified twice";
        }
        requires_grad = parsed_requires_grad;
      }
      return;
    }
    if (field == "strides") {
      L.expect('=');
      parseList('[', ',', ']', [&] {
        if (L.cur().kind == '*') {
          throw ErrorReport(L.cur().range)
              << "Strides with unknown values are not supported";
        } else if (L.cur().kind == TK_NUMBER) {
          const std::string& num = L.expect(TK_NUMBER).text();
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          std::string::size_type num_len;
          try {
            auto stride = std::stoll(num, &num_len);
            strides.push_back(stride);
          } catch (const std::invalid_argument& e) {
            throw ErrorReport(L.cur().range)
                << "The stride value cannot be converted to int";
          } catch (const std::out_of_range& e) {
            throw ErrorReport(L.cur().range) << "The stride is too big";
          }
        } else {
          const std::string symbol_or_expr =
              parseUntil(std::vector<int>{',', ']'});
          strides.push_back(symbol_or_expr);
        }
      });
      return;
    }
    if (field == "shape") {
      L.expect('=');
      parseList('[', ',', ']', [&] {
        if (L.cur().kind == '*') {
          throw ErrorReport(L.cur().range)
              << "Shape with unknown values is not supported";
        } else if (L.cur().kind == TK_IDENT && L.cur().text() == "SS") {
          throw ErrorReport(L.cur().range)
              << "Shape with old fashioned symbolic values is not supported";
        } else if (L.cur().kind == TK_NUMBER) {
          const std::string& num = L.expect(TK_NUMBER).text();
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          std::string::size_type num_len;
          int64_t dim = 0;
          try {
            dim = std::stoll(num, &num_len);
          } catch (const std::invalid_argument& e) {
            throw ErrorReport(L.cur().range)
                << "The number can't be converted to int";
          } catch (const std::out_of_range& e) {
            throw ErrorReport(L.cur().range) << "Number is too big";
          }
          shape.push_back(dim);
        } else {
          std::string symbol_or_expr = parseUntil(std::vector<int>{',', ']'});
          shape.push_back(symbol_or_expr);
        }
      });
      return;
    }
    throw ErrorReport(L.cur().range)
        << "Unexpected specifier '" << field << "'";
  });

  return TypeWrapper::createTensorTypeWrapper(
      dtype, shape, strides, device, requires_grad);
}

std::pair<TypeWrapper, c10::optional<AliasInfo>> SchemaTypeParser::parseType() {
  auto r = parseFakeAndRealType();
  return std::make_pair(std::move(std::get<0>(r)), std::move(std::get<2>(r)));
}

std::tuple<
    /*fake*/ TypeWrapper,
    /*real*/ TypeWrapper,
    c10::optional<AliasInfo>>
SchemaTypeParser::parseFakeAndRealType() {
  TypeWrapper fake_value;
  TypeWrapper real_value;
  c10::optional<AliasInfo> alias_info;
  // Tuple type
  if (L.cur().kind == '(') {
    std::vector<TypePtr> types;
    parseList('(', ',', ')', [&] {
      auto r = parseType();
      types.push_back(std::move(r.first.getType()));
      if (alias_info && r.second) {
        alias_info->addContainedType(std::move(*r.second));
      }
    });
    fake_value = real_value =
        c10::TypeFactory::create<TupleType>(std::move(types));
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "Future") {
    L.next(); // Future
    L.expect('(');
    auto p = parseType();
    auto subtype = std::move(p.first.getType());
    auto subalias = std::move(p.second);
    L.expect(')');
    fake_value = real_value = c10::TypeFactory::create<FutureType>(subtype);
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "Await") {
    L.next(); // Await
    L.expect('(');
    auto p = parseType();
    auto subtype = std::move(p.first.getType());
    auto subalias = std::move(p.second);
    L.expect(')');
    fake_value = real_value = c10::TypeFactory::create<AwaitType>(subtype);
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "RRef") {
    L.next(); // RRef
    L.expect('(');
    auto p = parseType();
    auto subtype = std::move(p.first.getType());
    auto subalias = std::move(p.second);
    L.expect(')');
    fake_value = real_value = c10::TypeFactory::create<RRefType>(subtype);
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "Tensor") {
    L.next();
    fake_value = real_value = c10::TypeFactory::get<TensorType>();
    alias_info = parseAliasAnnotation();
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "Dict") {
    L.next();
    L.expect('(');
    auto key_type = parseType().first.getType();
    L.expect(',');
    auto value_type = parseType().first.getType();
    L.expect(')');
    alias_info = parseAliasAnnotation();
    fake_value = real_value =
        c10::TypeFactory::create<DictType>(key_type, value_type);
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "Union") {
    L.next();
    L.expect('(');
    std::vector<TypePtr> types;
    types.emplace_back(parseType().first.getType());
    while (L.cur().kind != ')') {
      L.expect(',');
      types.emplace_back(parseType().first.getType());
    }
    L.expect(')');
    alias_info = parseAliasAnnotation();
    fake_value = real_value =
        c10::TypeFactory::create<c10::UnionType>(std::move(types));
  } else if (
      complete_tensor_types && L.cur().kind == TK_IDENT &&
      parseTensorDType(L.cur().text())) {
    fake_value = real_value = parseRefinedTensor();
    alias_info = parseAliasAnnotation();
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "__torch__") {
    L.next();
    L.expect('.');
    auto torch_tok = L.expect(TK_IDENT);
    if (torch_tok.text() != "torch") {
      throw ErrorReport(torch_tok.range)
          << "Expected classes namespace but got " << torch_tok.text();
    }
    L.expect('.');
    auto classes_tok = L.expect(TK_IDENT);
    if (classes_tok.text() != "classes") {
      throw ErrorReport(classes_tok.range)
          << "Expected classes namespace but got " << classes_tok.text();
    }
    L.expect('.');
    auto ns_tok = L.expect(TK_IDENT);
    L.expect('.');
    auto class_tok = L.expect(TK_IDENT);
    fake_value = real_value = torch::getCustomClass(
        std::string("__torch__.torch.classes.") + ns_tok.text() + "." +
        class_tok.text());
    if (!fake_value) {
      throw ErrorReport(class_tok.range)
          << "Unknown custom class type "
          << ns_tok.text() + "." + class_tok.text()
          << ". Please ensure it is registered.";
    }
  } else {
    real_value = parseBaseType();
    if (real_value->kind() == ScalarTypeType::Kind ||
        real_value->kind() == MemoryFormatType::Kind ||
        real_value->kind() == LayoutType::Kind) {
      fake_value = c10::TypeFactory::get<IntType>();
    } else {
      fake_value = real_value;
    }
    alias_info = parseAliasAnnotation();
  }
  while (true) {
    if (L.cur().kind == '[' && L.lookahead().kind == ']') {
      L.next(); // [
      L.next(); // ]
      fake_value = c10::TypeFactory::create<ListType>(fake_value.getType());
      real_value = c10::TypeFactory::create<ListType>(real_value.getType());
      auto container = parseAliasAnnotation();
      if (alias_info) {
        if (!container) {
          container = c10::optional<AliasInfo>(AliasInfo());
          container->setIsWrite(alias_info->isWrite());
        }
        container->addContainedType(std::move(*alias_info));
      }
      alias_info = std::move(container);
    } else if (L.nextIf('?')) {
      fake_value = c10::OptionalType::get(fake_value.getType());
      real_value = c10::OptionalType::get(real_value.getType());
    } else {
      break;
    }
  }
  return std::make_tuple(
      std::move(fake_value), std::move(real_value), std::move(alias_info));
}

void SchemaTypeParser::parseList(
    int begin,
    int sep,
    int end,
    c10::function_ref<void()> callback) {
  auto r = L.cur().range;
  if (begin != TK_NOTHING)
    L.expect(begin);
  if (L.cur().kind != end) {
    do {
      callback();
    } while (L.nextIf(sep));
  }
  if (end != TK_NOTHING)
    L.expect(end);
}

std::string SchemaTypeParser::parseUntil(
    const std::vector<int>& kinds,
    const std::vector<std::pair<int, int>>& following_kinds) {
  std::string parse_result;

  auto find_patterns = [&]() -> bool {
    const bool simple_pattern_found =
        std::find(kinds.cbegin(), kinds.cend(), L.cur().kind) != kinds.end();
    const bool advanced_pattern_found =
        std::find(
            following_kinds.cbegin(),
            following_kinds.cend(),
            std::pair<int, int>(L.cur().kind, L.lookahead().kind)) !=
        following_kinds.end();

    return simple_pattern_found || advanced_pattern_found;
  };

  while (!find_patterns()) {
    parse_result += L.next().text();
  };
  return parse_result;
}

std::string SchemaTypeParser::parseUntil(int kind) {
  return parseUntil(std::vector<int>{kind});
}

} // namespace habana_torch::jit
