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

#include <utilities/exprtk.hpp>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

namespace habana {

typedef double exprtk_T;

typedef exprtk::symbol_table<exprtk_T> symbol_table_t;
typedef exprtk::expression<exprtk_T> expression_t;
typedef exprtk::parser<exprtk_T> parser_t;

typedef std::unordered_map<std::string, std::shared_ptr<exprtk_T>>
    SymbolValueMap;

class SymExpression {
  std::string m_expr_str;
  symbol_table_t m_symbol_table;
  expression_t m_expr_t;

 public:
  SymExpression(std::string e, SymbolValueMap& in_symbol_value_map);
  std::string& get_expr_str();
  int64_t eval();
  void dump_symbol_table();
};

class SizeExpression {
  std::string m_size_str;
  std::vector<SymExpression> m_size_expr;
  std::vector<std::string> tokenizer(std::string s);

 public:
  SizeExpression(){};
  SizeExpression(std::string size_str, SymbolValueMap& in_symbol_value_map);
  std::vector<SymExpression>& get_expressions();
  std::string get_size_expr_str();
};

class SymExprFactory {
  std::unordered_map<std::string_view, int64_t> expr_value_cache;
  SymExprFactory() {}
  ~SymExprFactory() {}

 public:
  SymExprFactory(const SymExprFactory&) = delete;
  SymExprFactory& operator=(const SymExprFactory&) = delete;

  static SymExprFactory& getInstance() {
    static SymExprFactory instance;
    return instance;
  }

  std::vector<int64_t> evaluate_symsize(
      std::shared_ptr<SizeExpression> size_expr);

  void clear_expr_cache();
};

} // namespace habana
