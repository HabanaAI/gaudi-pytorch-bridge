/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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
