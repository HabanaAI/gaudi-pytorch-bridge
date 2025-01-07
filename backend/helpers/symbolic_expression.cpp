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

#include "backend/helpers/symbolic_expression.h"
#include "habana_helpers/logging.h"

namespace habana {

parser_t parser;

std::string formatExpression(const std::string& input) {
  PT_BRIDGE_DEBUG("Expression before ", input);
  std::string result = input;

  // Replacing python power ** with C++ power ^
  std::string toReplace = "**";
  std::string replacement = "^";
  size_t pos = 0;
  while ((pos = result.find(toReplace, pos)) != std::string::npos) {
    result.replace(pos, toReplace.length(), replacement);
    pos += replacement.length();
  }

  // Replacing python ceiling with C++ ceil
  toReplace = "ceiling";
  replacement = "ceil";
  pos = 0;
  while ((pos = result.find(toReplace, pos)) != std::string::npos) {
    result.replace(pos, toReplace.length(), replacement);
    pos += replacement.length();
  }
  PT_BRIDGE_DEBUG("Expression after ", result);
  return result;
}

habana::SymExpression::SymExpression(
    std::string e,
    SymbolValueMap& in_symbol_value_map) {
  m_expr_str = formatExpression(e);

  for (auto it = in_symbol_value_map.begin(); it != in_symbol_value_map.end();
       it++) {
    m_symbol_table.add_variable(it->first, *(it->second));
    PT_BRIDGE_DEBUG(
        "SizeExpression m_expr_str:", it->first, ", value:", *(it->second));
  }

  m_expr_t.register_symbol_table(m_symbol_table);

  if (!parser.compile(m_expr_str, m_expr_t)) {
    TORCH_CHECK(0, "ExprtK expression Compilation error... ", m_expr_str);
  }
}

std::string& SymExpression::get_expr_str() {
  return m_expr_str;
}

int64_t SymExpression::eval() {
  dump_symbol_table();
  exprtk_T value = m_expr_t.value();
  int64_t result = static_cast<int64_t>(value);
  PT_BRIDGE_DEBUG("SymExpression eval result:", result);
  return result;
}

void SymExpression::dump_symbol_table() {
  std::vector<std::string> variable_list;
  m_symbol_table.get_variable_list(variable_list);
  for (const auto& variable_name : variable_list) {
    auto variable_ptr = m_symbol_table.get_variable(variable_name);
    PT_BRIDGE_DEBUG(
        "SymExpression Variable:", variable_name, " = ", variable_ptr->ref());
  }
}

std::vector<std::string> SizeExpression::tokenizer(std::string s) {
  std::vector<std::string> exprs;
  std::stringstream ss(s);
  std::string word;
  while (!ss.eof()) {
    std::getline(ss, word, ',');
    exprs.push_back(word);
  }
  return exprs;
}

SizeExpression::SizeExpression(
    std::string size_str,
    SymbolValueMap& in_symbol_value_map) {
  m_size_str = size_str;
  auto size_str_updated = m_size_str.substr(1, m_size_str.length() - 2);
  std::vector<std::string> exprs = tokenizer(size_str_updated);
  int64_t count = 0;
  for (auto expr_str : exprs) {
    m_size_expr.emplace_back(expr_str, in_symbol_value_map);
    count++;
  }
}

std::vector<SymExpression>& SizeExpression::get_expressions() {
  return m_size_expr;
}

std::string SizeExpression::get_size_expr_str() {
  return m_size_str;
}

std::vector<int64_t> SymExprFactory::evaluate_symsize(
    std::shared_ptr<SizeExpression> size_expr) {
  std::vector<int64_t> result;
  std::vector<SymExpression>& sym_exprs = size_expr->get_expressions();
  for (auto& sym_expr : sym_exprs) {
    auto sym_expr_str = sym_expr.get_expr_str();
    const auto& it = expr_value_cache.find(sym_expr_str);
    if (it != expr_value_cache.end()) {
      result.push_back(expr_value_cache[sym_expr_str]);
    } else {
      int64_t value = sym_expr.eval();
      expr_value_cache[sym_expr_str] = value;
      result.push_back(value);
    }
  }

  PT_BRIDGE_DEBUG("SizeExpression eval result:", result);
  return result;
}

void SymExprFactory::clear_expr_cache() {
  expr_value_cache.clear();
}

} // namespace habana