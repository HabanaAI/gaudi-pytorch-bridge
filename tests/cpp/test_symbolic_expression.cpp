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

#include <gtest/gtest.h>
#include "backend/helpers/symbolic_expression.h"

TEST(TestSymExpression, SizeExpression_tokenizer) {
  constexpr auto simple_expr = "s35 + s35*((-1) + s44";
  constexpr auto advance_expr = "max(25, s35 + s35*((-1) + s44)), 1024";

  auto one_token = habana::SizeExpression::tokenizer(simple_expr);
  ASSERT_EQ(simple_expr, one_token.at(0));

  constexpr auto expected_first_token{"max(25, s35 + s35*((-1) + s44))"};
  constexpr auto expected_second_token{" 1024"};
  auto two_tokens = habana::SizeExpression::tokenizer(advance_expr);
  ASSERT_EQ(expected_first_token, two_tokens.at(0));
  ASSERT_EQ(expected_second_token, two_tokens.at(1));
}

TEST(
    TestSymExpression,
    SymExpression_ExtractSymbolValueFromExpression_MultiplicationRight) {
  constexpr auto key = "8*S52";
  constexpr auto value = 32;

  auto result =
      habana::SymExpression::ExtractSymbolValueFromExpression(key, value);

  ASSERT_TRUE(result);
  if (result.has_value()) {
    ASSERT_EQ(result.value().first, "S52");
    ASSERT_EQ(result.value().second, 4);
  }
}

TEST(
    TestSymExpression,
    SymExpression_ExtractSymbolValueFromExpression_MultiplicationLeft) {
  constexpr auto key = "S4*220";
  constexpr auto value = 660;

  auto result =
      habana::SymExpression::ExtractSymbolValueFromExpression(key, value);

  ASSERT_TRUE(result);
  if (result.has_value()) {
    ASSERT_EQ(result.value().first, "S4");
    ASSERT_EQ(result.value().second, 3);
  }
}

TEST(
    TestSymExpression,
    SymExpression_ExtractSymbolValueFromExpression_DivisionLeft) {
  constexpr auto key = "200/S99";
  constexpr auto value = 20;

  auto result =
      habana::SymExpression::ExtractSymbolValueFromExpression(key, value);

  ASSERT_TRUE(result);
  if (result.has_value()) {
    ASSERT_EQ(result.value().first, "S99");
    ASSERT_EQ(result.value().second, 10);
  }
}

TEST(
    TestSymExpression,
    SymExpression_ExtractSymbolValueFromExpression_DivisionRight) {
  constexpr auto key = "S65/19";
  constexpr auto value = 8;

  auto result =
      habana::SymExpression::ExtractSymbolValueFromExpression(key, value);

  ASSERT_TRUE(result);
  if (result.has_value()) {
    ASSERT_EQ(result.value().first, "S65");
    ASSERT_EQ(result.value().second, 152);
  }
}

TEST(
    TestSymExpression,
    SymExpression_ExtractSymbolValueFromExpression_NoAction) {
  constexpr auto key = "S55";
  constexpr auto value = 32;

  auto result =
      habana::SymExpression::ExtractSymbolValueFromExpression(key, value);
  ASSERT_TRUE(!result);
}

TEST(
    TestSymExpression,
    SymExpression_ExtractSymbolValueFromExpression_InvalidMulArgSymbol) {
  constexpr auto key = "S55*a";
  constexpr auto value = 32;

  auto result =
      habana::SymExpression::ExtractSymbolValueFromExpression(key, value);
  ASSERT_TRUE(!result);
}

TEST(
    TestSymExpression,
    SymExpression_ExtractSymbolValueFromExpression_InvalidDivArgSymbol) {
  constexpr auto key = "a3/30";
  constexpr auto value = 32;

  auto result =
      habana::SymExpression::ExtractSymbolValueFromExpression(key, value);
  ASSERT_TRUE(!result);
}

TEST(
    TestSymExpression,
    SymExpression_ExtractSymbolValueFromExpression_InvalidOnlySymbols) {
  auto result_mul =
      habana::SymExpression::ExtractSymbolValueFromExpression("S55*S1", 32);
  auto result_div =
      habana::SymExpression::ExtractSymbolValueFromExpression("S55/S1", 32);
  ASSERT_TRUE(!result_mul);
  ASSERT_TRUE(!result_div);
}

TEST(
    TestSymExpression,
    SymExpression_ExtractSymbolValueFromExpression_InvalidOnlyValues) {
  auto result_mul =
      habana::SymExpression::ExtractSymbolValueFromExpression("32/16", 2);
  auto result_div =
      habana::SymExpression::ExtractSymbolValueFromExpression("8*4", 32);
  ASSERT_TRUE(!result_mul);
  ASSERT_TRUE(!result_div);
}
